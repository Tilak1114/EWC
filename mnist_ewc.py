import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

# Helper function to create datasets
def create_datasets():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist = load_dataset('mnist')
    mnist_train = mnist['train']
    mnist_test = mnist['test']
    
    # Convert to 'RGB' before transformations
    train_data_A = [(transform(x['image'].convert("RGB")), x['label']) for x in mnist_train if x['label'] < 5]
    train_data_B = [(transform(x['image'].convert("RGB")), x['label']) for x in mnist_train if x['label'] >= 5]
    test_data_A = [(transform(x['image'].convert("RGB")), x['label']) for x in mnist_test if x['label'] < 5]
    test_data_B = [(transform(x['image'].convert("RGB")), x['label']) for x in mnist_test if x['label'] >= 5]
    
    collate_fn = lambda x: (torch.stack([i[0] for i in x]), torch.tensor([i[1] for i in x]))
    
    train_loader_A = DataLoader(train_data_A, batch_size=32, shuffle=True, collate_fn=collate_fn)
    train_loader_B = DataLoader(train_data_B, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader_A = DataLoader(test_data_A, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader_B = DataLoader(test_data_B, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return train_loader_A, train_loader_B, test_loader_A, test_loader_B

# Train the model on Task A
def train_task_A(model, optimizer, criterion, train_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.to('cpu')
    return model

# Evaluate the model
def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    model.to('cpu')
    return correct / total

# Compute the Fisher Information Matrix
def compute_fisher(model, data_loader, criterion):
    fisher = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p).to(device)
    
    model.eval()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for n, p in model.named_parameters():
            fisher[n] += p.grad.data ** 2 / len(data_loader)
    
    model.to('cpu')
    return fisher

# Compute the EWC loss
def ewc_loss(model, fisher, params_old, lambda_=5):
    loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for n, p in model.named_parameters():
        _loss = fisher[n].to(device) * (p - params_old[n].to(device)) ** 2
        loss += _loss.sum()
    return lambda_ * loss

# Fine-tune the model on Task B
def fine_tune_task_B(model, optimizer, criterion, fisher, params_old, train_loader, test_loader_A, test_loader_B, epochs=5, use_ewc=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accuracies_A = []
    accuracies_B = []

    for epoch in range(epochs):
        model.to(device)
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if use_ewc:
                loss += ewc_loss(model, fisher, params_old)
            loss.backward()
            optimizer.step()

        acc_A = evaluate_model(model, test_loader_A)
        acc_B = evaluate_model(model, test_loader_B)
        accuracies_A.append(acc_A)
        accuracies_B.append(acc_B)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Task A Accuracy: {acc_A:.4f}, Task B Accuracy: {acc_B:.4f}")

    model.to('cpu')
    return accuracies_A, accuracies_B

# Main script
def main():
    train_loader_A, train_loader_B, test_loader_A, test_loader_B = create_datasets()

    # Load a pre-trained model from timm
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Training on Task A...")
    model = train_task_A(model, optimizer, criterion, train_loader_A)

    # Save the trained model parameters and compute Fisher Information Matrix
    params_old = {n: p.clone() for n, p in model.named_parameters()}
    fisher = compute_fisher(model, train_loader_A, criterion)

    # Fine-tune on Task B without EWC
    model_no_ewc = copy.deepcopy(model)
    optimizer_no_ewc = optim.SGD(model_no_ewc.parameters(), lr=0.001)
    print("\nFine-tuning on Task B without EWC...")
    accuracies_A_no_ewc, accuracies_B_no_ewc = fine_tune_task_B(model_no_ewc, optimizer_no_ewc, criterion, fisher, params_old, train_loader_B, test_loader_A, test_loader_B, use_ewc=False)

    # Fine-tune on Task B with EWC
    model_ewc = copy.deepcopy(model)
    optimizer_ewc = optim.SGD(model_ewc.parameters(), lr=0.001)
    print("\nFine-tuning on Task B with EWC...")
    accuracies_A_ewc, accuracies_B_ewc = fine_tune_task_B(model_ewc, optimizer_ewc, criterion, fisher, params_old, train_loader_B, test_loader_A, test_loader_B, use_ewc=True)

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot for Task A Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(accuracies_A_no_ewc, label="Without EWC")
    plt.plot(accuracies_A_ewc, label="With EWC")
    plt.xlabel("Epochs")
    plt.ylabel("Task A Accuracy")
    plt.title("Task A Accuracy During Fine-Tuning on Task B")
    plt.legend()

    # Plot for Task B Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies_B_no_ewc, label="Without EWC")
    plt.plot(accuracies_B_ewc, label="With EWC")
    plt.xlabel("Epochs")
    plt.ylabel("Task B Accuracy")
    plt.title("Task B Accuracy During Fine-Tuning")
    plt.legend()
    plt.savefig("task_b_accuracy_during_fine_tuning.png")


if __name__ == "__main__":
    main()
