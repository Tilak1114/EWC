import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split


# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Helper function to create datasets


def create_datasets():
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.2, random_state=0)
    X_circles, y_circles = make_circles(
        n_samples=1000, noise=0.2, factor=0.5, random_state=1)

    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_moons, y_moons, test_size=0.3, random_state=0)
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
        X_circles, y_circles, test_size=0.3, random_state=1)

    return (X_train_A, y_train_A, X_test_A, y_test_A), (X_train_B, y_train_B, X_test_B, y_test_B)


def plot_data(dataset, title, filename):
    (X_train, y_train, X_test, y_test) = dataset

    plt.figure(figsize=(12, 6))

    # Plotting training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, cmap='viridis', marker='.')
    plt.title(f'Training Data: {title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plotting test data
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1],
                c=y_test, cmap='viridis', marker='.')
    plt.title(f'Test Data: {title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.tight_layout()

    plt.savefig(filename)
    plt.close()

# Train the model on Task A


def train_task_A(model, optimizer, criterion, X_train, y_train, epochs=100):
    # Check if CUDA is available and move the model to GPU if it is
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Convert data to tensors and move to the appropriate device
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    
    for epoch in range(epochs):
        model.train()
        inputs = X_train
        labels = y_train

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Move the model back to CPU if it was on GPU to align with the rest of the script
    model.to('cpu')
    return model

# Evaluate the model


def evaluate_model(model, X_test, y_test):
    model.eval()
    inputs = torch.Tensor(X_test)
    labels = torch.LongTensor(y_test)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)
    return accuracy

# Compute the Fisher Information Matrix


def compute_fisher(model, data_loader, criterion):
    fisher = {}
    device = next(model.parameters()).device  # Get the device of the model
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p).to(device)  # Ensure tensor is on the same device as the model
    model.eval()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for n, p in model.named_parameters():
            fisher[n] += (p.grad.data ** 2).to(device) / len(data_loader)  # Ensure gradient is on the same device

    return fisher

# Compute the EWC loss


def ewc_loss(model, fisher, params_old, lambda_=0.99):
    loss = 0
    
    # Ensure params_old and fisher are on the same device as the model parameters
    device = next(model.parameters()).device
    params_old = {n: p.to(device) for n, p in params_old.items()}
    fisher = {n: f.to(device) for n, f in fisher.items()}
    
    for n, p in model.named_parameters():
        _loss = fisher[n] * (p - params_old[n]) ** 2
        loss += _loss.sum()
    return lambda_ * loss

# Fine-tune the model on Task B


def fine_tune_task_B(model, optimizer, criterion, fisher, params_old, X_train, y_train, X_test_A, y_test_A, X_test_B, y_test_B, epochs=100, use_ewc=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = [(torch.Tensor(X_train[i:i+10]).to(device),
        torch.LongTensor(y_train[i:i+10]).to(device)) for i in range(0, len(X_train), 10)]

    accuracies_A = []
    accuracies_B = []

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if use_ewc:
                loss += ewc_loss(model, fisher, params_old)
            loss.backward()
            optimizer.step()

        acc_A = evaluate_model(model.to('cpu'), X_test_A, y_test_A)
        acc_B = evaluate_model(model.to('cpu'), X_test_B, y_test_B)

        model.to(device)
        
        accuracies_A.append(acc_A)
        accuracies_B.append(acc_B)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Task A Accuracy: {acc_A:.4f}, Task B Accuracy: {acc_B:.4f}")

    model.to('cpu')
    return accuracies_A, accuracies_B


def main():
    # Main script
    dataset_A, dataset_B = create_datasets()
    (X_train_A, y_train_A, X_test_A, y_test_A), (X_train_B,
                                                 y_train_B, X_test_B, y_test_B) = dataset_A, dataset_B

    # Visualize the data
    plot_data(dataset_A, title='Dataset A', filename=f'moons.png')
    plot_data(dataset_B, title='Dataset B', filename=f'circles.png')

    # Training on Task A
    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Training on Task A...")
    model = train_task_A(model, optimizer, criterion, X_train_A, y_train_A)

    # Save the trained model parameters and compute Fisher Information Matrix
    params_old = {n: p.clone() for n, p in model.named_parameters()}
    data_loader_A = [(torch.Tensor(X_train_A[i:i+10]), 
        torch.LongTensor(y_train_A[i:i+10])) for i in range(0, len(X_train_A), 10)]
    fisher = compute_fisher(model, data_loader_A, criterion)

    # Fine-tune on Task B without EWC
    model_no_ewc = copy.deepcopy(model)
    optimizer_no_ewc = optim.SGD(model_no_ewc.parameters(), lr=0.0001)
    print("\nFine-tuning on Task B without EWC...")
    accuracies_A_no_ewc, accuracies_B_no_ewc = fine_tune_task_B(
        model_no_ewc, optimizer_no_ewc, criterion,
        fisher, params_old,
        X_train_B, y_train_B,
        X_test_A, y_test_A,
        X_test_B, y_test_B,
        use_ewc=False
    )

    # Fine-tune on Task B with EWC
    model_ewc = copy.deepcopy(model)
    optimizer_ewc = optim.SGD(model_ewc.parameters(), lr=0.0001)
    print("\nFine-tuning on Task B with EWC...")
    accuracies_A_ewc, accuracies_B_ewc = fine_tune_task_B(
        model_ewc, optimizer_ewc, criterion,
        fisher, params_old,
        X_train_B, y_train_B,
        X_test_A, y_test_A,
        X_test_B, y_test_B,
        use_ewc=True
    )

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
    plt.savefig("task_b_accuracy_during_fine_tuning_simple_cnn.png")

    # Close the plot to avoid displaying it
    plt.close()


if __name__ == "__main__":
    main()
