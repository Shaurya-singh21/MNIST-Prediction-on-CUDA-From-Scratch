import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    train_transfrom = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
            transforms.RandomHorizontalFlip(0.6),
            transforms.RandomRotation(degrees=120),
        ]
    )
    test_val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
        ]
    )
    train_data = datasets.MNIST(
        root="data/raw", download=True, transform=train_transfrom, train=True
    )
    full_test_data = datasets.MNIST(
        root="data/raw", download=True, train=False, transform=test_val_transform
    )
    test_data, val_data = random_split(
        full_test_data, [int(len(full_test_data) * 0.6), int(len(full_test_data) * 0.4)]
    )
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size: {len(test_data)}")
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)
    return train_loader, test_loader, val_loader


class MNIST_model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, X):
        X = torch.flatten(X, start_dim=1)
        X = self.MLP(X)
        return X

    def loss_func(self):
        return nn.CrossEntropyLoss()


def train_model(model, train_loader, test_loader):
    my_model = model.to(device)
    epochs = 5
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0001)
    loss_fn = my_model.loss_func()

    train_loss_per_cycle = []
    test_loss_per_cycle = []
    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        my_model.train()
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = my_model(X_train)

            # backpropagation
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_train)
            loss.backward()

            # update the weights
            optimizer.step()

            train_loss += loss.item()
        my_model.eval()
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_pred = my_model(X_test)
                loss = loss_fn(y_pred, y_test)
                test_loss += loss.item()

        train_loss_per_cycle.append(train_loss / len(train_loader))
        test_loss_per_cycle.append(test_loss / len(test_loader))
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_per_cycle[-1]:.4f}, Test Loss: {test_loss_per_cycle[-1]:.4f}"
        )
    return train_loss_per_cycle, test_loss_per_cycle


def evaluate_model(model, val_loader):
    predicted = []
    actual = []
    cnt = 0
    model.to(device).eval()
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_val_pred = model(X_val)
            y_val_pred = torch.argmax(y_val_pred, dim=1).cpu().numpy()
            # print(f"y_val_pred for {cnt} is:", y_val_pred)
            predicted.append(y_val_pred)
            actual.append(y_val.cpu().numpy())
            cnt += 1
        predicted_flat = np.concatenate(predicted, axis=0)
        actual_flat = np.concatenate(actual, axis=0)
        print("VALIDATION RESULTS")
        classification_report_str = classification_report(
            np.array(actual_flat),
            np.array(predicted_flat),
            labels=[i for i in range(10)],
        )
        confusion = confusion_matrix(np.array(actual_flat), np.array(predicted_flat))
        return confusion, classification_report_str


def main():
    input_size = 28 * 28
    print("Loading data...")
    train_loader, test_loader, val_loader = load_data()
    print("Data loaded. Starting training...")
    model = MNIST_model(input_size)
    train_loss, test_loss = train_model(model, train_loader, test_loader)
    print("Training complete.")
    confusion, classification_report_str = evaluate_model(model, val_loader)
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(classification_report_str)
    print("Evaluation complete.")
    torch.save(model.state_dict(), "mnist_model.pth")


if __name__ == "__main__":
    main()
