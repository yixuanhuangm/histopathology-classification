# CNN + ResNet on MedMNIST (PathMNIST)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from medmnist import INFO, Evaluator
import medmnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Set dataset name and download
DATA_FLAG = 'pathmnist'
DOWNLOAD = True
info = INFO[DATA_FLAG]
data_class = getattr(medmnist, info['python_class'])

# 2. Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# 3. Load datasets
train_dataset = data_class(split='train', transform=transform, download=DOWNLOAD)
val_dataset = data_class(split='val', transform=transform, download=DOWNLOAD)
test_dataset = data_class(split='test', transform=transform, download=DOWNLOAD)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

n_classes = len(info['label'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Define Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 5. Define ResNet
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 6. Training function
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 7. Evaluation function
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
            output = model(imgs)
            preds = output.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc, all_preds, all_labels

# 8. Main Experiment Loop
def run_experiment(model_class, model_name):
    print(f"\nTraining {model_name}...")
    model = model_class(n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    for epoch in range(5):
        loss = train(model, train_loader, optimizer, criterion)
        train_losses.append(loss)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    acc, preds, labels = evaluate(model, test_loader)
    print(f"{model_name} Test Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    print(f"\nClassification Report for {model_name}:")
    print(classification_report(labels, preds, digits=4))

    return train_losses

cnn_losses = run_experiment(SimpleCNN, "SimpleCNN")
resnet_losses = run_experiment(ResNet18, "ResNet18")

# 9. Plot training loss
plt.plot(cnn_losses, label='CNN Loss')
plt.plot(resnet_losses, label='ResNet18 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()