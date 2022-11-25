import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


TRAIN_PATH = "data/train"
TEST_PATH = "data/test"
BATCH_SIZE = 16
LEARNING_RATE = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t = transforms.Compose(
    [
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees=10, shear=50),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.CenterCrop(size=200),
        transforms.ToTensor()
        # transforms.Normalize(mean, std)
    ]
)


train_dataset = ImageFolder(root=TRAIN_PATH, transform=t)
test_dataset = ImageFolder(root=TEST_PATH)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=3)
test_loader = DataLoader(test_dataset)


model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(in_features=num_ftrs, out_features=3)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    sum_loss = 0
    n_samples = 0
    correct = 0
    for batch_id, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        correct = pred.eq(label.view_as(pred)).sum().item()

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        n_samples += len(data)
        sum_loss += loss.item()

    avg_loss = sum_loss / n_samples
    accuracy = 100 * correct / n_samples

    print(f"Epoch={epoch}, avg_loss={avg_loss:.3f}, acc={accuracy:.2f}")
    return avg_loss, accuracy


def validate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += criterion(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)

    print(f"\tVAL: Loss={test_loss:.3f}, Acc={accuracy:.1f}")
    return test_loss, accuracy

