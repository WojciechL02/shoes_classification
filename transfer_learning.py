import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomHorizontalFlip, RandomRotation
from cnn import test, save_plots


def train(model, device, dataloader, criterion, optimizer, scheduler, epoch):
    model.train()
    correct = 0
    loss_sum = 0.0
    n_batches = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        scheduler.step()
        n_batches += 1
    avg_loss = loss_sum / n_batches
    accuracy = 100 * correct / (n_batches * len(data))
    print(f"Epoch {epoch} | avg_loss: {avg_loss:.3f} | acc: {accuracy:.1f}")
    return avg_loss, accuracy


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    train_path = "data/train"
    test_path = "data/test"
    save_path = "tl_model.pth"

    mean = (0.6851, 0.6713, 0.6657)
    std = (0.3085, 0.3110, 0.3153)

    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    GAMMA = 0.96
    EPOCHS = 20

    transforms = Compose(
        [
            RandomRotation(degrees=20),
            RandomHorizontalFlip(p=0.3),
            ToTensor(),
            Normalize(mean, std)
        ]
    )

    train_dataset = ImageFolder(root=train_path, transform=transforms)
    test_dataset = ImageFolder(root=test_path, transform=transforms)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    # RESNET
    model_conv = models.resnet50(weights=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 3)
    )

    model_conv.fc = classifier
    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_conv.fc.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=GAMMA)

    best_acc = 0.0
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for epoch in range(1, EPOCHS+1):
        # TRAINING
        loss, acc = train(model_conv, device, train_loader, criterion, optimizer, lr_scheduler, epoch)
        train_acc.append(acc)
        train_loss.append(loss)
        # VALIDATION
        val_loss, val_acc = test(model_conv, device, test_loader, criterion)
        test_acc.append(val_acc)
        test_loss.append(val_loss)
        # SAVE MODEL
        if val_acc > best_acc:
            torch.save(model_conv.state_dict(), save_path)
            best_acc = val_acc

    save_plots(EPOCHS, train_acc, train_loss, test_acc, test_loss, "ResNet50")


if __name__ == "__main__":
    main()
