import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomHorizontalFlip, RandomRotation
from matplotlib import pyplot as plt
from matplotlib import figure


class Net(nn.Module): # Resnet50 z przetrenowanymi wagami
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.b_norm2d = nn.BatchNorm2d(16)
        self.b_norm1d = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(58*58*32, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.b_norm2d(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.b_norm1d(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    sum_loss = 0
    n_batches = 0
    correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        n_batches += 1
        sum_loss += loss.item()
    avg_loss = sum_loss / n_batches
    accuracy = 100 * correct / (n_batches * len(data))
    print(f"Epoch={epoch}, avg_loss={avg_loss:.3f}, acc={accuracy:.1f}")
    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    n_batches = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            n_batches += 1

    test_loss /= n_batches
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f"\tTEST: Loss={test_loss:.3f}, Acc={accuracy:.1f}")
    return test_loss, accuracy


def get_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def save_plots(epochs, train_acc, train_loss, test_acc, test_loss, model_type):
    plt.plot(range(1, epochs+1), train_acc)
    plt.plot(range(1, epochs+1), test_acc)
    plt.legend(["Train", "Test"])
    plt.title(f"{model_type} accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    fig1 = plt.gcf()
    fig1.savefig(f"plots/{model_type}_acc.png")
    plt.clf()
    plt.plot(range(1, epochs+1), train_loss)
    plt.plot(range(1, epochs+1), test_loss)
    plt.legend(["Train", "Test"])
    plt.title(f"{model_type} loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    fig2 = plt.gcf()
    fig2.savefig(f"plots/{model_type}_loss.png")


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    train_path = "data/train"
    test_path = "data/test"
    save_path = "cnn/cnn_model.pth"

    mean = (0.6851, 0.6713, 0.6657)
    std = (0.3085, 0.3110, 0.3153)

    BATCH_SIZE = 16
    LEARNING_RATE = 0.0018
    GAMMA = 1e-5
    EPOCHS = 50

    transforms = Compose(
        [
            # CenterCrop(size=220),
            # Resize(size=200),
            RandomRotation(degrees=20),
            RandomHorizontalFlip(p=0.5),# color jitter shear
            ToTensor(),
            Normalize(mean, std)
        ]
    )

    train_dataset = ImageFolder(root=train_path, transform=transforms)
    test_dataset = ImageFolder(root=test_path)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=1e-5)
    # scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=GAMMA)

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    for epoch in range(1, EPOCHS+1):
        loss, acc = train(model, device, train_loader, criterion, optimizer, epoch)
        train_acc.append(acc)
        train_loss.append(loss)

        val_loss, val_acc = test(model, device, test_loader, criterion)
        test_acc.append(val_acc)
        test_loss.append(val_loss)
        # scheduler.step()

    if max(test_acc) > 55:
        torch.save(model.state_dict(), save_path)
        print("MODEL SAVED")

    save_plots(EPOCHS, train_acc, train_loss, test_acc, test_loss, "cnn")


if __name__ == "__main__":
    main()
# recall, precison, F1, AUC, Confusion MAtrix, TP, Tn, FP, FN