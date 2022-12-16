import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from train import train
from validate import validate
from metricsLogger import MetricsLogger


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding='same')
        # self.b_norm2d = nn.BatchNorm2d(16)
        # self.b_norm1d = nn.BatchNorm1d(num_features=1024)
        # self.dropout1 = nn.Dropout(0.3)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    train_path = "data/train"
    test_path = "data/test"
    save_path = "cnn/cnn_model.pth"

    t = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=10, shear=(-20, 20)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(size=224),
            transforms.ToTensor()
        ]
    )

    test_t = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    )

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    GAMMA = 1e-5
    EPOCHS = 20

    train_dataset = ImageFolder(root=train_path, transform=t)
    test_dataset = ImageFolder(root=test_path, transform=test_t)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=GAMMA)

    metrics_logger = MetricsLogger(device, model_name="CNN", num_classes=3)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, criterion, optimizer, epoch, metrics_logger)
        validate(model, device, test_loader, criterion, epoch, metrics_logger)
        # scheduler.step()


if __name__ == "__main__":
    main()
# recall, precison, F1, AUC, Confusion MAtrix, TP, Tn, FP, FN
