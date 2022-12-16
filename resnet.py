import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from train import train
from validate import validate
from metricsLogger import MetricsLogger


def main() -> None:
    TRAIN_PATH = "data/train"
    TEST_PATH = "data/test"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=10, shear=(-20, 20)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(size=224),
            transforms.ToTensor()
        ]
    )

    BATCH_SIZE = 64
    LEARNING_RATE = 0.1
    EPOCHS = 5

    train_dataset = ImageFolder(root=TRAIN_PATH, transform=t)
    test_dataset = ImageFolder(root=TEST_PATH, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset)


    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # freeze extractor weights
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_features=num_ftrs, out_features=3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)  # ustawic L2, momentum, nesterov
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.NAdam(model.parameters(), lr=LEARNING_RATE)

    metrics_logger = MetricsLogger(num_classes=3)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, criterion, optimizer, epoch, metrics_logger)
        validate(model, device, test_loader, criterion, epoch, metrics_logger)


if __name__ == "__main__":
    main()
