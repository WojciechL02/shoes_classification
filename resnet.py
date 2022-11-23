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

train_loader = DataLoader(train_dataset, shuffle=True)
test_loader = DataLoader(test_dataset)


model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(in_features=num_ftrs, out_features=3)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(model, device, train_loader, criterion, optimizer, epoch):
    pass

