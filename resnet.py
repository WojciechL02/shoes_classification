import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torchvision.transforms as transforms


TRAIN_PATH = "data/train"
TEST_PATH = "data/test"


t = transforms.Compose(
    [
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees=10, shear=50),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.CenterCrop(size=200)
        # transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]
)


train_dataset = ImageFolder(root=TRAIN_PATH, transform=t)
test_dataset = ImageFolder(root=TEST_PATH)

train_dataset[15][0].show()




