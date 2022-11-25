import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Recall, Precision, F1Score
import copy
from train import train
from validate import validate


def main():
    TRAIN_PATH = "data/train"
    TEST_PATH = "data/test"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 5

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
    test_dataset = ImageFolder(root=TEST_PATH, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset)


    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # freeze extractor weights
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_features=num_ftrs, out_features=3)
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    metrics = {"recall": Recall(num_classes=3), "precision": Precision(num_classes=3), "f1score": F1Score(num_classes=3)}

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    train_recall = []
    train_precision = []
    train_f1score = []
    best_val_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1):
        # loss, acc, recall, prec, f1 = train(model, device, train_loader, criterion, optimizer, metrics, epoch)
        # train_acc.append(acc)
        # train_loss.append(loss)
        # train_recall.append(recall)
        # train_precision.append(prec)
        # train_f1score.append(f1)

        v_loss, v_acc = validate(model, device, test_loader, criterion)
        val_acc.append(v_acc)
        val_loss.append(v_loss)

        # if v_acc > best_val_acc:
        #     best_val_acc = v_acc
        #     best_model_wts = copy.deepcopy(model.state_dict())

    # torch.save(best_model_wts, "resnet.pth")

    print(train_acc)
    print(train_recall)
    print(train_precision)
    print(train_f1score)


if __name__ == "__main__":
    main()
