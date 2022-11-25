import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Recall, Precision, F1Score
import copy


def train(model, device, train_loader, criterion, optimizer, metrics, epoch):
    model.train()
    sum_loss = 0
    n_samples = 0
    n_batches = 0
    correct = 0

    recall_sum = 0
    precision_sum = 0
    f1_sum = 0
    for batch_id, (data, labels) in enumerate(train_loader):
        n_batches += 1
        n_samples += len(data)

        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(labels.view_as(pred)).sum().item()

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()


        sum_loss += loss.item()
        recall_sum += metrics["recall"](output, labels).item()
        precision_sum += metrics["precision"](output, labels).item()
        f1_sum += metrics["f1score"](output, labels).item()

    avg_loss = sum_loss / n_samples
    accuracy = 100 * correct / n_samples
    avg_recall = recall_sum / n_batches
    avg_precision = precision_sum / n_batches
    avg_f1 = f1_sum / n_batches

    print(f"Epoch={epoch}, avg_loss={avg_loss:.3f}, acc={accuracy:.2f}, recall={avg_recall:.2f}, precision={avg_precision:.2f}, f1={avg_f1:.2f}")
    return avg_loss, accuracy, avg_recall, avg_precision, avg_f1


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
    test_dataset = ImageFolder(root=TEST_PATH)
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
        loss, acc, recall, prec, f1 = train(model, device, train_loader, criterion, optimizer, metrics, epoch)
        train_acc.append(acc)
        train_loss.append(loss)
        train_recall.append(recall)
        train_precision.append(prec)
        train_f1score.append(f1)

        # v_loss, v_acc = validate(model, device, test_dataset, criterion)
        # val_acc.append(v_acc)
        # val_loss.append(v_loss)

        # if v_acc > best_val_acc:
        #     best_val_acc = v_acc
        #     best_model_wts = copy.deepcopy(model.state_dict())

    # torch.save(best_model_wts, "resnet.pth")

    print(train_acc)
    print(train_recall)
    print(train_precision)
    print(train_f1score)
    # draw_plots() TODO


if __name__ == "__main__":
    main()
