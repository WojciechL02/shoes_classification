import torch


def validate(model, device, test_loader, criterion, epoch, logger):
    model.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            running_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(labels.view_as(pred)).sum().item()

            if batch_id == len(test_loader) - 1:
                logger.log_epoch(output, labels, running_loss, epoch, training=False)

    test_loss = running_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    print(f"\tVAL: Loss={test_loss:.3f}, Acc={accuracy:.4f}")
