

def train(model, device, train_loader, criterion, optimizer, epoch, metrics_logger) -> None:
    model.train()
    running_loss = 0
    correct = 0

    for batch_id, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(labels.view_as(pred)).sum().item()

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_id == len(train_loader) - 1:
            metrics_logger.log_epoch(output, labels, running_loss, epoch)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)

    print(f"Epoch={epoch}, avg_loss={avg_loss:.3f}, acc={accuracy:.4f}")
