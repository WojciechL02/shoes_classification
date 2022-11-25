

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

