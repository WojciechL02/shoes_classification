from torchmetrics import Recall, Precision, F1Score, Accuracy
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(self, num_classes) -> None:
        self.writer = SummaryWriter()
        self.accuracy = Accuracy(average='macro', num_classes=num_classes)
        self.recall = Recall(average='macro', num_classes=num_classes)
        self.precision = Precision(average='macro', num_classes=num_classes)
        self.f1score = F1Score(average='macro', num_classes=num_classes)

    def log_epoch(self, y_pred, y_true, loss, epoch, training=True):
        t = "train" if training else "val"
        self.writer.add_scalar(f"Loss/{t}", loss, epoch)
        self.writer.add_scalar(f"Accuracy/{t}", self.accuracy(y_pred, y_true).item(), epoch)
        self.writer.add_scalar(f"Recall/{t}", self.recall(y_pred, y_true).item(), epoch)
        self.writer.add_scalar(f"Precision/{t}", self.precision(y_pred, y_true).item(), epoch)
        self.writer.add_scalar(f"F1Score/{t}", self.f1score(y_pred, y_true).item(), epoch)
