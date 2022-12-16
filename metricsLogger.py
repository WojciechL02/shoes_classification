from torchmetrics import Recall, Precision, F1Score, Accuracy
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(self, model_name, num_classes) -> None:
        self.mn = model_name
        self.writer = SummaryWriter()
        self.accuracy = Accuracy(average='macro', num_classes=num_classes)
        self.recall = Recall(average='macro', num_classes=num_classes)
        self.precision = Precision(average='macro', num_classes=num_classes)
        self.f1score = F1Score(average='macro', num_classes=num_classes)

    def log_epoch(self, y_pred, y_true, loss, epoch, training=True):
        t = "train" if training else "val"
        self.writer.add_scalar(f"{self.mn}/Loss/{t}", loss, epoch)
        self.writer.add_scalar(f"{self.mn}/Accuracy/{t}", self.accuracy(y_pred, y_true).item(), epoch)
        self.writer.add_scalar(f"{self.mn}/Recall/{t}", self.recall(y_pred, y_true).item(), epoch)
        self.writer.add_scalar(f"{self.mn}/Precision/{t}", self.precision(y_pred, y_true).item(), epoch)
        self.writer.add_scalar(f"{self.mn}/F1Score/{t}", self.f1score(y_pred, y_true).item(), epoch)
