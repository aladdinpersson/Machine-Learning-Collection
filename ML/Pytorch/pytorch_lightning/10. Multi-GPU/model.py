import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision


class NN(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, 1_000_000)
        self.fc2 = nn.Linear(1_000_000, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)

        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("mnist_images", grid, self.global_step)

        return {"loss": loss, "scores": scores, "y": y}
    
    def training_epoch_end(self, outputs):
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        self.log_dict(
            {
                "train_acc": self.accuracy(scores, y),
                "train_f1": self.f1_score(scores, y),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
