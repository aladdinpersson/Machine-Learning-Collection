"""
Simple pytorch lightning example
"""

# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import Callback, EarlyStopping


precision = "medium"
torch.set_float32_matmul_precision(precision)
criterion = nn.CrossEntropyLoss()


## use 20% of training data for validation
# train_set_size = int(len(train_dataset) * 0.8)
# valid_set_size = len(train_dataset) - train_set_size
#
## split the train set into two
# seed = torch.Generator().manual_seed(42)
# train_dataset, val_dataset = torch.utils.data.random_split(
#    train_dataset, [train_set_size, valid_set_size], generator=seed
# )


class CNNLightning(pl.LightningModule):
    def __init__(self, lr=3e-4, in_channels=1, num_classes=10):
        super().__init__()
        self.lr = lr
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._common_step(x, batch_idx)
        loss = criterion(y_hat, y)
        accuracy = self.train_acc(y_hat, y)
        self.log(
            "train_acc_step",
            self.train_acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def training_epoch_end(self, outputs):
        self.train_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._common_step(x, batch_idx)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.test_acc(y_hat, y)
        self.log("test_loss", loss, on_step=True)
        self.log("test_acc", accuracy, on_step=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._common_step(x, batch_idx)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.test_acc(y_hat, y)
        self.log("val_loss", loss, on_step=True)
        self.log("val_acc", accuracy, on_step=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._common_step(x)
        return y_hat

    def _common_step(self, x, batch_idx):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        y_hat = self.fc1(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=512):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):
        mnist_full = train_dataset = datasets.MNIST(
            root="dataset/", train=True, transform=transforms.ToTensor(), download=True
        )
        self.mnist_test = datasets.MNIST(
            root="dataset/", train=False, transform=transforms.ToTensor(), download=True
        )
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(
            mnist_full, [55000, 5000]
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=2, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=2, shuffle=False
        )


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
if __name__ == "__main__":
    # Initialize network
    model_lightning = CNNLightning()

    trainer = pl.Trainer(
        #fast_dev_run=True,
        # overfit_batches=3,
        max_epochs=5,
        precision=16,
        accelerator="gpu",
        devices=[0,1],
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        auto_lr_find=True,
        enable_model_summary=True,
        profiler="simple",
        strategy="deepspeed_stage_1",
        # accumulate_grad_batches=2,
        # auto_scale_batch_size="binsearch",
        # log_every_n_steps=1,
    )

    dm = MNISTDataModule()

    # trainer tune first to find best batch size and lr
    trainer.tune(model_lightning, dm)

    trainer.fit(
        model=model_lightning,
        datamodule=dm,
    )

    # test model on test loader from LightningDataModule
    trainer.test(model=model_lightning, datamodule=dm)
