# Imports
import torch
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

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
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )


# check that it works
if __name__ == "__main__":
    dm = MNISTDataModule()
    dm.setup("fit")
    print(len(dm.mnist_train))
    print(len(dm.mnist_val))
    print(len(dm.mnist_test))
