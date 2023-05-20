import torch
import torchvision.datasets as datasets  # Standard datasets
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import WhisperFinetuning
from dataset import  WhisperDataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
torch.set_float32_matmul_precision("medium")

# things to add
lr = 1e-5
batch_size = 32
num_workers = 4
model = WhisperFinetuning(lr)
dm = WhisperDataset(data_dir="data/", batch_size=batch_size, num_workers=num_workers)

if __name__ == "__main__":
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=[0],
        precision=16,
    )

    trainer.fit(model, dm)
 
