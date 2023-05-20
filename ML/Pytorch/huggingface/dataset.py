import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch


class cnn_dailymail(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)

        # if the csv_file is "train.csv" then only take out 10% of the data. make sure to reset indices etc 
        #if csv_file == "train.csv":
        #    self.data = self.data.sample(frac=0.05, random_state=42).reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data.loc[idx, "article"]
        highlights = self.data.loc[idx, "highlights"]

        inputs = self.tokenizer(
            article,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            highlights,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self, train_csv, val_csv, test_csv, tokenizer, batch_size=16, max_length=512
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = cnn_dailymail(
                self.train_csv, self.tokenizer, self.max_length
            )
            self.val_dataset = cnn_dailymail(
                self.val_csv, self.tokenizer, self.max_length
            )
        if stage in ("test", None):
            self.test_dataset = cnn_dailymail(
                self.test_csv, self.tokenizer, self.max_length
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=6,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )
