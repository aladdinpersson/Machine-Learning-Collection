import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from utils import save_checkpoint, load_checkpoint, check_accuracy
from sklearn.metrics import cohen_kappa_score
import config
import os
import pandas as pd


def make_prediction(model, loader, file):
    preds = []
    filenames = []
    model.eval()

    for x, y, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)
            # Convert MSE floats to integer predictions
            predictions[predictions < 0.5] = 0
            predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
            predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
            predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
            predictions[(predictions >= 3.5) & (predictions < 1000000000000)] = 4
            predictions = predictions.long().view(-1)
            y = y.view(-1)

            preds.append(predictions.cpu().numpy())
            filenames += map(list, zip(files[0], files[1]))

    filenames = [item for sublist in filenames for item in sublist]
    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(file, index=False)
    model.train()
    print("Done with predictions")


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.csv = pd.read_csv(csv_file)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        example = self.csv.iloc[index, :]
        features = example.iloc[: example.shape[0] - 4].to_numpy().astype(np.float32)
        labels = example.iloc[-4:-2].to_numpy().astype(np.int64)
        filenames = example.iloc[-2:].values.tolist()
        return features, labels, filenames


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d((1536 + 1) * 2),
            nn.Linear((1536+1) * 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = MyModel().to(config.DEVICE)
    ds = MyDataset(csv_file="train/train_blend.csv")
    loader = DataLoader(ds, batch_size=256, num_workers=3, pin_memory=True, shuffle=True)
    ds_val = MyDataset(csv_file="train/val_blend.csv")
    loader_val = DataLoader(
        ds_val, batch_size=256, num_workers=3, pin_memory=True, shuffle=True
    )
    ds_test = MyDataset(csv_file="train/test_blend.csv")
    loader_test = DataLoader(
        ds_test, batch_size=256, num_workers=2, pin_memory=True, shuffle=False
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    if config.LOAD_MODEL and "linear.pth.tar" in os.listdir():
        load_checkpoint(torch.load("linear.pth.tar"), model, optimizer, lr=1e-4)
        model.train()

    for _ in range(5):
        losses = []
        for x, y, files in tqdm(loader_val):
            x = x.to(config.DEVICE).float()
            y = y.to(config.DEVICE).view(-1).float()

            # forward
            scores = model(x).view(-1)
            loss = loss_fn(scores, y)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        print(f"Loss: {sum(losses)/len(losses)}")

    if config.SAVE_MODEL:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename="linear.pth.tar")

    preds, labels = check_accuracy(loader_val, model)
    print(cohen_kappa_score(labels, preds, weights="quadratic"))

    preds, labels = check_accuracy(loader, model)
    print(cohen_kappa_score(labels, preds, weights="quadratic"))

    make_prediction(model, loader_test, "test_preds.csv")
