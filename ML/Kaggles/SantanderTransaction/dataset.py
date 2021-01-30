import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from math import ceil

def get_data():
    train_data = pd.read_csv("new_shiny_train.csv")
    y = train_data["target"]
    X = train_data.drop(["ID_code", "target"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    train_ds, val_ds = random_split(ds, [int(0.999*len(ds)), ceil(0.001*len(ds))])

    test_data = pd.read_csv("new_shiny_test.csv")
    test_ids = test_data["ID_code"]
    X = test_data.drop(["ID_code"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    test_ds = TensorDataset(X_tensor, y_tensor)

    return train_ds, val_ds, test_ds, test_ids
