""" 
Implementation of Neural collaborative filtering (NCF)
Next:
    * Understand and use NDCG = Normalized Discounted Cumulative Gain
    * Use SVD and compare results
"""

import torch
import pytorch_lightning as pl
import pandas as pd
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split

torch.set_float32_matmul_precision("medium") # to make lightning happy

class MovieLens(Dataset):
    def __init__(self, df_ratings):
        self.df_ratings = df_ratings

    def __len__(self):
        return len(self.df_ratings)

    def __getitem__(self, idx):
        row = self.df_ratings.iloc[idx]
        user_id = torch.tensor(row["user_id"], dtype=torch.long)
        movie_id = torch.tensor(row["movie_id"], dtype=torch.long)
        rating = torch.tensor(row["rating"], dtype=torch.float)
        return user_id, movie_id, rating


class LightningData(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        self.df_ratings = pd.read_csv(
            "data/ratings.dat",
            sep="::",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
        )


        # split into train and test 
        self.df_ratings_train, self.df_ratings_val = train_test_split(
            self.df_ratings, test_size=0.2, random_state=42
        )

    def setup(self, stage=None):
        self.dataset_train = MovieLens(self.df_ratings_train)
        self.dataset_val = MovieLens(self.df_ratings_val)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=2)

class Net(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.lin = nn.Linear(n_factors * 2, 1)

    def forward(self, user, movie):
        user_embedding = self.user_factors(user)
        movie_embedding = self.movie_factors(movie)
        x = torch.cat([user_embedding, movie_embedding], dim=1)
        return self.lin(x)


class NetLightning(pl.LightningModule):
    def __init__(self, n_users, n_movies, n_factors=50, lr=3e-4):
        super().__init__()
        self.num_users = n_users
        self.num_movies = n_movies
        self.net = Net(n_users, n_movies, n_factors)
        self.loss_fn = nn.MSELoss()
        self.MAE = torchmetrics.MeanAbsoluteError()
        self.lr = lr

    def forward(self, user, movie):
        return self.net(user, movie)

    def training_step(self, batch, batch_idx):
        user, movie, rating = batch
        out = self.forward(user, movie)
        mae = self.MAE(out.squeeze(1), rating.float())
        loss = self.loss_fn(out.squeeze(1), rating.float())
        self.log_dict({"train_loss": loss, "train_mae": mae}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user, movie, rating = batch
        out = self.forward(user, movie)
        mae = self.MAE(out.squeeze(1), rating.float())
        loss = self.loss_fn(out.squeeze(1), rating.float())
        self.log_dict({"val_loss": loss, "val_mae": mae}, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, user_id):
        out = self.forward(user_id, torch.arange(0, self.num_movies))
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


dm = LightningData(batch_size=512)
dm.prepare_data()
dm.setup()

num_movies = dm.df_ratings["movie_id"].max() + 1
num_users = dm.df_ratings["user_id"].max() + 1

model = NetLightning(num_users, num_movies)
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=3)
trainer.fit(model, dm)
trainer.validate(model, dm)
