import torch
import torchvision
from torch import nn
import pytorch_lightning as pl


class VAEpl(pl.LightningModule):
    def __init__(self, lr, input_dim=784, h_dim=200, z_dim=20):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.BCELoss(reduction="sum")
        self.input_dim = input_dim

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma * epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(-1, self.input_dim)
        x_reconstructed, mu, sigma = self.forward(x)
        reconstruction_loss = self.loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        loss = reconstruction_loss + kl_div
        self.log("train_loss", loss, sync_dist=True)

        # add logging of images to tensorboard, x_reconstructed and x, so that
        # it updates every step and we can the progress pictures in tensorboard
        if batch_idx % 100 == 0:
            # take out the first 8
            x = x[:8]
            x_reconstructed = x_reconstructed[:8]
            grid = torchvision.utils.make_grid(x_reconstructed.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("reconstructed", grid, self.global_step)
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("original", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(-1, self.input_dim)
        x_reconstructed, mu, sigma = self.forward(x)
        reconstruction_loss = self.loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        loss = reconstruction_loss + kl_div
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(-1, self.input_dim)
        x_reconstructed, mu, sigma = self.forward(x)
        reconstruction_loss = self.loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        loss = reconstruction_loss + kl_div
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    batch_size = 8
    x = torch.randn(batch_size, 28 * 28 * 1)
    vae_pl = VAEpl()
    x_reconstructed, mu, sigma = vae_pl(x)
    print(x_reconstructed.shape)
