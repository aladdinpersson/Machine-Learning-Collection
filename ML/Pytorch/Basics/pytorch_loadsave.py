"""
Small code example of how to save and load checkpoint of a model.
This example doesn't perform any training, so it would be quite useless.

In practice you would save the model as you train, and then load before 
continuining training at another point.

Video explanation of code & how to save and load model: https://youtu.be/g6kQl_EFn84
Got any questions leave a comment on youtube :)

Coded by Aladdin Persson <aladdin dot person at hotmail dot com>
-   2020-04-07 Initial programming

"""

# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def main():
    # Initialize network
    model = torchvision.models.vgg16(pretrained=False)
    optimizer = optim.Adam(model.parameters())

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    # Try save checkpoint
    save_checkpoint(checkpoint)

    # Try load checkpoint
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


if __name__ == "__main__":
    main()
