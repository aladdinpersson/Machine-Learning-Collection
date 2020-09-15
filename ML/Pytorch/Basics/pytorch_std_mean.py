"""
Calculation Of standard deviation and mean (per channel) over all images of the image dataset


*    2020-08-25 

"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import (
    tqdm,
)  # show progress bar when dataset is large install with pip install tqdm

# check device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# cifar10 dataset
train_set = datasets.CIFAR10(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)

"""
custom dataset( w/ CoCoannotation) ---- first one is trainset and second one is for validation set
if all your images in your custom dataset have signle image sizes your bigger batch_size
doc : https://pytorch.org/docs/stable/torchvision/datasets.html
note : you might need to install cocotools with pip install pycocotools

custom_train_set = datasets.CocoDetection(root = "../data/normalized/images",
                                annFile = "../data/normalized/datatrain90n.json",
                                  transform=transforms.ToTensor())
custom_train_loader = DataLoader(dataset=custom_train_set,batch_size=1,shuffle=True)

custom_val_set = datasets.CocoDetection(root = "../data/normalized/images",
                                        annFile = "../data/normalized/datatrain90n.json",
                                        transform=transforms.ToTensor())
custom_val_loader = DataLoader(dataset=custom_val_set,batch_size=1,shuffle=True)

"""


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


mean, std = get_mean_std(train_loader)
print(mean)
print(std)
