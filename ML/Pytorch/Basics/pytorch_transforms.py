"""
Shows a small example of how to use transformations (perhaps unecessarily many)
on CIFAR10 dataset and training on a small CNN toy network.

Video explanation: https://youtu.be/Zvd276j9sZ8
Got any questions leave a comment I'm pretty good at responding on youtube

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-09 Initial coding
Bugfixes by Ohad Shapira <shapiraohad at gmail dot com>
*    2021-12-08 bug fixes
"""

# Imports
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from custom_dataset.custom_dataset import CatsAndDogsDataset

# Load data
my_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Resizes to (256,256)
        transforms.RandomCrop((224, 224)),  # Crop to random (224,224)
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomRotation(degrees=45),  # random rotation
        transforms.RandomGrayscale(p=0.2),  # Converts to grayscale with probability 0.2
        transforms.RandomHorizontalFlip(p=0.5),  # Flips horizontally with probability 0.5
        transforms.RandomVerticalFlip(p=0.05),  # Flips vertically with probability 0.05
        transforms.ToTensor(),  # Converts PIL image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

dataset = CatsAndDogsDataset(csv_file="custom_dataset/cats_dogs.csv", root_dir='custom_dataset/cats_dogs_resized',
                             transform=my_transforms)

img_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, 'custom_dataset/cats_dogs_transforms/img_{img_num}.png'.format(img_num=str(img_num)))
        img_num += 1
