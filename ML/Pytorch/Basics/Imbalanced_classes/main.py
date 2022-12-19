"""
This code is for dealing with imbalanced datasets in PyTorch. Imbalanced datasets 
are those where the number of samples in one or more classes is significantly lower 
than the number of samples in the other classes. This can be a problem because it 
can lead to a model that is biased towards the more common classes, which can result 
in poor performance on the less common classes.

To deal with imbalanced datasets, this code implements two methods: oversampling and 
class weighting.

Oversampling involves generating additional samples for the underrepresented classes, 
while class weighting involves assigning higher weights to the loss of samples in the 
underrepresented classes, so that the model pays more attention to them.

In this code, the get_loader function takes a root directory for a dataset and a batch 
size, and returns a PyTorch data loader. The data loader is used to iterate over the 
dataset in batches. The get_loader function first applies some transformations to the 
images in the dataset using the transforms module from torchvision. Then it calculates 
the class weights based on the number of samples in each class. It then creates a 
WeightedRandomSampler object, which is used to randomly select a batch of samples with a 
probability proportional to their weights. Finally, it creates the data loader using the 
dataset and the weighted random sampler.

The main function then uses the data loader to iterate over the dataset for 10 epochs, 
and counts the number of samples in each class. Finally, it prints the counts for each class.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-04-08: Initial coding
* 2021-03-24: Added more detailed comments also removed part of
              check_accuracy which would only work specifically on MNIST.
* 2022-12-19: Updated detailed comments, small code revision, checked code still works with latest PyTorch. 
"""

import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Methods for dealing with imbalanced datasets:
# 1. Oversampling (probably preferable)
# 2. Class weighting


def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    subdirectories = dataset.classes
    class_weights = []

    # loop through each subdirectory and calculate the class weight
    # that is 1 / len(files) in that subdirectory
    for subdir in subdirectories:
        files = os.listdir(os.path.join(root_dir, subdir))
        class_weights.append(1 / len(files))

    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def main():
    loader = get_loader(root_dir="dataset", batch_size=8)

    num_retrievers = 0
    num_elkhounds = 0
    for epoch in range(10):
        for data, labels in loader:
            num_retrievers += torch.sum(labels == 0)
            num_elkhounds += torch.sum(labels == 1)

    print(num_retrievers.item())
    print(num_elkhounds.item())


if __name__ == "__main__":
    main()
