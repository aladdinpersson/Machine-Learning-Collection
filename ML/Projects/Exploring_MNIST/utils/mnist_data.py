import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler


class mnist_data(object):
    def __init__(
        self,
        shuffle,
        transform_train,
        transform_test,
        num_workers=0,
        create_validation_set=True,
        batch_size=128,
        validation_size=0.2,
        random_seed=1,
    ):
        self.shuffle = shuffle
        self.validation_size = validation_size
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.random_seed = random_seed
        self.create_validation_set = create_validation_set
        self.batch_size = batch_size
        self.num_workers = num_workers

    def download_data(self):
        mnist_trainset = datasets.MNIST(
            root="./data", train=True, download=True, transform=self.transform_train
        )
        mnist_testset = datasets.MNIST(
            root="./data", train=False, download=True, transform=self.transform_test
        )

        return mnist_trainset, mnist_testset

    def create_validationset(self, mnist_trainset):
        num_train = len(mnist_trainset)
        indices = list(range(num_train))
        split = int(self.validation_size * num_train)

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(valid_idx)

        loader_train = DataLoader(
            dataset=mnist_trainset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
        )
        loader_validation = DataLoader(
            dataset=mnist_trainset,
            batch_size=self.batch_size,
            sampler=validation_sampler,
            num_workers=self.num_workers,
        )

        return loader_train, loader_validation

    def main(self):
        mnist_trainset, mnist_testset = self.download_data()

        if self.create_validation_set:
            loader_train, loader_validation = self.create_validationset(mnist_trainset)
            loader_test = DataLoader(
                dataset=mnist_testset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

            return loader_train, loader_validation, loader_test

        else:
            loader_train = DataLoader(
                dataset=mnist_trainset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
            loader_test = DataLoader(
                dataset=mnist_testset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

            return loader_train, loader_test
