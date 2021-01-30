import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, SubsetRandomSampler
from networks.import_all_networks import *
from utils.import_utils import *


class Train_MNIST(object):
    def __init__(self):
        self.best_acc = 0
        self.in_channels = 1  # 1 because MNIST is grayscale
        self.dataset = mnist_data  # Class that is imported from utils that imports data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        self.args = self.prepare_args()
        self.transform_train, self.transform_test = self.prepare_transformations()

        if self.args.create_validationset:
            (
                self.loader_train,
                self.loader_validation,
                self.loader_test,
            ) = self.prepare_data()
            self.data_check_acc = self.loader_validation
        else:
            self.loader_train, self.loader_test = self.prepare_data()
            self.data_check_acc = self.loader_train

    def prepare_args(self):
        parser = argparse.ArgumentParser(description="PyTorch MNIST")
        parser.add_argument(
            "--resume",
            default="",
            type=str,
            metavar="PATH",
            help="path to latest checkpoint (default: none)",
        )
        parser.add_argument(
            "--lr",
            default=0.001,
            type=float,
            metavar="LR",
            help="initial learning rate",
        )
        parser.add_argument(
            "--weight-decay",
            default=1e-5,
            type=float,
            metavar="R",
            help="L2 regularization lambda",
        )
        parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="SGD with momentum"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            metavar="N",
            help="number of epochs to train (default: 100)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
            metavar="N",
            help="input batch size for training (default: 128)",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=240,
            metavar="N",
            help="how many batches to wait before logging training status",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        parser.add_argument(
            "--number-workers",
            type=int,
            default=0,
            metavar="S",
            help="number of workers (default: 0)",
        )
        parser.add_argument(
            "--init-padding",
            type=int,
            default=2,
            metavar="S",
            help=" If use initial padding or not. (default: 2 because mnist 28x28 to make 32x32)",
        )
        parser.add_argument(
            "--create-validationset",
            action="store_true",
            default=False,
            help="If you want to use a validation set (default: False). Default size = 10%",
        )
        parser.add_argument(
            "--save-model",
            action="store_true",
            default=False,
            help="If you want to save this model(default: False).",
        )
        args = parser.parse_args()
        return args

    def prepare_transformations(self):
        transform_train = transforms.Compose(
            [
                transforms.Pad(self.args.init_padding),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Pad(self.args.init_padding),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        return transform_train, transform_test

    def prepare_data(self, shuffle=True):

        data = self.dataset(
            shuffle,
            self.transform_train,
            self.transform_test,
            self.args.number_workers,
            self.args.create_validationset,
            self.args.batch_size,
            validation_size=0.1,
            random_seed=self.args.seed,
        )

        if self.args.create_validationset:
            loader_train, loader_validation, loader_test = data.main()

            return loader_train, loader_validation, loader_test

        else:
            loader_train, loader_test = data.main()

            return loader_train, loader_test

    def train(self):
        criterion = nn.CrossEntropyLoss()
        iter = 0

        # vis_plotting = visdom_plotting()
        loss_list, batch_list, epoch_list, validation_acc_list, training_acc_list = (
            [],
            [],
            [0],
            [0],
            [0],
        )

        for epoch in range(self.args.epochs):
            for batch_idx, (x, y) in enumerate(self.loader_train):
                self.model.train()
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=torch.long)

                scores = self.model(x)
                loss = criterion(scores, y)

                loss_list.append(loss.item())
                batch_list.append(iter + 1)
                iter += 1

                if batch_idx % self.args.log_interval == 0:
                    print(f"Batch {batch_idx}, epoch {epoch}, loss = {loss.item()}")
                    print()
                    self.model.eval()
                    train_acc = check_accuracy(self.data_check_acc, self.model)
                    # validation_acc = self.check_accuracy(self.data_check_acc)
                    validation_acc = 0
                    validation_acc_list.append(validation_acc)
                    training_acc_list.append(train_acc)
                    epoch_list.append(epoch + 0.5)
                    print()
                    print()
                    # call to plot in visdom
                    # vis_plotting.create_plot(loss_list, batch_list, validation_acc_list, epoch_list, training_acc_list)

                    # save checkpoint
                    if train_acc > self.best_acc and self.args.save_model:
                        self.best_acc = train_acc
                        save_checkpoint(
                            self.filename,
                            self.model,
                            self.optimizer,
                            self.best_acc,
                            epoch,
                        )

                self.model.train()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def choose_network(self):
        self.model = LeNet(
            in_channels=self.in_channels, init_weights=True, num_classes=10
        )
        self.filename = "checkpoint/mnist_LeNet.pth.tar"

        # self.model = VGG('VGG16', in_channels = self.in_channels)
        # self.filename =  'checkpoint/mnist_VGG16.pth.tar'

        # self.model = ResNet50(img_channel=1)
        # self.filename =  'checkpoint/mnist_ResNet.pth.tar'

        # self.model = GoogLeNet(img_channel=1)
        # self.filename =  'checkpoint/mnist_GoogLeNet.pth.tar'

        self.model = self.model.to(self.device)

    def main(self):
        if __name__ == "__main__":
            self.choose_network()
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            )
            cudnn.benchmark = True

            if self.args.resume:
                self.model.eval()
                (
                    self.model,
                    self.optimizer,
                    self.checkpoint,
                    self.start_epoch,
                    self.best_acc,
                ) = load_model(self.args, self.model, self.optimizer)
            else:
                load_model(self.args, self.model, self.optimizer)

            self.train()


## Mnist
network = Train_MNIST()
Train_MNIST.main(network)
