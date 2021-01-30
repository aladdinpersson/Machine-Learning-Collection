"""
Example code of how to use the TensorBoard in PyTorch.
This code uses a lot of different functions from TensorBoard
and tries to have them all in a compact way, it might not be
super clear exactly what calls does what, for that I recommend
watching the YouTube video.

Video explanation: https://youtu.be/RLqsxWaQdHE
Got any questions leave a comment on youtube :)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-17 Initial coding
"""

# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)

# To do hyperparameter search, include more batch_sizes you want to try
# and more learning rates!
batch_sizes = [256]
learning_rates = [0.001]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        # Initialize network
        model = CNN(in_channels=in_channels, num_classes=num_classes)
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        writer = SummaryWriter(
            f"runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}"
        )

        # Visualize model in TensorBoard
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(device))
        writer.close()

        for epoch in range(num_epochs):
            losses = []
            accuracies = []

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate 'running' training accuracy
                features = data.reshape(data.shape[0], -1)
                img_grid = torchvision.utils.make_grid(data)
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct) / float(data.shape[0])
                accuracies.append(running_train_acc)

                # Plot things to tensorboard
                class_labels = [classes[label] for label in predictions]
                writer.add_image("mnist_images", img_grid)
                writer.add_histogram("fc1", model.fc1.weight)
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar(
                    "Training Accuracy", running_train_acc, global_step=step
                )

                if batch_idx == 230:
                    writer.add_embedding(
                        features,
                        metadata=class_labels,
                        label_img=data,
                        global_step=batch_idx,
                    )
                step += 1

            writer.add_hparams(
                {"lr": learning_rate, "bsize": batch_size},
                {
                    "accuracy": sum(accuracies) / len(accuracies),
                    "loss": sum(losses) / len(losses),
                },
            )
