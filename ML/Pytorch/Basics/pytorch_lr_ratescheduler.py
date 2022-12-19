"""
Example code of how to use a learning rate scheduler simple, in this
case with a (very) small and simple Feedforward Network training on MNIST
dataset with a learning rate scheduler. In this case ReduceLROnPlateau
scheduler is used, but can easily be changed to any of the other schedulers
available. I think simply reducing LR by 1/10 or so, when loss plateaus is 
a good default. 

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-10 Initial programming
*    2022-12-19 Updated comments, made sure it works with latest PyTorch

"""

# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 10
learning_rate = (
    0.1  # way too high learning rate, but we want to see the scheduler in action
)
batch_size = 128
num_epochs = 100

# Define a very simple model
model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10)).to(device)

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

# Train Network
for epoch in range(1, num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.reshape(data.shape[0], -1)
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

    mean_loss = sum(losses) / len(losses)
    mean_loss = round(mean_loss, 2)  # we should see difference in loss at 2 decimals

    # After each epoch do scheduler.step, note in this scheduler we need to send
    # in loss for that epoch! This can also be set using validation loss, and also
    # in the forward loop we can do on our batch but then we might need to modify
    # the patience parameter
    scheduler.step(mean_loss)
    print(f"Average loss for epoch {epoch} was {mean_loss}")

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            x = x.reshape(x.shape[0], -1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


check_accuracy(train_loader, model)
