"""
Example code of how to code GANs and more specifically DCGAN,
for more information about DCGANs read: https://arxiv.org/abs/1511.06434

We then train the DCGAN on the MNIST dataset (toy dataset of handwritten digits)
and then generate our own. You can apply this more generally on really any dataset
but MNIST is simple enough to get the overall idea.

Video explanation: https://youtu.be/5RYETbFFQ7s
Got any questions leave a comment on youtube :)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-20 Initial coding

"""

# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from model_utils import (
    Discriminator,
    Generator,
)  # Import our models we've defined (from DCGAN paper)

# Hyperparameters
lr = 0.0005
batch_size = 64
image_size = 64
channels_img = 1
channels_noise = 256
num_epochs = 10

# For how many channels Generator and Discriminator should use
features_d = 16
features_g = 16

my_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(
    root="dataset/", train=True, transform=my_transforms, download=True
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create discriminator and generator
netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)

# Setup Optimizer for G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

netG.train()
netD.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(64, channels_noise, 1, 1).to(device)
writer_real = SummaryWriter(f"runs/GAN_MNIST/test_real")
writer_fake = SummaryWriter(f"runs/GAN_MNIST/test_fake")
step = 0

print("Starting Training...")

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        label = (torch.ones(batch_size) * 0.9).to(device)
        output = netD(data).reshape(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()

        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = netG(noise)
        label = (torch.ones(batch_size) * 0.1).to(device)

        output = netD(fake.detach()).reshape(-1)
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        ### Train Generator: max log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        # Print losses ocassionally and print to tensorboard
        if batch_idx % 100 == 0:
            step += 1
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f} D(x): {D_x:.4f}"
            )

            with torch.no_grad():
                fake = netG(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
