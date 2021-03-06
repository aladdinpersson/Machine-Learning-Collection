""" Training of ProGAN using WGAN-GP loss"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, plot_to_tensorboard, save_checkpoint, load_checkpoint
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import time

torch.backends.cudnn.benchmarks = True
torch.manual_seed(0)

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZES = [128, 128, 64, 16, 8, 4, 2, 2, 1]
IMAGE_SIZE = 128
CHANNELS_IMG = 3
Z_DIM = 128
IN_CHANNELS = 128
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE / 4)) + 1
PROGRESSIVE_EPOCHS = [2 ** i for i in range(int(log2(IMAGE_SIZE / 4) + 1))]
PROGRESSIVE_EPOCHS = [8 for i in range(int(log2(IMAGE_SIZE / 4) + 1))]
fixed_noise = torch.randn(8, Z_DIM, 1, 1).to(device)
NUM_WORKERS = 4

def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = BATCH_SIZES[int(log2(image_size/4))]
    dataset = datasets.ImageFolder(root="celeb_dataset", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    return loader, dataset

def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
):
    start = time.time()
    total_time = 0
    training = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(training):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        model_start = time.time()

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        for _ in range(CRITIC_ITERATIONS):
            critic.zero_grad()
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step).reshape(-1)
            critic_fake = critic(fake, alpha, step).reshape(-1)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
            )
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen.zero_grad()
        fake = gen(noise, alpha, step)
        gen_fake = critic(fake, alpha, step).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (PROGRESSIVE_EPOCHS[step]*0.5) * len(dataset) # - step
        )
        alpha = min(alpha, 1)
        total_time += time.time()-model_start

        if batch_idx % 300 == 0:
            with torch.no_grad():
                fixed_fakes = gen(fixed_noise, alpha, step)
            plot_to_tensorboard(
                writer, loss_critic, loss_gen, real, fixed_fakes, tensorboard_step
            )
            tensorboard_step += 1

    print(f'Fraction spent on model training: {total_time/(time.time()-start)}')
    return tensorboard_step, alpha


def main():
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(Z_DIM, IN_CHANNELS, img_size=IMAGE_SIZE, img_channels=CHANNELS_IMG).to(device)
    critic = Discriminator(IMAGE_SIZE, Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(device)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/gan")

    load_checkpoint(torch.load("celeba_wgan_gp.pth.tar"), gen, critic)
    gen.train()
    critic.train()

    tensorboard_step = 0
    for step, num_epochs in enumerate(PROGRESSIVE_EPOCHS):
        alpha = 0.01
        if step < 3:
            continue

        if step == 4:
            print(f"Img size is: {4*2**step}")

        loader, dataset = get_loader(4 * 2 ** step)
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
            )

            checkpoint = {'gen': gen.state_dict(),
                          'critic': critic.state_dict(),
                          'opt_gen': opt_gen.state_dict(),
                          'opt_critic': opt_critic.state_dict()}

            save_checkpoint(checkpoint)

if __name__ == "__main__":
    main()