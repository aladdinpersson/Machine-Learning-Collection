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
import config

torch.backends.cudnn.benchmarks = True

def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size/4))]
    dataset = datasets.ImageFolder(root="celeb_dataset", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
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
    scaler_gen,
    scaler_critic,
):
    start = time.time()
    total_time = 0
    loop = tqdm(loader, leave=True)
    losses_critic = []

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]
        model_start = time.time()

        for _ in range(4):
            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # which is equivalent to minimizing the negative of the expression
            for _ in range(config.CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

                with torch.cuda.amp.autocast():
                    fake = gen(noise, alpha, step)
                    critic_real = critic(real, alpha, step).reshape(-1)
                    critic_fake = critic(fake, alpha, step).reshape(-1)
                    gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
                    loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake))
                        + config.LAMBDA_GP * gp
                    )

                losses_critic.append(loss_critic.item())
                opt_critic.zero_grad()
                scaler_critic.scale(loss_critic).backward()
                scaler_critic.step(opt_critic)
                scaler_critic.update()
                #loss_critic.backward(retain_graph=True)
                #opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            with torch.cuda.amp.autocast():
                fake = gen(noise, alpha, step)
                gen_fake = critic(fake, alpha, step).reshape(-1)
                loss_gen = -torch.mean(gen_fake)

            opt_gen.zero_grad()
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()
            #gen.zero_grad()
            #loss_gen.backward()
            #opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (config.PROGRESSIVE_EPOCHS[step]*0.5) * len(dataset) # - step
        )
        alpha = min(alpha, 1)
        total_time += time.time()-model_start

        if batch_idx % 10 == 0:
            print(alpha)
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step)
            plot_to_tensorboard(
                writer, loss_critic, loss_gen, real, fixed_fakes, tensorboard_step
            )
            tensorboard_step += 1

        mean_loss = sum(losses_critic) / len(losses_critic)
        loop.set_postfix(loss=mean_loss)

    print(f'Fraction spent on model training: {total_time/(time.time()-start)}')
    return tensorboard_step, alpha


def main():
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_size=config.IMAGE_SIZE, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Discriminator(config.IMAGE_SIZE, config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/gan")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )

    gen.train()
    critic.train()

    tensorboard_step = 0
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE/4))

    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 0.01
        loader, dataset = get_loader(4 * 2 ** step) # 4->0, 8->1, 16->2, 32->3
        print(f"Current image size: {4*2**step}")
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
                scaler_gen,
                scaler_critic,
            )

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

        step += 1

if __name__ == "__main__":
    main()