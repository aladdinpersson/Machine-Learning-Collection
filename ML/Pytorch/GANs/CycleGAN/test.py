import torch
import config
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import HorseZebraDataset
from generator_model import Generator
from utils import load_checkpoint



def test_fn(gen_Z, gen_H, loader):

    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)

        save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
        save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

def main():
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)


    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    load_checkpoint(
        config.CHECKPOINT_GEN_H,
        gen_H,
        opt_gen,
        config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_GEN_Z,
        gen_Z,
        opt_gen,
        config.LEARNING_RATE,
    )

    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR + "/testA",
        root_zebra=config.VAL_DIR + "/testB",
        transform=config.transforms,
    )

    loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_fn(gen_Z, gen_H, loader)

if __name__ == "__main__":
    main()