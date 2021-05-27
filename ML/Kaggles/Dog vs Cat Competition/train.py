# Imports
import os
import torch
import torch.nn.functional as F
import numpy as np
import config
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CatDog
from efficientnet_pytorch import EfficientNet
from utils import check_accuracy, load_checkpoint, save_checkpoint


def save_feature_vectors(model, loader, output_size=(1, 1), file="trainb7"):
    model.eval()
    images, labels = [], []

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size)
        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.numpy())

    np.save(f"data_features/X_{file}.npy", np.concatenate(images, axis=0))
    np.save(f"data_features/y_{file}.npy", np.concatenate(labels, axis=0))
    model.train()


def train_one_epoch(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(config.DEVICE)
        targets = targets.to(config.DEVICE).unsqueeze(1).float()

        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())


def main():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    model._fc = nn.Linear(2560, 1)
    train_dataset = CatDog(root="data/train/", transform=config.basic_transform)
    test_dataset = CatDog(root="data/test/", transform=config.basic_transform)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = model.to(config.DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model)

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, loss_fn, optimizer, scaler)
        check_accuracy(train_loader, model, loss_fn)

    if config.SAVE_MODEL:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

    save_feature_vectors(model, train_loader, output_size=(1, 1), file="train_b7")
    save_feature_vectors(model, test_loader, output_size=(1, 1), file="test_b7")


if __name__ == "__main__":
    main()