import torch
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset
from torchvision.utils import save_image
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    make_prediction,
    get_csv_for_blend,
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):
        # save examples and make sure they look ok with the data augmentation,
        # tip is to first set mean=[0,0,0], std=[1,1,1] so they look "normal"
        #save_image(data, f"hi_{batch_idx}.png")

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.unsqueeze(1).float())

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"Loss average over epoch: {sum(losses)/len(losses)}")


def main():
    train_ds = DRDataset(
        images_folder="train/images_preprocessed_1000/",
        path_to_csv="train/trainLabels.csv",
        transform=config.val_transforms,
    )
    val_ds = DRDataset(
        images_folder="train/images_preprocessed_1000/",
        path_to_csv="train/valLabels.csv",
        transform=config.val_transforms,
    )
    test_ds = DRDataset(
        images_folder="test/images_preprocessed_1000",
        path_to_csv="train/trainLabels.csv",
        transform=config.val_transforms,
        train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, num_workers=6, shuffle=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    loss_fn = nn.MSELoss()

    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(1536, 1)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    # Run after training is done and you've achieved good result
    # on validation set, then run train_blend.py file to use information
    # about both eyes concatenated
    get_csv_for_blend(val_loader, model, "../train/val_blend.csv")
    get_csv_for_blend(train_loader, model, "../train/train_blend.csv")
    get_csv_for_blend(test_loader, model, "../train/test_blend.csv")
    make_prediction(model, test_loader, "submission_.csv")
    import sys
    sys.exit()
    #make_prediction(model, test_loader)

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # get on validation
        preds, labels = check_accuracy(val_loader, model, config.DEVICE)
        print(f"QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        # get on train
        #preds, labels = check_accuracy(train_loader, model, config.DEVICE)
        #print(f"QuadraticWeightedKappa (Training): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"b3_{epoch}.pth.tar")



if __name__ == "__main__":
    main()
