import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Net
from utils import check_accuracy, load_checkpoint, save_checkpoint, make_prediction
import config
from dataset import MyImageFolder


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.float())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def main():
    train_ds = MyImageFolder(root_dir="train/", transform=config.train_transforms)
    val_ds = MyImageFolder(root_dir="val/", transform=config.val_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,pin_memory=config.PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,pin_memory=config.PIN_MEMORY,shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    model = Net(net_version="b0", num_classes=10).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    make_prediction(model, config.val_transforms, 'test/', config.DEVICE)
    check_accuracy(val_loader, model, config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
        check_accuracy(val_loader, model, config.DEVICE)
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

if __name__ == "__main__":
    main()