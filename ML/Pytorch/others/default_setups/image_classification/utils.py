import torch
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = torch.sigmoid(model(x))
            predictions = (scores>0.5).float()
            num_correct += (predictions == y).sum()
            num_samples += predictions.shape[0]

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def make_prediction(model, transform, rootdir, device):
    files = os.listdir(rootdir)
    preds = []
    model.eval()

    files = sorted(files, key=lambda x: float(x.split(".")[0]))
    for file in tqdm(files):
        img = Image.open(os.path.join(rootdir, file))
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(img))
            preds.append(pred.item())


    df = pd.DataFrame({'id': np.arange(1, len(preds)+1), 'label': np.array(preds)})
    df.to_csv('submission.csv', index=False)
    model.train()
    print("Done with predictions")