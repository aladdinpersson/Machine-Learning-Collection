from tqdm import tqdm
import os
from collections import Counter
import pandas as pd
import spacy
import sys
from torchvision.datasets import Flickr30k
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((456, 456)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

spacy_eng = spacy.load("en")


def tokenizer_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


class Vocabulary:
    def __init__(self, freq_threshold):
        # Integer to String, and String to Integer
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in tokenizer_eng(sentence):
                # This is a more elementary way of doing it
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                # If it ever reaches a frequency of the threshold
                # we add it to our vocabulary (stoi & itos)
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class CaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=1):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, caption column
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary class and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # If you want to use pack_padded_sequence you should also get
        # all the lengths of the sequence here. I won't use it so I
        # skip it
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets


dataset = FlickrDataset(
    "flickr8k/images/", "flickr8k/captions.txt", transform=transform
)
loader = DataLoader(
    dataset=dataset, batch_size=4, collate_fn=MyCollate(dataset.vocab.stoi["<SOS>"])
)

for idx, (img, numeric_caption) in enumerate(loader):
    print(numeric_caption)
    print(img.shape)
