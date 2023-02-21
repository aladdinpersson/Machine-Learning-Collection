"""
Create a PyTorch Custom dataset that loads file in data/other.tsv that contains 
the path to image audio and text transcription.
"""
import pytorch_lightning as pl
from tqdm import tqdm
import ffmpeg
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
import sys 

class CommonVoice(Dataset):
    def __init__(self, data_dir, whisper_model="tiny"):
        self.sampling_rate = 16_000
        self.data_dir = data_dir
        self.data = pd.read_csv(
            os.path.join(data_dir, "other.tsv"),
            sep="\t",
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            f"openai/whisper-{whisper_model}"
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(
            f"openai/whisper-{whisper_model}", language="sv", task="transcribe"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file_path = os.path.join(
            self.data_dir + "clips/", self.data.iloc[idx]["path"]
        )
        sentence = self.data.iloc[idx]["sentence"]
        text = self.tokenizer(sentence).input_ids
        
        out, _ = (
            ffmpeg.input(audio_file_path, threads=0)
            .output(
                "-", format="s16le", acodec="pcm_s16le", ac=1, ar=self.sampling_rate
            )
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        # run feature extractor
        audio_features = self.feature_extractor(
            out, sampling_rate=self.sampling_rate, return_tensors="pt"
        )

        return audio_features, text


# Create a collator that will pad the audio features and text labels
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, batch):
        text_features = [{"input_ids": x[1]} for x in batch]
        batch_text = self.tokenizer.pad(
            text_features, return_tensors="pt",
        )
        audio_features = [{"input_features": x[0]["input_features"]} for x in batch]

        batch_audio = self.feature_extractor.pad(
            audio_features, return_tensors="pt",
        )
        batch_text["input_ids"] = batch_text["input_ids"].masked_fill(
            batch_text["attention_mask"].ne(1), -100
        )
        
        batch_audio["input_features"] = batch_audio["input_features"].squeeze(1)

        labels = batch_text["input_ids"].clone()
        if (labels[:, 0] == self.tokenizer.encode("")[0]).all().cpu().item():
            labels = labels[:, 1:]

        batch_text["labels"] = labels
        return batch_audio, batch_text


# Put into a lightning datamodule 
class WhisperDataset(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0, whisper_model="tiny"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.whisper_model = whisper_model
        self.sampling_rate = 16_000

    def setup(self, stage=None):
        self.dataset = CommonVoice(self.data_dir, self.whisper_model)
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            self.dataset.feature_extractor, self.dataset.tokenizer
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )


# Test if lightning datamodule working as intended 
if __name__ == "__main__":
    dm = WhisperDataset(data_dir="data/")
    dm.setup()
    from tqdm import tqdm 
    for batch in tqdm(dm.train_dataloader()):
        pass
