import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration


class WhisperFinetuning(pl.LightningModule):
    def __init__(self, lr, whisper_model="tiny"):
        super().__init__()
        self.lr = lr
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{whisper_model}")
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

    def training_step(self, batch, batch_idx):
        encoder_input = batch[0]["input_features"]
        decoder_labels = batch[1]["labels"]
        
        out = self.model(
            input_features=encoder_input,
            labels=decoder_labels,
        )
        loss = out["loss"] 
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    pass
