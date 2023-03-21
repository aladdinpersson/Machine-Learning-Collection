import torch
import pytorch_lightning as pl
from datasets import load_dataset, load_metric
from transformers import T5Config, T5ForConditionalGeneration

from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


class MyLightningModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, weight_decay):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the pre-trained model and tokenizer
        #self.model = torch.compile(
        #    AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        #)

        # Create a T5-small configuration
        config = T5Config.from_pretrained("t5-small")

        # Initialize the T5 model with random weights
        self.model = torch.compile(T5ForConditionalGeneration(config))

        # Load the ROUGE metric
        self.metric = load_metric("rouge")
        self.logits = []
        self.labels = []

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, on_epoch=True, on_step=False)

        # add logits and labels to instance attributes, but make sure to detach them
        # from the computational graph first
        self.logits.append(logits.argmax(dim=-1).detach().cpu())
        self.labels.append(labels.detach().cpu())
        return {"loss": loss, "logits": logits, "labels": labels}

    def on_validation_epoch_end(self):
        # Concatenate tensors in logits and labels lists
        pred_token_ids = torch.cat(self.logits, dim=0)
        true_labels = torch.cat(self.labels, dim=0)

        # Decode predictions and labels using the saved instance attributes
        decoded_preds = self.tokenizer.batch_decode(
            pred_token_ids, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            true_labels, skip_special_tokens=True
        )

        # Compute ROUGE scores
        scores = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels, rouge_types=["rouge1"]
        )["rouge1"].mid

        self.log("rouge1_precision", scores.precision, prog_bar=True)
        self.log("rouge1_recall", scores.recall, prog_bar=True)
        self.log("rouge1_fmeasure", scores.fmeasure, prog_bar=True)

        # Clear logits and labels instance attributes for the next validation epoch
        self.logits.clear()
        self.labels.clear()

    def predict(self, article: str, max_input_length: int = 512, max_output_length: int = 150) -> str:
        # Set the model to evaluation mode
        self.model.eval()

        # Tokenize the input article
        inputs = self.tokenizer(
            article,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move the input tensors to the same device as the model
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Generate summary
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_output_length,
                num_return_sequences=1,
            )

        # Decode and return the summary
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return summary

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


