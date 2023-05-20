from dataset import MyDataModule
from model import MyLightningModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="my_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True,
    )
    logger = TensorBoardLogger("tb_logs", name="t5_dailymail")

    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # File paths
    train_csv = "train.csv"
    val_csv = "validation.csv"
    test_csv = "test.csv"

    # Create the data module
    dm = MyDataModule(train_csv, val_csv, test_csv, tokenizer, batch_size=32)
    dm.setup()

    model = MyLightningModule(
        model_name="t5-small", learning_rate=1e-4, weight_decay=1e-5
    )


    #checkpoint_path = "checkpoints/curr.ckpt"
    #checkpoint = torch.load(checkpoint_path)
    #model.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0, 1],
        max_epochs=10,
        precision=16,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    
    #example = """Former President Donald Trump claims in a social media post that he will be arrested next week. The claim comes while a New York prosecutor considers charging Trump in connection with hush money paid to adult film actress Stormy Daniels but there has been no official announcement of any plans for an indictment. What we know about Trump possibly facing criminal indictment in New York City. Trump has been entangled in several criminal investigations but the case related to Daniels is the longest-running of all of them, reaching back to 2016. On his platform Truth Social on Saturday morning, Trump cited "illegal leaks" that he will be arrested Tuesday and he called for protests. Trump, who is running for president in 2024, also defended himself, saying that he has not committed a crime — though he did not disclose what he expects to be charged with — and he accused the Manhattan District Attorney's Office of being "corrupt & highly political.". 'I'M BACK!' Trump posts on Facebook, YouTube for first time in two years. The Manhattan District Attorney's Office declined to comment on whether it will soon be pursing an arrest warrant for Trump. But the Associated Press reported that law enforcement officials in New York are discussing security preparations in anticipation that Trump may be indicted in coming weeks. If it does occur, Trump would become the first former president to be indicted in U.S. history.""" 
    #print(len(tokenizer(example)["input_ids"]))
    #summary = model.predict(example)
    #print(summary)
