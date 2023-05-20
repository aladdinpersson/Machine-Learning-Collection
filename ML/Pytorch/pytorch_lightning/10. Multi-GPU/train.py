import torch
import pytorch_lightning as pl
from model import NN
from dataset import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy

torch.set_float32_matmul_precision("medium") # to make lightning happy

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")
    strategy = DeepSpeedStrategy()
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    model = NN(
        input_size=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
    )
    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    trainer = pl.Trainer(
        strategy=strategy,
        profiler=profiler,
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
