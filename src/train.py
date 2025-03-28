import os
import argparse
from clearml import Task
from clearml_log import clearml_logging

from config import Config
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning_module import DetrLightning
from datamodule import DetrDataModule


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config, config_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.devices

    task = Task.init(project_name=config.project_name, task_name=config.task)
    logger = task.get_logger()
    clearml_logging(config, logger)

    datamodule = DetrDataModule(config.data_config)
    torch.set_float32_matmul_precision('high')
    model = DetrLightning(config)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="detr-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        mode='max',
    )

    early_stopping = EarlyStopping(
    monitor=config.monitor_metric,
        patience=10,
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(
        accelerator=config.accelerator,
        devices=1,
        max_epochs=config.n_epochs,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10,
        precision=32,
    )
    task.upload_artifact(
        name='config_file',
        artifact_object=config_file,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    args = arg_parse()
    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config, args.config_file)
