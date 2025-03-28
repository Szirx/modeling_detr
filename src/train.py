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
from transformers import DetrImageProcessor


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config, config_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.devices
    torch.set_float32_matmul_precision('high')

    task = Task.init(project_name=config.project_name, task_name=config.task)
    logger = task.get_logger()
    clearml_logging(config, logger)


    processor = DetrImageProcessor.from_pretrained(
        config.model_path,
        revision="no_timm",
        size={
            'max_height': config.data_config.processor_image_size,
            'max_width': config.data_config.processor_image_size,
        },
    )

    datamodule = DetrDataModule(config.data_config, processor)
    model = DetrLightning(config, processor)
    checkpoint_callback = ModelCheckpoint(
        monitor=config.monitor_metric,
        dirpath="checkpoints",
        filename="detr-{epoch:02d}-{val_map_50:.2f}",
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
