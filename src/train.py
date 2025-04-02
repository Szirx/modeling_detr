import os
import argparse
from clearml import Task
from clearml_log import clearml_logging
from train_utils import load_object

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
    torch.set_float32_matmul_precision('high')

    task = Task.init(
        project_name=config.clearml_config.project_name,
        task_name=config.clearml_config.task,
    )
    logger = task.get_logger()
    clearml_logging(config, logger)

    processor = load_object(config.processor).from_pretrained(
        config.model_path,
        size={
            'max_height': config.data_config.processor_image_size,
            'max_width': config.data_config.processor_image_size,
        },
    )

    datamodule = DetrDataModule(config.data_config, processor)
    model = DetrLightning(config, processor)
    checkpoint_callback = ModelCheckpoint(
        monitor=config.monitor_metric,
        filename="detr-{epoch:02d}-{val_map_50:.2f}",
        save_top_k=config.save_top_k,
        verbose=True,
        mode='max',
    )

    early_stopping = EarlyStopping(
    monitor=config.monitor_metric,
        patience=config.patience,
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(
        accelerator=config.accelerator,
        devices=1,
        max_epochs=config.n_epochs,
        precision='32-true',
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=config.log_every_n_steps,
        
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
