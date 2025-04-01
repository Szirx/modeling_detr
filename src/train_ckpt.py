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
        save_top_k=1,
        verbose=True,
        mode='max',
    )

    early_stopping = EarlyStopping(
    monitor=config.monitor_metric,
        patience=100,
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(
        accelerator=config.accelerator,
        devices=1,
        max_epochs=config.n_epochs,
        precision='32-true',
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=1,
        
    )
    task.upload_artifact(
        name='config_file',
        artifact_object=config_file,
    )
    trainer.fit(model=model,
                ckpt_path=config.ckpt_path,
                datamodule=datamodule,
    )

    trained_model = DetrLightning.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trained_model.model.save_pretrained('saved_models/detr/')
    task.upload_artifact(
        name='best_transformers_model',
        artifact_object='saved_models/detr'
    )



if __name__ == '__main__':
    args = arg_parse()
    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config, args.config_file)
