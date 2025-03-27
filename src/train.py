import os
import argparse
from config import Config
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning_module import DetrLightning
from datamodule import DetrDataModule


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config, config_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.devices

    datamodule = DetrDataModule(config.data_config)
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

    trainer = Trainer(
        accelerator=config.accelerator,
        devices=1,
        max_epochs=config.n_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=config.log_every_n_steps,
        precision=16
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    args = arg_parse()
    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config, args.config_file)
