from typing import Optional
from config import DataConfig
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from dataset import CocoDetectionTransforms


class DetrDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig, processor):
        super().__init__()
        self._config = config
        self.batch_size = self._config.batch_size
        self.n_workers = self._config.n_workers
        self.processor = processor
        
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
    
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = CocoDetectionTransforms(
                self._config,
                set_name='train',
                processor=self.processor,
            )
            self.valid_dataset = CocoDetectionTransforms(
                self._config,
                set_name='val',
                processor=self.processor,
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        pixel_values = torch.stack(pixel_values)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }