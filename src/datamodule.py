from typing import Optional
from config import DataConfig
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
        self.collator = BatchCollator(processor=processor)

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
                set_name='valid',
                processor=self.processor,
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            collate_fn=self.collator,
        )

class BatchCollator:
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch if item is not None]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch if item is not None]
        batch: dict = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
