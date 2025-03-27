from typing import List, Dict, Any
from config import Config
from train_utils import load_object
import torch
import pytorch_lightning as pl
from metrics import get_metrics
from transformers import DetrForObjectDetection, DetrConfig


class DetrLightning(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        
        if self._config.pretrained:
            self._model = DetrForObjectDetection.from_pretrained(
                self._config.model_path,
                num_labels=self._config.num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = DetrConfig(num_labels=self._config.num_classes)
            self._model = DetrForObjectDetection(config)
        
        metrics = get_metrics(class_metrics=True)
        self._valid_metrics = metrics.clone(prefix='val_')

    def forward(self, pixel_values, pixel_mask=None):
        return self._model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        outputs = self._model(
            pixel_values=pixel_values,
            labels=labels
        )
        
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v.item(), prog_bar=True)
        
        with torch.no_grad():
            outputs = self._model(pixel_values=batch["pixel_values"])
        
        preds = self._convert_outputs_to_coco_format(outputs)
        targets = self._convert_labels_to_coco_format(batch["labels"])
        
        self._valid_metrics.update(preds, targets)
        
        return loss

    def on_validation_epoch_end(self):
        map_metrics = self._valid_metrics.compute()
        self.log_dict({f"val_{k}": v for k, v in map_metrics.items()})
        self._valid_metrics.reset()

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def _convert_outputs_to_coco_format(self, outputs) -> List[Dict[str, Any]]:
        """Convert DETR outputs to COCO evaluation format"""
        preds = []
        for idx in range(len(outputs.logits)):
            boxes = outputs.pred_boxes[idx].cpu()
            scores = outputs.logits[idx].softmax(-1)[:, :-1].max(-1).values.cpu()
            labels = outputs.logits[idx].softmax(-1)[:, :-1].argmax(-1).cpu()
            
            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })
        return preds

    def _convert_labels_to_coco_format(self, labels) -> List[Dict[str, Any]]:
        """Convert DETR labels to COCO evaluation format"""
        targets = []
        for label in labels:
            targets.append({
                "boxes": label["boxes"].cpu(),
                "labels": label["class_labels"].cpu()
            })
        return targets