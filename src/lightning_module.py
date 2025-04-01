from config import Config
from train_utils import load_object
import torch
import pytorch_lightning as pl
from metrics import get_metrics
from transformers.image_transforms import center_to_corners_format
from transformers import DetrForObjectDetection, DetrConfig

# https://github.com/Isalia20/DETR-finetune/blob/main/detr_model.py
class DetrLightning(pl.LightningModule):
    def __init__(self, config: Config, processor):
        super().__init__()
        self._config = config
        self.processor = processor
        self.target_sizes = [(
            self._config.data_config.processor_image_size,
            self._config.data_config.processor_image_size,
        )]

        if self._config.pretrained:
            self._model = DetrForObjectDetection.from_pretrained(
                self._config.model_path,
                num_labels=self._config.num_classes,
                ignore_mismatched_sizes=True,
                num_queries=self._config.num_queries,
                id2label=self._config.id2label,
                label2id={v:k for k,v in self._config.id2label.items()},
            )
        else:
            config = DetrConfig(num_labels=self._config.num_classes)
            self._model = DetrForObjectDetection(config)
        
        metrics = get_metrics(box_format="cxcywh", iou_type="bbox", class_metrics=False)
        self._train_metrics = metrics.clone(prefix='train_')
        self._valid_metrics = metrics.clone(prefix='val_')
        self._val_outputs: list = []
        
    def common_step(self, batch):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        
        outputs = self._model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict
        
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch)
        self.update_map(batch, mode='train')
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), prog_bar=True)

        return loss

    def on_train_epoch_end(self):

        for key, value in self._train_metrics.compute().items():
            if 'per_class' in key:
                self.log(key, value.mean(), on_epoch=True)
            elif 'classes' in key:
                continue
            else:
                if value.numel() > 0:
                    self.log(key, value, on_epoch=True)
        
        self._train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch)
        self.update_map(batch)
        self._val_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self._val_outputs).mean()

        for key, value in self._valid_metrics.compute().items():
            if 'per_class' in key:
                self.log(key, value.mean(), on_epoch=True)
            elif 'classes' in key:
                continue
            else:
                if value.numel() > 0:
                    self.log(key, value, on_epoch=True)

        self.log('val_loss', avg_loss, on_epoch=True)
        
        self._val_outputs.clear()
        self._valid_metrics.reset()

    def predict_image(self, batch):
        with torch.no_grad():
            outputs = self._model(pixel_values=batch["pixel_values"].cuda(), pixel_mask=batch["pixel_mask"].cuda())
        return outputs
    
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

    def update_map(self, batch, mode='val'):
            outputs = self.predict_image(batch)
        
            postprocessed_outputs = self.processor.post_process_object_detection(
                outputs,
                threshold=self._config.threshold,
            )
            predictions = [
                {
                    "boxes": postprocessed_output["boxes"].unsqueeze(0).cuda() if len(postprocessed_output["boxes"].shape) == 1 else postprocessed_output["boxes"].cuda(),
                    "labels": torch.tensor(postprocessed_output["labels"], device="cuda", dtype=torch.int),
                    "scores": postprocessed_output["scores"].cuda(),
                } for postprocessed_output in postprocessed_outputs
            ]
            target = [center_to_corners_format(i["boxes"]).squeeze(0).cuda() for i in batch["labels"]]
            ground_truths = [
                {
                    "boxes": box.unsqueeze(0) if len(box.shape) == 1 else box,
                    "labels": batch["labels"][i]["class_labels"].to("cuda"),
                } for i, box in enumerate(target)
            ]
            if mode == 'train':
                self._train_metrics.update(predictions, ground_truths)
            elif mode == 'val':
                self._valid_metrics.update(predictions, ground_truths)
