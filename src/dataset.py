import os
import torch
from torchvision.datasets import CocoDetection
from config import DataConfig
from pycocotools.coco import COCO


class CocoDetectionTransforms(CocoDetection):
    def __init__(self, config: DataConfig, set_name: str, transform=None, resize=False):
        self._config = config
        self.root = self._config.data_path
        self.set_name = set_name
        self._transform = transform
        self.image_path = os.path.join(self.root, self.set_name)
        self.coco = COCO(os.path.join(self.image_path, '_annotations.coco.json'))

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        
        # Преобразование аннотаций COCO в формат DETR
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        
        # Преобразование аннотаций
        w, h = img.size
        target["size"] = torch.as_tensor([h, w])
        target["orig_size"] = torch.as_tensor([h, w])
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return {"pixel_values": img, "labels": target}