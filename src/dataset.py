from typing import Dict, Any
import os
import numpy as np
import torch
from config import DataConfig
from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor, RTDetrImageProcessor
import torch.nn.functional as F
from augmentations import hflip_image_and_targets, color_jitter


def pad_to_size(image_tensor, target_size=(1280, 1280)):
    """
    Добавляет паддинг к изображению до target_size (H, W)
    Args:
        image_tensor: torch.Tensor [B, C, H, W] или [C, H, W]
        target_size: кортеж (height, width)
    Returns:
        Padded tensor [B, C, target_height, target_width]
    """
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    _, _, h, w = image_tensor.shape
    target_h, target_w = target_size
    
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    
    padding = (0, pad_w, 0, pad_h)
    
    return F.pad(image_tensor, padding, mode='constant', value=0)


class CocoDetectionTransforms(CocoDetection):
    def __init__(
        self,
        config: DataConfig,
        set_name: str,
        processor: DetrImageProcessor | RTDetrImageProcessor,
    ):
        self._config = config
        self.set_name = set_name
        root = os.path.join(self._config.data_path, self.set_name)
        annotation_file = os.path.join(root, "_annotations.coco.json")
        self.processor = processor
        super().__init__(root=root, annFile=annotation_file)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, target = super().__getitem__(idx)
        

        for item in target:
            item['category_id'] -= 1

        if self.set_name == 'train':
            img = color_jitter(img, p=0.5)

        image_id = self.ids[idx]
        selected_target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(
            images=img,
            annotations=selected_target,
            return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target_return = encoding["labels"][0] if len(encoding["labels"]) > 0 else torch.tensor([])
        if self.set_name == 'train':
            pixel_values, target_return = hflip_image_and_targets(pixel_values, target_return, p=0.5)
        return pixel_values, target_return
