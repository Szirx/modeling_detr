from typing import Optional, Callable, Dict, Any
import os
import torch
from config import DataConfig
from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor
import torch.nn.functional as F

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
        processor: DetrImageProcessor,
        trans: Optional[Callable] = None,
    ):
        self._config = config
        self.set_name = set_name
        root = os.path.join(self._config.data_path + self.set_name, 'images')
        annotation_file = os.path.join(root, "_annotations.coco.json")
        self.processor = processor
        self.num_classes = 11
        super().__init__(root=root, annFile=annotation_file)

    def _normalize_bbox(self, bbox, img_width, img_height):
        x_min, y_min, x_max, y_max = bbox
        
        x = max(0, x_min / img_width)
        y = max(0, y_min / img_height)
        x1 = min(1, x_max / img_width)
        y1 = min(1, y_max / img_height)
        
        return [x, y, x1, y1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        orig_w, orig_h = img_info["width"], img_info["height"]
        
        inputs = self.processor(
            images=img, 
            return_tensors="pt",
            size={"max_height": 1280, "max_width": 1280}
        )
        
        detr_target = {
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "size": torch.as_tensor(
                [
                    self._config.processor_image_size,
                    self._config.processor_image_size,
                ],
                dtype=torch.int64,
            ),
            "orig_size": torch.as_tensor([orig_h, orig_w], dtype=torch.int64),
            "boxes": [],
            "non_norm_boxes": [],
            "class_labels": [],
        }
        
        for ann in target:
            bbox = ann["bbox"]
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h
            
            if w > 0 and h > 0:
                normalized_bbox = self._normalize_bbox(
                    [x_min, y_min, x_max, y_max],
                    self._config.processor_image_size,
                    self._config.processor_image_size,
                )
                detr_target["boxes"].append(normalized_bbox)
                detr_target["non_norm_boxes"].append([x_min, y_min, x_max, y_max])
                detr_target["class_labels"].append(ann["category_id"])

        if len(detr_target["boxes"]) > 0:
            detr_target["boxes"] = torch.as_tensor(detr_target["boxes"], dtype=torch.float32)
            detr_target["non_norm_boxes"] = torch.as_tensor(detr_target["non_norm_boxes"], dtype=torch.float32)
            detr_target["class_labels"] = torch.as_tensor(detr_target["class_labels"], dtype=torch.int64)
        else:
            detr_target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            detr_target["non_norm_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            detr_target["class_labels"] = torch.zeros(0, dtype=torch.int64)
        
        pixel_values = pad_to_size(inputs["pixel_values"])
        pixel_mask = pad_to_size(inputs["pixel_mask"])
        
        return {
            "pixel_values": pixel_values.squeeze(0),
            "pixel_mask": pixel_mask.squeeze(0),
            "labels": detr_target,
        }
