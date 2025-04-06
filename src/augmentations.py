from torchvision.transforms import functional as F  # Правильный импорт
import torch
from torchvision.transforms.functional import hflip
import random


def hflip_image_and_targets(image, target):
    target_boxes = target["boxes"]
    target_xs = target_boxes[:, [0]]
    target_xs = 1 - target_xs
    image = hflip(image)
    targets = torch.cat([target_xs, target_boxes[:, 1:]], dim=1)
    target["boxes"] = targets
    return image, target


def color_jitter(pil_img, p=0.5):
        img_tensor = F.to_tensor(pil_img)
        """Цветовые аугментации"""
        # Random brightness
        if random.random() < p:
            brightness_factor = random.uniform(0.8, 1.2)
            img_tensor = F.adjust_brightness(img_tensor, brightness_factor)
        
        # Random contrast
        if random.random() < p:
            contrast_factor = random.uniform(0.8, 1.2)
            img_tensor = F.adjust_contrast(img_tensor, contrast_factor)
        
        # Random saturation
        if random.random() < p:
            saturation_factor = random.uniform(0.8, 1.2)
            img_tensor = F.adjust_saturation(img_tensor, saturation_factor)
        
        # Random hue
        if random.random() < p:
            hue_factor = random.uniform(-0.1, 0.1)
            img_tensor = F.adjust_hue(img_tensor, hue_factor)
        
        pil_img = F.to_pil_image(img_tensor)
        return pil_img


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
