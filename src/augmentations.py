import albumentations as albu
from albumentations.pytorch import ToTensorV2


# https://www.kaggle.com/code/artgor/object-detection-with-pytorch-lightning
def get_transforms(
        width: int,
        height: int,
        preprocessing: bool = True,
        augmentations: bool = True,
        postprocessing: bool = True,
) -> albu.Compose:
    transforms: list = []

    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))
    
    if augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5,
                ),
                albu.RandomRain(p=0.5),
                albu.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                albu.ShiftScaleRotate(),
                albu.GaussianBlur(),
            ],
        )
    
    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])
    
    return albu.Compose(transforms)
