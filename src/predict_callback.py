from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import cv2

class PredictAfterValidationCallback(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def setup(self, trainer, pl_module, stage):
        if stage in ("fit", "validate"):
            # setup the predict data even for fit/validate, as we will call it during `on_validation_epoch_end`
            trainer.datamodule.setup("predict")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:  # optional skip
            return

        val_dataloader = trainer.datamodule.val_dataloader()
        batch = pl_module.transfer_batch_to_device(next(iter(val_dataloader)), trainer.strategy.root_device, 0)

        outputs = pl_module.predict_step(batch)
        images = self.denormalize(batch['pixel_values'])
        for i, out in enumerate(outputs[:]):
            image = images[i].permute(1,2,0).cpu().numpy()
            masked_image = self.draw_random_masks(image, out['segmentation'].cpu().numpy())

            self.logger.report_image(title='validation_epoch', series=f'{i}_image', iteration=trainer.current_epoch, image=image)
            self.logger.report_image(title='validation_epoch', series=f'{i}_mask', iteration=trainer.current_epoch, image=masked_image)

    @staticmethod
    def denormalize(x):
        mean=np.array([123.675, 116.280, 103.530]) / 255
        std=np.array([58.395, 57.120, 57.375]) / 255

        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    @staticmethod
    def draw_random_masks(image, masks):
        alpha = 0.3
        masked_image = image.copy()
        max_value = 255
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, axis=0)
        for mask in masks:
            masked_image = np.where(
                np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                np.random.randint(0, max_value, size=(3)),
                masked_image,
            )
            masked_image = masked_image.astype(np.uint8)
        return cv2.addWeighted(image, alpha, masked_image, 1 - alpha, 0, dtype=cv2.CV_8U)
