from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class PredictAfterValidationCallback(Callback):
    def __init__(self, logger, num_images_to_log=8):
        super().__init__()
        self.logger = logger
        self.num_images_to_log = num_images_to_log
        self.colors = plt.cm.get_cmap('tab20', 20).colors * 255  # Colors for different classes

    def setup(self, trainer, pl_module, stage):
        if stage in ("fit", "validate"):
            trainer.datamodule.setup("predict")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        val_dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_dataloader))
        batch = pl_module.transfer_batch_to_device(batch, trainer.strategy.root_device, 0)

        # Get model predictions
        outputs = pl_module._model(**batch)
        
        # Convert outputs to readable format
        processed_outputs = pl_module.processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=[torch.tensor(img.shape[1:]) for img in batch["pixel_values"]]
        )

        # Denormalize images
        images = self.denormalize(batch['pixel_values'])
        
        # Log images with predictions
        for i, (image, detections) in enumerate(zip(images[:self.num_images_to_log], processed_outputs[:self.num_images_to_log])):
            img_with_boxes = self.draw_detections(
                image.permute(1, 2, 0).cpu().numpy(),
                detections['boxes'].cpu().numpy(),
                detections['labels'].cpu().numpy(),
                detections['scores'].cpu().numpy()
            )
            
            self.logger.report_image(
                title='validation_predictions',
                series=f'epoch_{trainer.current_epoch}_img_{i}',
                iteration=trainer.current_epoch,
                image=img_with_boxes
            )

    @staticmethod
    def denormalize(x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device)
        return torch.clamp((x * std) + mean, 0, 1)

    def draw_detections(self, image, boxes, labels, scores):
        """Draw bounding boxes and labels on image"""
        # Convert numpy array to PIL Image
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        
        for box, label, score in zip(boxes, labels, scores):
            # Convert box from [xmin, ymin, xmax, ymax] to PIL format
            box = [float(coord) for coord in box]
            color = tuple(self.colors[label % len(self.colors)].astype(int).tolist())
            
            # Draw rectangle
            draw.rectangle(box, outline=color, width=2)
            
            # Draw label text
            text = f"{label}: {score:.2f}"
            text_position = (box[0], box[1] - 10)
            draw.text(text_position, text, fill=color)
        
        return np.array(image)