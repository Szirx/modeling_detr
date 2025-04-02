# Modeling_detr

End-to-End Object Detection with Transformers (DETR) implementation with PyTorch Lightning.

## Features

- 🚀 Training and evaluation of DETR models
- ⚡ PyTorch Lightning integration
- 📊 Comprehensive metrics tracking
- 🛠️ Easy configuration via YAML files
- 🔄 Support for both pretrained and custom models

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Szirx/modeling_detr.git
    cd modeling_detr
    ```

2. Install dependencies:

    ```bash
    make install
    # or manually:
    pip install -r requirements.txt
    ```

## Quick start

```bash
make train
# or directly:
python train.py --config configs/config.yaml
```

## Inference example

```python
from modeling_detr import DetrLightning
from transformers import DetrImageProcessor

# Load pretrained model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrLightning.load_from_checkpoint("checkpoints/last.ckpt")

# Prepare image
image = Image.open("example.jpg")

# Process and predict
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Post-process results
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, 
    target_sizes=target_sizes, 
    threshold=0.7
)[0]

# Display results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"Detected {model.config.id2label[label.item()]} with confidence {score.item():.2f} at {box.tolist()}")
```

## Configuration

Modify configs/config.yaml to customize training:

- [Configuration](./configs/config.yaml)

## Acknowledgements

- Original DETR paper: End-to-End Object Detection with Transformers
- Hugging Face Transformers implementation
- PyTorch Lightning team
