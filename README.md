# Medical Image Segmentation Benchmark

A flexible and extensible framework for benchmarking different segmentation models on ODOC (Optic Disc and Optic Cup) and vessel segmentation tasks.

## Project Structure

```
.
├── models/
│   ├── __init__.py      # Model registry and factory functions
│   ├── base_model.py    # Base model class
│   └── unet.py          # UNet implementation
├── utils/
│   ├── loss.py          # Various loss functions
│   └── metrics.py       # Evaluation metrics
├── train.py             # Training script
├── inference.py         # Inference script
└── requirements.txt     # Project dependencies
```

## Features

- Multiple segmentation models support (UNet, with extensibility for SwinUNet, CS2Net, etc.)
- Various loss functions (Dice, BCE-Dice, Focal)
- Common evaluation metrics (Dice, IoU, Accuracy, Sensitivity, Specificity)
- Support for both ODOC and vessel segmentation tasks
- Flexible training configuration through command-line arguments
- Easy inference with optional overlay visualization
- Evaluation against ground truth masks

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd segmentation-benchmark

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train a model using the following command:

```bash
python train.py \
    --task odoc \                     # or 'vessel'
    --data_dir path/to/dataset \
    --model unet \
    --batch_size 8 \
    --epochs 100 \
    --loss dice \                     # Available: dice, bce_dice, focal, bce
    --metrics dice iou accuracy \
    --output_dir outputs \
    --exp_name experiment_1
```

Available arguments:
- `--task`: Choose between 'odoc' or 'vessel' segmentation
- `--model`: Model architecture to use (default: 'unet')
- `--loss`: Loss function to use
- `--metrics`: Evaluation metrics to compute
- `--optimizer`: Optimizer to use (adam, sgd, adamw)
- See `train.py` for full list of arguments

### Inference

Run inference on new images:

```bash
python inference.py \
    --task odoc \                     # or 'vessel'
    --input path/to/images \          # Single image or directory
    --model unet \
    --model_weights path/to/weights \
    --save_overlay \                  # Optional: save predictions overlaid on input
    --output_dir predictions
```

For evaluation against ground truth:

```bash
python inference.py \
    --task odoc \
    --input path/to/images \
    --model_weights path/to/weights \
    --gt_dir path/to/ground_truth \
    --metrics dice iou
```

## Adding New Components

### Models

1. Create a new model file in `models/`
2. Inherit from `BaseModel`
3. Register model in `models/__init__.py`

Example:
```python
from .base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        # Model implementation

    def forward(self, x):
        # Forward pass implementation
        return x

# In models/__init__.py:
register_model('new_model', NewModel)
```

### Loss Functions

Add new loss functions in `utils/loss.py` and register them in the `get_loss_function` factory function.

### Metrics

Add new metrics in `utils/metrics.py` and register them in the `get_metrics` factory function.

## Expected Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

- Images should be in common formats (jpg, png)
- Masks should be binary images (0 or 255 pixel values)
- For ODOC: use 2-channel masks (OD and OC)
- For vessels: use single-channel masks

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
