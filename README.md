# Medical Image Segmentation Benchmark

A flexible and extensible framework for benchmarking different segmentation models on ODOC (Optic Disc and Optic Cup) and vessel segmentation tasks.

## Project Structure

```
.
├── models/
│   ├── __init__.py      # Model registry and factory functions
│   ├── unet.py          # UNet implementation
│   ├── swinunetr.py     # SwinUNETR implementation
│   └── unetr.py         # UNETR implementation
├── utils/
│   ├── __init__.py      # Utility exports
│   ├── cli.py           # Command line interface utilities
│   ├── data.py          # Dataset and dataloader utilities
│   ├── io.py            # File and directory operations
│   ├── loss.py          # Loss functions
│   ├── metrics.py       # Metrics and evaluation
│   ├── model.py         # Model handling utilities
│   ├── transforms.py    # Data transformations
│   └── visualization.py # Visualization utilities
├── train.py             # Training script
├── predict.py           # Prediction script
├── train_odoc.sh        # ODOC training script
├── train_vessel.sh      # Vessel training script
├── predict_odoc.sh      # ODOC prediction script
└── predict_vessel.sh    # Vessel prediction script
```

## Features

- Multiple segmentation models:
  - UNet with configurable backbone
  - SwinUNETR with vision transformer
  - UNETR for transformers-based segmentation
- Advanced loss functions:
  - Dice Loss
  - Combined BCE-Dice Loss
  - Focal Loss
  - IoU Loss
  - DIoU Loss
- Comprehensive evaluation metrics:
  - Dice coefficient
  - IoU (Intersection over Union)
  - Surface Distance
  - Hausdorff Distance
  - Confusion Matrix metrics
- Rich visualization features:
  - Colorized overlays (orange/red for ODOC, blue for vessels)
  - Training curves for loss and metrics
  - Progress bars with real-time metrics
- Support for both tasks:
  - ODOC (Optic Disc and Cup Segmentation)
  - Vessel Segmentation
- Modular and extensible design
- Automatic dataset detection
- Command-line interface with extensive options
- Efficient data loading and processing

## Installation

```bash
# Clone the repository
git clone https://github.com/Vidhyotha/Segmentation_Benchmark.git
cd segmentation-benchmark

# Create and activate conda environment
conda env create -f environment.yml
conda activate segmentation
```

## Usage

### Quick Start

For ODOC segmentation:
```bash
# Training
./train_odoc.sh

# Prediction
./predict_odoc.sh
```

For vessel segmentation:
```bash
# Training
./train_vessel.sh

# Prediction
./predict_vessel.sh
```

### Detailed Usage

#### Training

```bash
python train.py \
    --task odoc \                    # or 'vessel'
    --model unet \                   # unet, swinunetr, unetr
    --dataset drishti \              # dataset name in data/
    --batch_size 8 \
    --val_batch_size 4 \
    --epochs 100 \
    --lr 3e-4 \
    --loss dicece \                  # dice, dicece, focal, iou, diou
    --metric dice \                  # dice, hausdorff, surface_distance, iou
    --image_size 1024 \
    --device cuda:0 \
    --seed 42
```

#### Prediction

```bash
python predict.py \
    --task odoc \                    # or 'vessel'
    --model unet \                   # must match trained model
    --dataset drishti \
    --model_path weights/best_unet_model.pth \
    --image_size 1024 \
    --device cuda:0 \
    --metric dice
```

## Dataset Structure

```
data/
├── drishti/                    # Example dataset
│   ├── images/
│   │   ├── 19_DRISHTI_001.png
│   │   └── ...
│   └── masks/
│       ├── odoc/              # For ODOC task
│       │   ├── 19_DRISHTI_001.png  # White(bg)/Gray(disc)/Black(cup)
│       │   └── ...
│       └── vessel/            # For vessel task
│           ├── 19_DRISHTI_001.png  # White(bg)/Black(vessel)
│           └── ...
```

- Images: RGB format in common formats (png, jpg)
- ODOC masks: 8-bit grayscale with:
  - 255 (white) = background
  - 128 (gray) = optic disc
  - 0 (black) = optic cup
- Vessel masks: Binary with:
  - 255 (white) = background
  - 0 (black) = vessel

## Directory Structure

- `outputs/`: Training outputs (one directory per experiment)
  - Training curves
  - Configuration files
  - TensorBoard logs
- `weights/`: Model weights
  - Best model weights
  - Latest model weights
  - Model checkpoints
- `results/`: Prediction results
  - Binary masks
  - Colorized overlays
  - Evaluation metrics

## Advanced Features

### Visualization
- Real-time training progress with colorized metrics
- Automatic overlay generation for predictions
- Training curves showing loss and metrics
- TensorBoard integration for detailed monitoring

### Model Management
- Automatic checkpoint saving
- Best model tracking
- Easy model loading with fallbacks
- Symlinks to latest weights

### Evaluation
- Comprehensive metrics for both tasks
- Per-class metrics calculation
- Support for multiple ground truth formats
- Detailed evaluation reports


## License

This project is licensed under the MIT License - see the LICENSE file for details.
