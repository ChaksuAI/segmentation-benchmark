# Medical Image Segmentation Benchmark

A flexible and extensible framework for benchmarking different segmentation models on ODOC (Optic Disc and Optic Cup) and vessel segmentation tasks.

## Project Structure

```
.
├── data/                                     # All datasets are stored here, structure explained later
├── outputs/                                  # Saves training config, and training curves of each model
├── results/                                  # Saves all the ouputs of the predictions
│   ├── odoc/
│   │   ├── unet_drishti_20250320_124438/     # model_dataset_date_time
│   │   │   ├── masks/                        # Predicted masks are stored here
│   │   │   ├── overlays/                     # Predicted masks are overlayed on original images and stored here
│   │   │   └── evaluation_results.txt        # Evaluated metrics are stored here
│   │   └── ...
│   └── vessel/
│       └── ...
├── weights/                                  # Model weights are saved here
├── models/
│   ├── __init__.py                           # Model registry and factory functions
│   ├── unet.py                               # UNet implementation
│   ├── swinunetr.py                          # SwinUNETR implementation
│   ├── unetr.py                              # UNETR implementation
│   └── ...                                   # Other models
├── utils/
│   ├── __init__.py                           # Utility exports
│   ├── cli.py                                # Command line interface utilities
│   ├── data.py                               # Dataset and dataloader utilities
│   ├── io.py                                 # File and directory operations
│   ├── loss.py                               # Loss functions
│   ├── metrics.py                            # Metrics and evaluation
│   ├── model.py                              # Model handling utilities
│   ├── transforms.py                         # Data transformations
│   └── visualization.py                      # Visualization utilities
├── train.py                                  # Training script
├── predict.py                                # Prediction script
├── train_odoc.sh                             # ODOC training script
├── train_vessel.sh                           # Vessel training script
├── predict_odoc.sh                           # ODOC prediction script
└── predict_vessel.sh                         # Vessel prediction script
```

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
    --model unet \                   # unet, unetr, swinunetr, etc.
    --dataset drishti \              # dataset name in data/
    --batch_size 8 \
    --val_batch_size 4 \
    --epochs 100 \
    --lr 3e-4 \
    --loss dicece \                  
    --metric dice \                  
    --image_size 512 \
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
    --image_size 512 \
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

- Images: RGB format in common formats
- ODOC masks: 8-bit grayscale with:
  - 255 (white) = background
  - 128 (gray) = optic disc
  - 0 (black) = optic cup
- Vessel masks: Binary with:
  - 255 (white) = background
  - 0 (black) = vessel
