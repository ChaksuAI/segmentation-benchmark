#!/usr/bin/env python3
"""
Medical Image Segmentation Training Script
Supports multiple models and datasets for optic disc/cup segmentation.
"""
import os
import sys
import logging
import time
from tabulate import tabulate
import torch
import monai
from monai.transforms import AsDiscrete, Compose
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from colorama import Fore, Style

# PyTorch 2.6 fix
import torch.serialization
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])

# Local imports
from models import get_model
from utils.transforms import get_train_transforms, get_val_transforms
from utils.loss import get_loss_function
from utils.metrics import get_metric
from utils.data import get_datasets, get_data_loaders, get_combined_datasets
from utils.cli import parse_args, print_header, print_config
from utils.model import generate_default_config, save_model, create_symlink, initialize_model
from utils.visualization import plot_training_curves, log_epoch_metrics
from utils.io import create_output_directories, save_config, get_dataset_path

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def train(args, config, available_datasets):
    """Main training function."""
    # Setup logging
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Check for multiple datasets
    if ',' in args.datasets:
        using_multiple_datasets = True
        print(f"{Fore.YELLOW}Using multiple datasets: {args.datasets}")
        # We'll set data_dir to None since we're using multiple
        data_dir = None
    else:
        using_multiple_datasets = False
        # Get single dataset path as before
        data_dir = get_dataset_path(available_datasets, args.datasets)
    
    # Set device
    device = torch.device(args.device)
    
    # Print welcome message and system info
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_header(f"Starting training at {current_time}")
    print_config(args, available_datasets, config)
    
    print_header("System Information")
    system_info = [
        ["PyTorch version", torch.__version__],
        ["CUDA available", torch.cuda.is_available()],
        ["CUDA version", torch.version.cuda if torch.cuda.is_available() else "N/A"],
        ["Selected device", args.device],
        ["CUDA device name", torch.cuda.get_device_name(device.index) if device.type == 'cuda' else "N/A"],
        ["CUDA memory", f"{torch.cuda.get_device_properties(device.index).total_memory / 512**3:.1f} GB" 
            if device.type == 'cuda' else "N/A"],
        ["MONAI version", monai.__version__],
        ["Dataset", f"{args.datasets} ({data_dir})"],
    ]
    print(tabulate(system_info, tablefmt="fancy_grid"))
    print()
    
    # Set deterministic training
    if args.seed is not None:
        monai.utils.set_determinism(seed=args.seed)
    
    # Create transforms and datasets
    transforms = {
        "train": get_train_transforms(config),
        "val": get_val_transforms(config)
    }
    
    print_header("Loading Datasets")
    start_time = time.time()
    
    if using_multiple_datasets:
        # Use new function to load multiple datasets
        train_ds, val_ds = get_combined_datasets(args.datasets, available_datasets, 
                                              transforms, 0.8, args.task)
    else:
        # Use existing function for single dataset
        train_ds, val_ds = get_datasets(data_dir, transforms, 0.8, args.task)
    
    batch_sizes = {"train": args.batch_size, "val": args.val_batch_size}
    train_loader, val_loader = get_data_loaders(train_ds, val_ds, batch_sizes)
    data_load_time = time.time() - start_time
    
    print(f"{Fore.GREEN}✓ Datasets loaded in {data_load_time:.2f} seconds")
    print(f"  • Training samples: {Fore.YELLOW}{len(train_ds)}")
    print(f"  • Validation samples: {Fore.YELLOW}{len(val_ds)}")
    print()
    
    # Initialize model
    print_header("Creating Model")
    model = initialize_model(args, config, device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{Fore.GREEN}✓ Model {Fore.CYAN}{args.model}{Fore.GREEN} created")
    print(f"  • Parameters: {Fore.YELLOW}{num_params:,}")
    print(f"  • Device: {Fore.YELLOW}{device}")
    print(f"{Fore.YELLOW}• Mixed precision training disabled")
    print()
    
    # Setup training components
    loss_function = get_loss_function(args.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=config.get("weight_decay", 1e-5)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    metric = get_metric(args.metric)
    num_classes = 2 if args.task == "vessel" else 3
    post_trans = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    
    # Create output directories
    output_dir, weights_dir = create_output_directories(args, timestamp)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))
    
    # Initialize tracking variables
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    
    # Save configuration
    config_path = os.path.join(output_dir, "training_config.json")
    save_config(config_path, args, config)
    
    # Start training loop
    print_header("Training")
    overall_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        epoch_loss = 0
        step = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"{Fore.GREEN}Training",
            unit="batch",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}"
        )
        
        for batch_data in progress_bar:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar("train_loss", loss.item(), len(train_loader) * epoch + step)
        
        scheduler.step()
        
        # Compute epoch metrics
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time
        
        # Log training metrics
        log_epoch_metrics(args, epoch, args.epochs, epoch_loss, scheduler, epoch_time)
        
        # Validation phase
        val_start_time = time.time()
        model.eval()
        
        val_progress = tqdm(
            val_loader, 
            desc=f"{Fore.BLUE}Validation",
            unit="batch",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}"
        )
        
        with torch.no_grad():
            for val_data in val_progress:
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                
                val_outputs = model(val_images)
                val_outputs = [post_trans(i) for i in monai.data.decollate_batch(val_outputs)]
                metric(y_pred=val_outputs, y=val_labels)
            
            # Aggregate validation metrics
            metric_value = metric.aggregate().item()
            metric.reset()
            
            metric_values.append(metric_value)
            val_time = time.time() - val_start_time
            
            # Check for best model
            is_best = metric_value > best_metric
            if is_best:
                best_metric = metric_value
                best_metric_epoch = epoch + 1
                best_indicator = save_model(model, args, weights_dir, True, config_path)
            else:
                best_indicator = f"{Fore.YELLOW}(best: {best_metric:.4f} @ epoch {best_metric_epoch})"
            
            print(f"  {Fore.BLUE}• Validation {args.metric.capitalize()}: {Fore.YELLOW}{metric_value:.4f} {best_indicator}")
            print(f"  {Fore.BLUE}• Validation time: {Fore.YELLOW}{val_time:.2f} seconds")
            
            # Log to TensorBoard
            writer.add_scalar(f"val_mean_{args.metric}", metric_value, epoch + 1)
            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch + 1)
    
    # Training complete - print summary
    total_time = time.time() - overall_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_header("Training Complete")
    print(f"{Fore.GREEN}• Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"{Fore.GREEN}• Best validation {args.metric.capitalize()}: {Fore.YELLOW}{best_metric:.4f} {Fore.GREEN}(epoch {best_metric_epoch})")
    
    # Plot training curves
    plot_training_curves(epoch_loss_values, metric_values, output_dir)
    
    writer.close()
    
    # Save final model
    save_model(model, args, weights_dir)
    create_symlink(args, weights_dir)
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Training completed successfully!")

if __name__ == "__main__":
    args, available_datasets = parse_args("Train a model for optic disc/cup segmentation")
    config = generate_default_config(args)
    train(args, config, available_datasets)
