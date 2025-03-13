import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
from models import get_model
from utils.loss import get_loss_function
from utils.metrics import get_metrics
from utils.data import create_dataloaders
from tqdm import tqdm
import time
from datetime import timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for segmentation')
    
    # Dataset parameters
    parser.add_argument('--task', type=str, required=True, choices=['odoc', 'vessel'],
                      help='Segmentation task (odoc or vessel)')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (e.g., drive, chase, stare)')
    parser.add_argument('--train_split', type=float, default=0.8,
                      help='Train/val split ratio')
    parser.add_argument('--img_size', type=int, default=1024,
                      help='Input image size')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='unet',
                      help='Model architecture to use')
    parser.add_argument('--pretrained', type=str, default=None,
                      help='Path to pretrained weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--loss', type=str, default='dice',
                      help='Loss function to use')
    parser.add_argument('--metrics', nargs='+', default=['dice', 'iou'],
                      help='Metrics to evaluate')
    parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'sgd', 'adamw'],
                      help='Optimizer to use')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    parser.add_argument('--gpu_id', type=int, default=0,
                      help='ID of GPU to use when multiple GPUs are available')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--exp_name', type=str, default=None,
                      help='Experiment name for logging')
    
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device, metrics):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    metric_values = {name: 0.0 for name in metrics.keys()}
    
    # Add tqdm progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            batch_metrics = {}
            for name, metric_fn in metrics.items():
                value = metric_fn(output, target).item()
                metric_values[name] += value
                batch_metrics[name] = value
        
        # Update progress bar with current metrics
        pbar.set_postfix(loss=f"{loss.item():.4f}", **{k: f"{v:.4f}" for k, v in batch_metrics.items()})
    
    # Average over batches
    num_batches = len(train_loader)
    epoch_loss /= num_batches
    for name in metric_values:
        metric_values[name] /= num_batches
    
    return epoch_loss, metric_values

def validate(model, val_loader, criterion, device, metrics):
    """Validate the model"""
    model.eval()
    val_loss = 0
    metric_values = {name: 0.0 for name in metrics.keys()}
    
    # Add tqdm progress bar
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = criterion(output, target).item()
            val_loss += batch_loss
            
            # Calculate metrics
            batch_metrics = {}
            for name, metric_fn in metrics.items():
                value = metric_fn(output, target).item()
                metric_values[name] += value
                batch_metrics[name] = value
            
            # Update progress bar with current metrics
            pbar.set_postfix(loss=f"{batch_loss:.4f}", **{k: f"{v:.4f}" for k, v in batch_metrics.items()})
    
    # Average over batches
    num_batches = len(val_loader)
    val_loss /= num_batches
    for name in metric_values:
        metric_values[name] /= num_batches
    
    return val_loss, metric_values

def main():
    args = parse_args()
    
    # Set specific GPU if requested
    if args.device == 'cuda':
        device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(args.gpu_id)
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Construct data directory path from dataset name
    data_dir = Path('data') / args.dataset
    
    # Create output directory
    if args.exp_name is None:
        args.exp_name = f"{args.dataset}_{args.task}_{args.model}_{args.loss}"
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up model
    if args.task == 'odoc':
        n_classes = 3  # Background, OD and OC
    else:  # vessel
        n_classes = 1
    
    model = get_model(args.model, n_classes=n_classes)
    
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    model = model.to(device)
    
    # Set up loss function and metrics
    criterion = get_loss_function(args.loss)
    metrics = get_metrics(args.metrics)
    
    # Set up optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:  # adamw
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir,  # Use constructed data_dir instead of args.data_dir
        args.task, 
        args.img_size, 
        args.batch_size, 
        args.train_split,
        args.num_workers
    )
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    print("\n" + "="*80)
    print(f"Starting training: {args.dataset}_{args.task} with {args.model} model")
    print(f"Training for {args.epochs} epochs with {args.optimizer} optimizer (lr={args.lr})")
    print(f"Loss function: {args.loss}, Metrics: {', '.join(args.metrics)}")
    print("="*80 + "\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, metrics
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, metrics
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            saved_text = "✓ Saved new best model"
        else:
            saved_text = ""
        
        # Calculate epoch time and estimate remaining time
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        estimated_total = (epoch_time * args.epochs) / (epoch + 1)
        estimated_remaining = estimated_total - elapsed
        
        # Log metrics with pretty formatting
        print(f"\n----- Epoch {epoch+1}/{args.epochs} | Time: {timedelta(seconds=int(epoch_time))} -----")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {saved_text}")
        
        # Print metrics in a table format
        print("\n{:<10} {:<15} {:<15}".format("Metric", "Train", "Validation"))
        print("-" * 40)
        for name in metrics:
            print("{:<10} {:<15.4f} {:<15.4f}".format(
                name, train_metrics[name], val_metrics[name]))
        
        # Print time information
        print(f"\nElapsed: {timedelta(seconds=int(elapsed))}, "
              f"Remaining: {timedelta(seconds=int(estimated_remaining))}\n")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, output_dir / 'last_checkpoint.pth')

if __name__ == '__main__':
    main()
