import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse
from datetime import datetime
import torch


def create_dataset_yaml(data_dir, output_path='dataset.yaml'):
    """
    Create YOLO dataset configuration file.
    
    Args:
        data_dir: Root directory containing train/val/test splits
        output_path: Path to save YAML configuration
    """
    data_config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes (single class: Cat)
        'names': ['Cat']  # Class names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to {output_path}")
    return output_path


def train_yolo(data_yaml, model_size='n', epochs=100, img_size=640, batch_size=16,
               device=None, project='runs/train', name='cat_detection', 
               pretrained=True, optimizer='auto', lr0=0.01, save_period=10):
    """
    Train YOLOv8 model for cat detection.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
        device: Device to use ('cpu', 'cuda', '0', etc.)
        project: Project directory for saving results
        name: Name of the training run
        pretrained: Whether to use pretrained weights
        optimizer: Optimizer to use ('SGD', 'Adam', 'AdamW', 'auto')
        lr0: Initial learning rate
        save_period: Save checkpoint every N epochs
        
    Returns:
        Training results
    """
    # Check device
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*50)
    print("YOLOv8 Training Configuration")
    print("="*50)
    print(f"Model size: YOLOv8{model_size}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Pretrained: {pretrained}")
    print(f"Optimizer: {optimizer}")
    print(f"Initial learning rate: {lr0}")
    print("="*50 + "\n")
    
    # Load model
    model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
    model = YOLO(model_name)
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        pretrained=pretrained,
        optimizer=optimizer,
        lr0=lr0,
        save_period=save_period,
        plots=True,
        cache=True,  # Cache images for faster training
        workers=8,  # Number of dataloader workers
        # Data augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=0.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,    # Scaling augmentation
        shear=0.0,    # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,   # Vertical flip probability
        fliplr=0.5,   # Horizontal flip probability
        mosaic=1.0,   # Mosaic augmentation probability
        mixup=0.0,    # MixUp augmentation probability
        copy_paste=0.0,  # Copy-paste augmentation probability
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Last model saved at: {results.save_dir}/weights/last.pt")
    print("="*50)
    
    return results


def evaluate_model(model_path, data_yaml, img_size=640, device=None, split='val'):
    """
    Evaluate trained model on validation or test set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML configuration
        img_size: Input image size
        device: Device to use for evaluation
        split: Dataset split to evaluate on ('val' or 'test')
        
    Returns:
        Evaluation metrics
    """
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*50)
    print(f"Model Evaluation on {split.upper()} set")
    print("="*50)
    
    # Load model
    model = YOLO(model_path)
    
    # Evaluate on specified split
    metrics = model.val(
        data=data_yaml,
        split=split,  # Specify which split to use
        imgsz=img_size,
        device=device,
        plots=True
    )
    
    # Print metrics
    print(f"\n{split.upper()} Set Evaluation Metrics:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("="*50)
    
    return metrics


def main():
    """Main function to train YOLOv8 model."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for cat detection')
    
    # Dataset arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Root directory containing data (with train/val/test folders)')
    
    # Model arguments
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=extra large)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--optimizer', type=str, default='auto',
                       choices=['SGD', 'Adam', 'AdamW', 'auto'],
                       help='Optimizer to use')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='cat_detection',
                       help='Name of the training run')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for evaluation (if not training)')
    parser.add_argument('--eval-split', type=str, default='val',
                       choices=['val', 'test'],
                       help='Dataset split to evaluate on (val or test). Use test only for final evaluation!')
    
    args = parser.parse_args()
    
    # Create dataset YAML
    dataset_yaml = create_dataset_yaml(args.data_dir)
    
    # Train model
    if not args.model_path:
        results = train_yolo(
            data_yaml=dataset_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device,
            project=args.project,
            name=args.name,
            pretrained=args.pretrained,
            optimizer=args.optimizer,
            lr0=args.lr0,
            save_period=args.save_period
        )
        
        # Get best model path
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    else:
        best_model_path = args.model_path
    
    # Evaluate model
    if args.evaluate:
        evaluate_model(
            model_path=str(best_model_path),
            data_yaml=dataset_yaml,
            img_size=args.img_size,
            device=args.device,
            split=args.eval_split  # Use specified split (val or test)
        )


if __name__ == '__main__':
    main()
