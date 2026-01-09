"""
Standalone model evaluation script for YOLOv8 cat detection.

This script evaluates a trained model on validation or test set.
Use this for final test set evaluation after training is complete.
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def evaluate_model(model_path, data_yaml, split='test', img_size=640, device=None, conf_threshold=0.25):
    """
    Evaluate trained model on specified dataset split.
    
    Args:
        model_path: Path to trained model weights (.pt file)
        data_yaml: Path to dataset YAML configuration
        split: Dataset split to evaluate ('val' or 'test')
        img_size: Input image size
        device: Device to use for evaluation
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        Evaluation metrics dictionary
    """
    # Auto-detect device
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print(f"YOLOv8 Model Evaluation - {split.upper()} SET")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Split: {split}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {conf_threshold}")
    print("="*70 + "\n")
    
    # Warning for test set evaluation
    if split == 'test':
        print("⚠️  WARNING: You are evaluating on the TEST set!")
        print("   This should only be done ONCE at the end of your project.")
        print("   Use validation set for model selection and hyperparameter tuning.\n")
    
    # Load model
    print("Loading model...")
    model = YOLO(model_path)
    
    # Evaluate on specified split
    print(f"Evaluating on {split} set...")
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=img_size,
        device=device,
        conf=conf_threshold,
        iou=0.6,
        plots=True,
        save_json=True
    )
    
    # Print detailed metrics
    print("\n" + "="*70)
    print(f"{split.upper()} SET EVALUATION RESULTS")
    print("="*70)
    print(f"\n📊 Detection Metrics:")
    print(f"   mAP@0.5       : {metrics.box.map50:.4f}   (0.85+ is good)")
    print(f"   mAP@0.5:0.95  : {metrics.box.map:.4f}   (0.60+ is good)")
    print(f"   Precision     : {metrics.box.mp:.4f}   (0.80+ is good)")
    print(f"   Recall        : {metrics.box.mr:.4f}   (0.75+ is good)")
    
    # Calculate F1 score
    if metrics.box.mp > 0 and metrics.box.mr > 0:
        f1_score = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
        print(f"   F1 Score      : {f1_score:.4f}   (0.75+ is good)")
    
    print(f"\n📈 Per-Class Metrics:")
    print(f"   Class: Cat")
    if hasattr(metrics.box, 'ap_class_index'):
        print(f"   AP@0.5: {metrics.box.ap50[0]:.4f}")
        print(f"   AP@0.5:0.95: {metrics.box.ap[0]:.4f}")
    
    # Model speed
    if hasattr(metrics, 'speed'):
        print(f"\n⚡ Inference Speed:")
        print(f"   Preprocessing : {metrics.speed['preprocess']:.1f}ms")
        print(f"   Inference     : {metrics.speed['inference']:.1f}ms")
        print(f"   Postprocessing: {metrics.speed['postprocess']:.1f}ms")
    
    print("\n" + "="*70)
    print("Evaluation complete! Check the plots in the results directory.")
    print("="*70 + "\n")
    
    return metrics


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv8 cat detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on validation set
  python evaluate_model.py --model runs/train/cat_detection/weights/best.pt --split val
  
  # Evaluate on test set (final evaluation only!)
  python evaluate_model.py --model runs/train/cat_detection/weights/best.pt --split test
  
  # Evaluate with custom confidence threshold
  python evaluate_model.py --model best.pt --split test --conf 0.5
        """
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    
    # Dataset arguments
    parser.add_argument('--data', type=str, default='../dataset.yaml',
                       help='Path to dataset YAML configuration')
    parser.add_argument('--split', type=str, default='test',
                       choices=['val', 'test'],
                       help='Dataset split to evaluate on (default: test)')
    
    # Evaluation parameters
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions (default: 0.25)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, 0, 1, etc.)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        print("   Please provide a valid path to a trained model.")
        return
    
    # Check if dataset YAML exists
    if not Path(args.data).exists():
        print(f"❌ Error: Dataset YAML not found: {args.data}")
        print("   Please provide a valid path to dataset.yaml")
        return
    
    # Run evaluation
    try:
        metrics = evaluate_model(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            img_size=args.img_size,
            device=args.device,
            conf_threshold=args.conf
        )
        
        # Save metrics summary
        output_dir = Path(args.model).parent.parent
        metrics_file = output_dir / f'{args.split}_metrics.txt'
        
        with open(metrics_file, 'w') as f:
            f.write(f"{args.split.upper()} Set Evaluation Metrics\n")
            f.write("="*50 + "\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
            f.write(f"Precision: {metrics.box.mp:.4f}\n")
            f.write(f"Recall: {metrics.box.mr:.4f}\n")
            if metrics.box.mp > 0 and metrics.box.mr > 0:
                f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
                f.write(f"F1 Score: {f1:.4f}\n")
        
        print(f"✅ Metrics saved to: {metrics_file}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise


if __name__ == '__main__':
    main()
