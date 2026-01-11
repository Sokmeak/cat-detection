import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Evaluate YOLO model on test set")
parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    help="Model folder name under runs/train (e.g., cat_original)"
)
parser.add_argument(
    "--data_config",
    type=str,
    default="data/original",
    help="Dataset directory path (e.g., data/original or data/enhanced)"
)
parser.add_argument(
    "--imgsz",
    type=int,
    default=640,
    help="Image size for evaluation"
)

args = parser.parse_args()

# ----------------------------
# Load model
# ----------------------------
model_path = PROJECT_ROOT / "runs" / "train" / args.model_type / "weights" / "best.pt"

print(f"Loading model from: {model_path}")
model = YOLO(str(model_path))

# ----------------------------
# Evaluate on TEST set
# ----------------------------
# Create a temporary data config pointing to the specified dataset
data_dir = PROJECT_ROOT / args.data_config

data_config = {
    'path': str(data_dir.resolve()),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 1,
    'names': ['Cat']
}

# Write temporary config
temp_config_path = PROJECT_ROOT / 'temp_test_config.yaml'
with open(temp_config_path, 'w') as f:
    yaml.dump(data_config, f)

metrics = model.val(
    data=str(temp_config_path),
    split="test",
    imgsz=args.imgsz
)

# Clean up temp file
os.remove(temp_config_path)

# ----------------------------
# Print results
# ----------------------------
print("\n" + "="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)
print(f"Model type: {args.model_type}")
print(f"mAP@0.5      : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
print(f"Precision    : {metrics.box.mp:.4f}")
print(f"Recall       : {metrics.box.mr:.4f}")
print("="*60 + "\n")

# ----------------------------
# Save results to file
# ----------------------------
results_file = model_path.parent / "test_set_metrics.txt"
with open(results_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("TEST SET EVALUATION RESULTS\n")
    f.write("="*60 + "\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Dataset: {args.data_config}\n")
    f.write(f"Image size: {args.imgsz}\n\n")
    f.write("METRICS:\n")
    f.write(f"  mAP@0.5      : {metrics.box.map50:.4f}\n")
    f.write(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}\n")
    f.write(f"  Precision    : {metrics.box.mp:.4f}\n")
    f.write(f"  Recall       : {metrics.box.mr:.4f}\n")
    f.write("="*60 + "\n")

print(f"✅ Metrics saved to: {results_file}")
