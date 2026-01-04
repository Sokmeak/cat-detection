"""
Export FiftyOne Dataset to YOLO Format
Converts Open Images dataset from FiftyOne to YOLO format for training.
"""

import os
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2


def export_fiftyone_to_yolo(dataset, output_dir, split_name):
    """
    Export FiftyOne dataset to YOLO format.
    
    Args:
        dataset: FiftyOne dataset
        output_dir: Output directory for YOLO format data
        split_name: Name of split (train/val/test)
    """
    # Create directories
    images_dir = Path(output_dir) / split_name / 'images'
    labels_dir = Path(output_dir) / split_name / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting {split_name} split...")
    print(f"Images will be saved to: {images_dir}")
    print(f"Labels will be saved to: {labels_dir}")
    
    # Process each sample
    for sample in tqdm(dataset, desc=f"Exporting {split_name}"):
        try:
            # Copy image
            image_path = sample.filepath
            image_name = Path(image_path).name
            output_image_path = images_dir / image_name
            shutil.copy(image_path, output_image_path)
            
            # Get image dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read {image_path}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Convert detections to YOLO format
            label_path = labels_dir / (Path(image_name).stem + '.txt')
            
            with open(label_path, 'w') as f:
                if sample.ground_truth and sample.ground_truth.detections:
                    for detection in sample.ground_truth.detections:
                        # Get bounding box coordinates (normalized)
                        bbox = detection.bounding_box  # [x, y, width, height]
                        
                        # Calculate center coordinates
                        x_center = bbox[0] + bbox[2] / 2
                        y_center = bbox[1] + bbox[3] / 2
                        width = bbox[2]
                        height = bbox[3]
                        
                        # Class ID (0 for Cat - single class)
                        class_id = 0
                        
                        # Write in YOLO format: class_id x_center y_center width height
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        except Exception as e:
            print(f"Error processing sample {sample.filepath}: {str(e)}")
    
    print(f"✓ {split_name} split exported successfully!")


def download_and_export_cat_dataset(output_dir='./data', 
                                    train_samples=1500,
                                    val_samples=300,
                                    test_samples=300,
                                    version='v6'):
    """
    Download cat images from Open Images Dataset and export to YOLO format.
    
    Args:
        output_dir: Output directory for dataset
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        version: Open Images version ('v6' or 'v7')
    """
    dataset_name = f"open-images-{version}"
    
    print("="*50)
    print("Cat Detection Dataset Download & Export")
    print("="*50)
    print(f"Dataset version: {dataset_name}")
    print(f"Training samples requested: {train_samples}")
    print(f"Validation samples requested: {val_samples}")
    print(f"Test samples requested: {test_samples}")
    print(f"Output directory: {output_dir}")
    print("="*50 + "\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download and export training set
    print("\n[1/3] Downloading training set...")
    try:
        train_dataset = foz.load_zoo_dataset(
            dataset_name,
            split="train",
            label_types=["detections"],
            classes=["Cat"],
            max_samples=train_samples,
            dataset_name="cat_detection_train"
        )
        print(f"✓ Found {len(train_dataset)} training samples")
    except Exception as e:
        print(f"Warning: {str(e)}")
        print("Note: Open Images may have fewer samples than requested")
        train_dataset = None
    
    if train_dataset and len(train_dataset) > 0:
        export_fiftyone_to_yolo(train_dataset, output_dir, "train")
    
    # Download and export validation set
    print("\n[2/3] Downloading validation set...")
    try:
        val_dataset = foz.load_zoo_dataset(
            dataset_name,
            split="validation",
            label_types=["detections"],
            classes=["Cat"],
            max_samples=val_samples,
            dataset_name="cat_detection_val"
        )
        print(f"✓ Found {len(val_dataset)} validation samples")
    except Exception as e:
        print(f"Warning: {str(e)}")
        val_dataset = None
    
    if val_dataset and len(val_dataset) > 0:
        export_fiftyone_to_yolo(val_dataset, output_dir, "val")
    
    # Download and export test set
    print("\n[3/3] Downloading test set...")
    try:
        test_dataset = foz.load_zoo_dataset(
            dataset_name,
            split="test",
            label_types=["detections"],
            classes=["Cat"],
            max_samples=test_samples,
            dataset_name="cat_detection_test"
        )
        print(f"✓ Found {len(test_dataset)} test samples")
    except Exception as e:
        print(f"Warning: {str(e)}")
        test_dataset = None
    
    if test_dataset and len(test_dataset) > 0:
        export_fiftyone_to_yolo(test_dataset, output_dir, "test")
    
    print("\n" + "="*50)
    print("✓ Dataset download and export completed!")
    print("="*50)
    print(f"\nDataset structure:")
    print(f"{output_dir}/")
    print(f"├── train/")
    print(f"│   ├── images/")
    print(f"│   └── labels/")
    print(f"├── val/")
    print(f"│   ├── images/")
    print(f"│   └── labels/")
    print(f"└── test/")
    print(f"    ├── images/")
    print(f"    └── labels/")
    print("\n" + "="*50)
    
    # Print statistics
    print("\nDataset Statistics:")
    train_count = len(train_dataset) if train_dataset and 'train_dataset' in locals() else 0
    val_count = len(val_dataset) if val_dataset and 'val_dataset' in locals() else 0
    test_count = len(test_dataset) if test_dataset and 'test_dataset' in locals() else 0
    
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images: {test_count}")
    print(f"Total images: {train_count + val_count + test_count}")
    
    if train_count < train_samples or val_count < val_samples or test_count < test_samples:
        print("\n⚠️  WARNING: Downloaded fewer images than requested!")
        print("This is normal - Open Images has limited samples for some classes.")
        print(f"You can still train with {train_count + val_count + test_count} total images.")
    
    print("="*50)


def main():
    """Main function to download and export dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and export cat dataset from Open Images')
    parser.add_argument('--output', type=str, default='./data',
                       help='Output directory for dataset')
    parser.add_argument('--train', type=int, default=1500,
                       help='Number of training samples')
    parser.add_argument('--val', type=int, default=300,
                       help='Number of validation samples')
    parser.add_argument('--test', type=int, default=300,
                       help='Number of test samples')
    parser.add_argument('--version', type=str, default='v6',
                       choices=['v6', 'v7'],
                       help='Open Images version (v6 or v7)')
    
    args = parser.parse_args()
    
    download_and_export_cat_dataset(
        output_dir=args.output,
        train_samples=args.train,
        val_samples=args.val,
        test_samples=args.test,
        version=args.version
    )

   
if __name__ == '__main__':
    main()
