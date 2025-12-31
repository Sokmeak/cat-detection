"""
Label Verification Script for Cat Detection Project
Checks bounding box annotations and removes duplicated or incorrectly annotated images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
from collections import defaultdict
import hashlib


def calculate_image_hash(image_path):
    """
    Calculate MD5 hash of an image file for duplicate detection.
    
    Args:
        image_path: Path to image file
        
    Returns:
        MD5 hash string
    """
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def read_yolo_label(label_path):
    """
    Read YOLO format label file.
    
    Args:
        label_path: Path to label file
        
    Returns:
        List of bounding boxes [class_id, x_center, y_center, width, height]
    """
    boxes = []
    
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        box = [float(x) for x in parts]
                        boxes.append(box)
                    except ValueError:
                        print(f"Warning: Invalid box format in {label_path}: {line}")
    
    return boxes


def validate_bbox(bbox, min_size=0.01, max_size=1.0):
    """
    Validate bounding box coordinates.
    
    Args:
        bbox: Bounding box [class_id, x_center, y_center, width, height]
        min_size: Minimum box size (normalized)
        max_size: Maximum box size (normalized)
        
    Returns:
        Boolean indicating if box is valid
    """
    class_id, x_center, y_center, width, height = bbox
    
    # Check if coordinates are within valid range [0, 1]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        return False
    
    # Check if dimensions are within valid range
    if not (min_size <= width <= max_size and min_size <= height <= max_size):
        return False
    
    # Check if box is within image bounds
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    
    if not (0 <= x_min and x_max <= 1 and 0 <= y_min and y_max <= 1):
        return False
    
    # Check class ID (for single class, should be 0)
    if class_id != 0:
        return False
    
    return True


def visualize_annotations(image_path, label_path, output_path=None):
    """
    Visualize bounding boxes on image.
    
    Args:
        image_path: Path to image file
        label_path: Path to label file
        output_path: Optional path to save visualization
        
    Returns:
        Image with bounding boxes drawn
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Read labels
    boxes = read_yolo_label(label_path)
    
    # Draw bounding boxes
    for bbox in boxes:
        class_id, x_center, y_center, width, height = bbox
        
        # Convert normalized coordinates to pixel coordinates
        x_center_px = int(x_center * w)
        y_center_px = int(y_center * h)
        width_px = int(width * w)
        height_px = int(height * h)
        
        # Calculate corner coordinates
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label_text = f"Cat"
        cv2.putText(image, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, image)
    
    return image


def check_dataset(images_dir, labels_dir, output_dir=None, remove_invalid=False):
    """
    Check dataset for issues and optionally remove problematic images.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        output_dir: Optional directory for visualization
        remove_invalid: Whether to remove invalid images/labels
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"\nChecking {len(image_files)} images...")
    
    # Statistics
    stats = {
        'total': len(image_files),
        'valid': 0,
        'no_label': 0,
        'empty_label': 0,
        'invalid_bbox': 0,
        'duplicates': 0,
        'removed': 0
    }
    
    # Track duplicates
    image_hashes = defaultdict(list)
    
    # Lists for invalid files
    files_to_remove = []
    
    # Process each image
    for img_file in tqdm(image_files, desc="Checking labels"):
        # Get corresponding label file
        label_file = labels_path / (img_file.stem + '.txt')
        
        # Check if label exists
        if not label_file.exists():
            stats['no_label'] += 1
            print(f"Warning: No label file for {img_file.name}")
            if remove_invalid:
                files_to_remove.append(img_file)
            continue
        
        # Read labels
        boxes = read_yolo_label(label_file)
        
        # Check if label is empty
        if len(boxes) == 0:
            stats['empty_label'] += 1
            print(f"Warning: Empty label file for {img_file.name}")
            if remove_invalid:
                files_to_remove.append(img_file)
                files_to_remove.append(label_file)
            continue
        
        # Validate each bounding box
        all_valid = True
        for bbox in boxes:
            if not validate_bbox(bbox):
                stats['invalid_bbox'] += 1
                print(f"Warning: Invalid bounding box in {img_file.name}: {bbox}")
                all_valid = False
                if remove_invalid:
                    files_to_remove.append(img_file)
                    files_to_remove.append(label_file)
                break
        
        if not all_valid:
            continue
        
        # Check for duplicates
        img_hash = calculate_image_hash(img_file)
        image_hashes[img_hash].append(img_file)
        
        if len(image_hashes[img_hash]) > 1:
            stats['duplicates'] += 1
            print(f"Warning: Duplicate image found: {img_file.name}")
            if remove_invalid:
                files_to_remove.append(img_file)
                files_to_remove.append(label_file)
            continue
        
        # If all checks pass
        stats['valid'] += 1
        
        # Visualize if output directory provided
        if output_dir and all_valid:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            visualize_annotations(
                str(img_file),
                str(label_file),
                str(output_path / img_file.name)
            )
    
    # Remove invalid files if requested
    if remove_invalid and files_to_remove:
        print(f"\nRemoving {len(files_to_remove)} invalid files...")
        for file_path in tqdm(files_to_remove, desc="Removing files"):
            try:
                os.remove(file_path)
                stats['removed'] += 1
            except Exception as e:
                print(f"Error removing {file_path}: {str(e)}")
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Validation Statistics:")
    print("="*50)
    print(f"Total images: {stats['total']}")
    print(f"Valid images: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"Images without labels: {stats['no_label']}")
    print(f"Images with empty labels: {stats['empty_label']}")
    print(f"Images with invalid bboxes: {stats['invalid_bbox']}")
    print(f"Duplicate images: {stats['duplicates']}")
    if remove_invalid:
        print(f"Files removed: {stats['removed']}")
    print("="*50)
    
    return stats


def main():
    """Main function to check labels."""
    parser = argparse.ArgumentParser(description='Check and validate YOLO labels')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                       help='Directory containing label files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for visualizations')
    parser.add_argument('--remove-invalid', action='store_true',
                       help='Remove invalid images and labels')
    
    args = parser.parse_args()
    
    check_dataset(
        args.images,
        args.labels,
        output_dir=args.output,
        remove_invalid=args.remove_invalid
    )


if __name__ == '__main__':
    main()
