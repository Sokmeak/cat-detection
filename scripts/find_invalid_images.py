"""
Script to find and display invalid images with their bounding box issues.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def read_yolo_label(label_path):
    """Read YOLO format label file."""
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
                        pass
    return boxes


def validate_bbox_detailed(bbox, min_size=0.01, max_size=1.0):
    """
    Validate bounding box and return detailed error message.
    
    Returns:
        (is_valid, error_message)
    """
    class_id, x_center, y_center, width, height = bbox
    
    # Check class ID
    if class_id != 0:
        return False, f"Wrong class ID: {class_id} (should be 0)"
    
    # Check if coordinates are within valid range [0, 1]
    if not (0 <= x_center <= 1):
        return False, f"x_center out of bounds: {x_center} (must be 0-1)"
    
    if not (0 <= y_center <= 1):
        return False, f"y_center out of bounds: {y_center} (must be 0-1)"
    
    # Check if dimensions are within valid range
    if width < min_size:
        return False, f"Width too small: {width} (min: {min_size})"
    
    if height < min_size:
        return False, f"Height too small: {height} (min: {min_size})"
    
    if width > max_size:
        return False, f"Width too large: {width} (max: {max_size})"
    
    if height > max_size:
        return False, f"Height too large: {height} (max: {max_size})"
    
    # Check if box is within image bounds
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    
    if x_min < 0:
        return False, f"Box extends left of image: x_min={x_min}"
    
    if x_max > 1:
        return False, f"Box extends right of image: x_max={x_max}"
    
    if y_min < 0:
        return False, f"Box extends above image: y_min={y_min}"
    
    if y_max > 1:
        return False, f"Box extends below image: y_max={y_max}"
    
    return True, "Valid"


def find_invalid_images(images_dir, labels_dir):
    """Find all images with invalid bounding boxes."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    invalid_images = []
    
    print("Scanning for invalid images...\n")
    
    for img_file in sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png')):
        label_file = labels_dir / (img_file.stem + '.txt')
        
        if not label_file.exists():
            continue
        
        boxes = read_yolo_label(label_file)
        
        for i, bbox in enumerate(boxes):
            is_valid, error_msg = validate_bbox_detailed(bbox)
            
            if not is_valid:
                invalid_images.append({
                    'image': str(img_file),
                    'label': str(label_file),
                    'bbox': bbox,
                    'error': error_msg
                })
                print(f"❌ {img_file.name}")
                print(f"   Label: {label_file.name}")
                print(f"   Box {i}: class={bbox[0]}, x={bbox[1]:.4f}, y={bbox[2]:.4f}, w={bbox[3]:.4f}, h={bbox[4]:.4f}")
                print(f"   Error: {error_msg}\n")
    
    return invalid_images


def visualize_invalid_image(image_path, label_path, output_path):
    """Visualize image with its bounding boxes."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    h, w = img.shape[:2]
    boxes = read_yolo_label(label_path)
    
    for i, bbox in enumerate(boxes):
        class_id, x_center, y_center, width, height = bbox
        
        # Convert normalized coordinates to pixel coordinates
        x_center_px = int(x_center * w)
        y_center_px = int(y_center * h)
        box_w = int(width * w)
        box_h = int(height * h)
        
        x1 = int(x_center_px - box_w / 2)
        y1 = int(y_center_px - box_h / 2)
        x2 = int(x_center_px + box_w / 2)
        y2 = int(y_center_px + box_h / 2)
        
        # Check if box is valid
        is_valid, error_msg = validate_bbox_detailed(bbox)
        
        # Draw box in red if invalid, green if valid
        color = (0, 0, 255) if not is_valid else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"Box {i}: {error_msg[:30]}"
        cv2.putText(img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save image
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Find and display invalid images')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                       help='Directory containing label files')
    parser.add_argument('--output', type=str, default='results/invalid_images',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Find invalid images
    invalid_images = find_invalid_images(args.images, args.labels)
    
    # Print summary
    print("="*60)
    print(f"Found {len(invalid_images)} invalid images")
    print("="*60)
    
    if invalid_images:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize each invalid image
        print("\nGenerating visualizations...\n")
        for item in invalid_images:
            img_name = Path(item['image']).name
            output_path = output_dir / img_name
            visualize_invalid_image(item['image'], item['label'], str(output_path))
        
        print(f"\n✅ All visualizations saved to: {args.output}")
        print(f"\nTo view them:")
        print(f"  open {args.output}")
    else:
        print("✅ No invalid images found!")


if __name__ == '__main__':
    main()
