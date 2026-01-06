"""
Separate flagged images into a correction folder
"""
import os
import shutil
from pathlib import Path

def separate_flagged_images(flagged_file, images_dir, labels_dir, output_dir):
    """
    Copy flagged images and labels to a separate folder for correction.
    
    Args:
        flagged_file: Path to flagged_images.txt
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        output_dir: Directory to copy flagged items for correction
    """
    flagged_file = Path(flagged_file)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_images = output_dir / 'images'
    output_labels = output_dir / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    if not flagged_file.exists():
        print(f"Error: {flagged_file} not found!")
        return
    
    # Read flagged images
    with open(flagged_file, 'r') as f:
        flagged_images = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(flagged_images)} flagged images to separate")
    
    copied_images = 0
    copied_labels = 0
    missing_images = []
    missing_labels = []
    
    for img_name in flagged_images:
        # Image path
        img_path = images_dir / img_name
        
        # Label path (replace .jpg with .txt)
        label_name = img_name.replace('.jpg', '.txt').replace('.JPG', '.txt')
        label_path = labels_dir / label_name
        
        # Copy image
        if img_path.exists():
            shutil.copy2(img_path, output_images / img_name)
            copied_images += 1
        else:
            missing_images.append(img_name)
        
        # Copy label
        if label_path.exists():
            shutil.copy2(label_path, output_labels / label_name)
            copied_labels += 1
        else:
            missing_labels.append(label_name)
    
    print("\n" + "="*50)
    print("Separation Summary:")
    print("="*50)
    print(f"Images copied: {copied_images}")
    print(f"Labels copied: {copied_labels}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"\nOutput directory: {output_dir}")
    print("="*50)
    
    if missing_images:
        print(f"\n⚠️  Warning: {len(missing_images)} images not found")
    if missing_labels:
        print(f"⚠️  Warning: {len(missing_labels)} labels not found")
    
    print(f"\n✅ Flagged items copied to {output_dir}")
    print(f"   Use the web viewer to correct labels in this folder")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Separate flagged images for correction')
    parser.add_argument('--flagged-file', type=str, 
                       default='../flagged_images.txt',
                       help='Path to flagged_images.txt')
    parser.add_argument('--images', type=str,
                       default='../data/original/train/images',
                       help='Directory containing images')
    parser.add_argument('--labels', type=str,
                       default='../data/original/train/labels',
                       help='Directory containing labels')
    parser.add_argument('--output', type=str,
                       default='../data/needs_correction',
                       help='Output directory for flagged items')
    
    args = parser.parse_args()
    
    separate_flagged_images(
        args.flagged_file,
        args.images,
        args.labels,
        args.output
    )
