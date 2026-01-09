"""
Image Enhancement using Histogram Equalization and CLAHE
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil


def equalize_histogram(image, clip_limit=3.0, tile_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    if len(image.shape) == 2:
        return clahe.apply(image)
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def copy_labels(src_dir, dst_dir):
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for label_file in src_dir.glob("*.txt"):
        shutil.copy2(label_file, dst_dir / label_file.name)
        count += 1
    return count


def enhance_dataset(dataset_type='train', clip_limit=3.0, tile_size=(8, 8), overwrite=False):
    base_dir = Path(__file__).parent.parent
    src_img = base_dir / "data" / "original" / dataset_type / "images"
    src_lbl = base_dir / "data" / "original" / dataset_type / "labels"
    dst_img = base_dir / "data" / "enhanced" / dataset_type / "images"
    dst_lbl = base_dir / "data" / "enhanced" / dataset_type / "labels"
    
    if not src_img.exists():
        print(f"Error: {src_img} not found")
        return
    
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    
    images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        images.extend(src_img.glob(f"*{ext}"))
    
    if not images:
        print(f"No images found in {src_img}")
        return
    
    print(f"\n{'='*60}")
    print(f"{dataset_type.upper()} Dataset Enhancement with CLAHE")
    print(f"{'='*60}")
    print(f"Images: {len(images)}")
    print(f"Parameters: clip={clip_limit}, tile={tile_size}")
    print(f"{'='*60}\n")
    
    success = skip = error = 0
    
    for img_path in tqdm(images, desc=f"Processing {dataset_type}"):
        dst_path = dst_img / img_path.name
        
        if dst_path.exists() and not overwrite:
            skip += 1
            continue
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Cannot read image")
            
            enhanced = equalize_histogram(img, clip_limit, tile_size)
            cv2.imwrite(str(dst_path), enhanced)
            success += 1
        except Exception as e:
            print(f"\nError: {img_path.name} - {e}")
            error += 1
    
    print("\nCopying labels...")
    labels = copy_labels(src_lbl, dst_lbl)
    
    print(f"\n{'='*60}")
    print(f"Enhanced: {success} | Skipped: {skip} | Errors: {error}")
    print(f"Labels: {labels}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Enhance images using CLAHE (optimal for object detection)',
        epilog='Examples:\n'
               '  python enhance_images.py --dataset train\n'
               '  python enhance_images.py --dataset all\n'
               '  python enhance_images.py --dataset test --clip-limit 2.5',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, default='train',
                       choices=['train', 'test', 'val', 'all'])
    parser.add_argument('--clip-limit', type=float, default=3.0)
    parser.add_argument('--tile-size', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    
    args = parser.parse_args()
    tile_size = (args.tile_size, args.tile_size)
    
    datasets = ['train', 'test', 'val'] if args.dataset == 'all' else [args.dataset]
    
    for dataset in datasets:
        enhance_dataset(dataset, args.clip_limit, tile_size, args.overwrite)


if __name__ == '__main__':
    main()

