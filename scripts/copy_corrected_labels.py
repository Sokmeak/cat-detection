"""
Copy corrected labels back to the original dataset
"""
import os
import shutil
from pathlib import Path

def copy_corrected_labels(corrected_labels_dir, target_labels_dir, backup=True):
    """
    Copy corrected labels back to the original dataset.
    
    Args:
        corrected_labels_dir: Directory with corrected labels
        target_labels_dir: Original labels directory
        backup: Whether to backup original labels
    """
    corrected_labels_dir = Path(corrected_labels_dir)
    target_labels_dir = Path(target_labels_dir)
    
    if not corrected_labels_dir.exists():
        print(f"Error: {corrected_labels_dir} not found!")
        return
    
    # Create backup if requested
    if backup:
        backup_dir = target_labels_dir.parent / 'labels_backup_before_correction'
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating backup at: {backup_dir}")
    
    corrected_count = 0
    label_files = list(corrected_labels_dir.glob('*.txt'))
    
    print(f"Found {len(label_files)} corrected labels")
    
    for label_file in label_files:
        target_file = target_labels_dir / label_file.name
        
        if target_file.exists() and backup:
            # Backup original
            shutil.copy2(target_file, backup_dir / label_file.name)
        
        # Copy corrected label
        shutil.copy2(label_file, target_file)
        corrected_count += 1
    
    print("\n" + "="*50)
    print("Copy Summary:")
    print("="*50)
    print(f"Labels copied: {corrected_count}")
    if backup:
        print(f"Backup location: {backup_dir}")
    print("="*50)
    print("\n✅ Corrected labels copied successfully!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Copy corrected labels back')
    parser.add_argument('--corrected', type=str,
                       default='../data/needs_correction/labels',
                       help='Directory with corrected labels')
    parser.add_argument('--target', type=str,
                       default='../data/original/train/labels',
                       help='Original labels directory')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backup of original labels')
    
    args = parser.parse_args()
    
    copy_corrected_labels(
        args.corrected,
        args.target,
        backup=not args.no_backup
    )
