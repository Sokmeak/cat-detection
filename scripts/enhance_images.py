"""
Image Enhancement Script for Cat Detection Project
Applies brightness correction, contrast adjustment, and sharpening to improve image quality.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def analyze_image_quality(image):
    """
    Analyze image brightness and contrast.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        dict: Dictionary containing brightness and contrast metrics
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness (mean pixel value)
    brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'is_dark': brightness < 80,
        'is_bright': brightness > 180,
        'is_low_contrast': contrast < 40
    }


def adjust_brightness(image, target_brightness=128):
    """
    Adjust image brightness to target value.
    
    Args:
        image: Input image (BGR format)
        target_brightness: Target mean brightness (0-255)
        
    Returns:
        Brightness-adjusted image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    
    # Calculate adjustment factor
    delta = target_brightness - current_brightness
    
    # Apply brightness adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=1, beta=delta)
    
    return adjusted


def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image (BGR format)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    # Merge channels
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def sharpen_image(image, kernel_size=5, sigma=1.0, amount=1.0):
    """
    Sharpen image using unsharp masking.
    
    Args:
        image: Input image (BGR format)
        kernel_size: Size of Gaussian kernel for blurring
        sigma: Standard deviation for Gaussian kernel
        amount: Sharpening strength (0.0 to 2.0)
        
    Returns:
        Sharpened image
    """
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Calculate unsharp mask
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return sharpened


def denoise_image(image, h=10, template_window_size=7, search_window_size=21):
    """
    Reduce noise in image using Non-local Means Denoising.
    
    Args:
        image: Input image (BGR format)
        h: Filter strength
        template_window_size: Size of template patch
        search_window_size: Size of search area
        
    Returns:
        Denoised image
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window_size, search_window_size
    )
    
    return denoised


def enhance_image(image, apply_brightness=True, apply_contrast=True, 
                 apply_sharpening=True, apply_denoising=False):
    """
    Apply complete image enhancement pipeline.
    
    Args:
        image: Input image (BGR format)
        apply_brightness: Whether to adjust brightness
        apply_contrast: Whether to enhance contrast
        apply_sharpening: Whether to sharpen image
        apply_denoising: Whether to apply denoising
        
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    # Analyze image quality
    quality = analyze_image_quality(enhanced)
    
    # Apply brightness correction if needed
    if apply_brightness and (quality['is_dark'] or quality['is_bright']):
        enhanced = adjust_brightness(enhanced)
    
    # Apply contrast enhancement
    if apply_contrast and quality['is_low_contrast']:
        enhanced = enhance_contrast(enhanced)
    
    # Apply denoising (optional)
    if apply_denoising:
        enhanced = denoise_image(enhanced)
    
    # Apply sharpening
    if apply_sharpening:
        enhanced = sharpen_image(enhanced)
    
    return enhanced


def process_dataset(input_dir, output_dir, apply_brightness=True, 
                   apply_contrast=True, apply_sharpening=True, apply_denoising=False):
    """
    Process entire dataset and save enhanced images.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save enhanced images
        apply_brightness: Whether to adjust brightness
        apply_contrast: Whether to enhance contrast
        apply_sharpening: Whether to sharpen images
        apply_denoising: Whether to apply denoising
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Processing {len(image_files)} images...")
    
    # Track statistics
    stats = {
        'total': len(image_files),
        'dark_images': 0,
        'bright_images': 0,
        'low_contrast': 0
    }
    
    # Process each image
    for img_file in tqdm(image_files, desc="Enhancing images"):
        try:
            # Read image
            image = cv2.imread(str(img_file))
            
            if image is None:
                print(f"Warning: Could not read {img_file.name}")
                continue
            
            # Analyze quality
            quality = analyze_image_quality(image)
            if quality['is_dark']:
                stats['dark_images'] += 1
            if quality['is_bright']:
                stats['bright_images'] += 1
            if quality['is_low_contrast']:
                stats['low_contrast'] += 1
            
            # Enhance image
            enhanced = enhance_image(
                image, 
                apply_brightness=apply_brightness,
                apply_contrast=apply_contrast,
                apply_sharpening=apply_sharpening,
                apply_denoising=apply_denoising
            )
            
            # Save enhanced image
            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), enhanced)
            
        except Exception as e:
            print(f"Error processing {img_file.name}: {str(e)}")
    
    # Print statistics
    print("\n" + "="*50)
    print("Image Quality Statistics:")
    print("="*50)
    print(f"Total images processed: {stats['total']}")
    print(f"Dark images: {stats['dark_images']} ({stats['dark_images']/stats['total']*100:.1f}%)")
    print(f"Bright images: {stats['bright_images']} ({stats['bright_images']/stats['total']*100:.1f}%)")
    print(f"Low contrast images: {stats['low_contrast']} ({stats['low_contrast']/stats['total']*100:.1f}%)")
    print("="*50)


def main():
    """Main function to run image enhancement."""
    parser = argparse.ArgumentParser(description='Enhance images for cat detection')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for enhanced images')
    parser.add_argument('--brightness', action='store_true', default=True,
                       help='Apply brightness correction')
    parser.add_argument('--contrast', action='store_true', default=True,
                       help='Apply contrast enhancement')
    parser.add_argument('--sharpen', action='store_true', default=True,
                       help='Apply image sharpening')
    parser.add_argument('--denoise', action='store_true', default=False,
                       help='Apply denoising (slower)')
    
    args = parser.parse_args()
    
    process_dataset(
        args.input,
        args.output,
        apply_brightness=args.brightness,
        apply_contrast=args.contrast,
        apply_sharpening=args.sharpen,
        apply_denoising=args.denoise
    )


if __name__ == '__main__':
    main()
