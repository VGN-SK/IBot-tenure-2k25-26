import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

def detect_edges(image_path, low_threshold=50, high_threshold=150, blur_kernel=5):
    """
    Apply Canny edge detection to an image with preprocessing.
    
    Args:
        image_path: Path to input image
        low_threshold: Lower threshold for Canny
        high_threshold: Upper threshold for Canny
        blur_kernel: Gaussian blur kernel size (must be odd)
    
    Returns:
        original_image_rgb, edges_image
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f'Error: Could not load {image_path}')
        return None, None
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    #blurred = cv2.bilateralFilter(gray,9,75,75)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return image_rgb, edges


def process_image_folder(folder_path, output_folder=None, 
                         low_threshold=50, high_threshold=150):
    """
    Process all images in a folder and display/save edge detection results.
    
    Args:
        folder_path: Path to folder containing images
        output_folder: Optional path to save edge-detected images
        low_threshold: Canny lower threshold
        high_threshold: Canny upper threshold
    """
    # Create output folder if specified
    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        print(f"Created output folder: {output_folder}")
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [
        f for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f'Found {len(image_files)} images')
    
    # Process each image
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, img_file)
        print(f'[{idx}/{len(image_files)}] Processing: {img_file}')
        
        # Detect edges
        original, edges = detect_edges(img_path, low_threshold, high_threshold)
        
        if original is None:
            continue
        
        # Create side-by-side display
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        axes[0].imshow(original)
        axes[0].set_title(f'Original: {img_file}')
        axes[0].axis('off')
        
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title(f'Edge Detection (Canny)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save edge image if output folder specified
        if output_folder:
            output_name = f'edges_{Path(img_file).stem}.png'
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, edges)
            print(f'  Saved: {output_path}')
        
        print()  # Empty line for readability


def batch_compare_thresholds(image_path):
    """
    Compare different Canny thresholds on a single image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Different threshold combinations
    thresholds = [
        (30, 100, 'Sensitive'),
        (50, 150, 'Balanced'),
        (100, 200, 'Strict'),
        (0, 255, 'Auto')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, (low, high, name) in enumerate(thresholds):
        edges = cv2.Canny(blurred, low, high)
        axes[idx].imshow(edges, cmap='gray')
        axes[idx].set_title(f'{name}\n(low={low}, high={high})')
        axes[idx].axis('off')
    
    plt.suptitle(f'Canny Edge Detection - Threshold Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Process folder
    process_image_folder(
        folder_path='wokwi',
        output_folder='edge_results',
        low_threshold=50,
        high_threshold=150
    )