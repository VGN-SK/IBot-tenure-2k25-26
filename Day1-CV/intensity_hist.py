import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "dog.jpg")
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = gray.ravel()

    fig,axes = plt.subplots(1,2, figsize=(8,4))
    axes = axes.ravel()

    no_of_bins = 256
    axes[0].hist(data , bins=no_of_bins, range=(0, no_of_bins - 1), color='black')
    axes[0].set_title('Grayscale Histogram')
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Number of Pixels')
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()