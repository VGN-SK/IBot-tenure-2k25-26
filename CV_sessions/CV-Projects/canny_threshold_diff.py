import numpy as np
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('dog.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur first (reduces noise)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
# threshold1 = lower threshold for weak edges
# threshold2 = upper threshold for strong edges
# Edge pixels between thresholds are included only if connected to strong edges

# Try different thresholds
edges_sensitive = cv2.Canny(blurred, 30, 100)   # More edges
edges_strict = cv2.Canny(blurred, 100, 200)     # Fewer edges
edges_auto = cv2.Canny(blurred, 0, 255)         # Auto thresholds

# Automatic threshold calculation (Otsu's method inspiration)
def auto_canny(image, sigma=0.33):
    """Automatically determine Canny thresholds"""
    median_val = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    return cv2.Canny(image, lower, upper)

edges_auto_calc = auto_canny(blurred)

# Display results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original Grayscale')

axes[1].imshow(blurred, cmap='gray')
axes[1].set_title('Blurred')

axes[2].imshow(edges, cmap='gray')
axes[2].set_title('Canny (50, 150)')

axes[3].imshow(edges_sensitive, cmap='gray')
axes[3].set_title('Sensitive (30, 100)')

axes[4].imshow(edges_strict, cmap='gray')
axes[4].set_title('Strict (100, 200)')

axes[5].imshow(edges_auto_calc, cmap='gray')
axes[5].set_title('Auto Threshold')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()