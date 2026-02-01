import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def augment_image(image, count, iterations=5, fig_size=3, output_path=None):

    fig, axes = plt.subplots(iterations, 8, figsize=(8* fig_size, fig_size * iterations))
    axes = axes.ravel()

    for i in range(iterations):

        base = i * 8

        # Flip
        choice = np.random.choice([-1, 0, 1])
        flip = cv2.flip(image, choice)
        axes[base+0].imshow(cv2.cvtColor(flip, cv2.COLOR_BGR2RGB))
        axes[base+0].set_title("flip")

        # Rotation
        angle = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.8, 1.2)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        rotate = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        axes[base+1].imshow(cv2.cvtColor(rotate, cv2.COLOR_BGR2RGB))
        axes[base+1].set_title("rotate")

        # Brightness (scale)
        factor = np.random.uniform(0.7, 1.3)
        bright1 = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        axes[base+2].imshow(cv2.cvtColor(bright1, cv2.COLOR_BGR2RGB))
        axes[base+2].set_title("brightness scale")

        # Brightness (add)
        delta = np.random.randint(-40, 40)
        bright2 = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        axes[base+3].imshow(cv2.cvtColor(bright2, cv2.COLOR_BGR2RGB))
        axes[base+3].set_title("brightness add")

        # Contrast
        contrast = np.random.uniform(0.7, 1.3)
        cont = np.clip(128 + contrast * (image.astype(np.float32) - 128), 0, 255).astype(np.uint8)
        axes[base+4].imshow(cv2.cvtColor(cont, cv2.COLOR_BGR2RGB))
        axes[base+4].set_title("contrast")

        # Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= np.random.uniform(0.6, 1.4)
        sat = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        axes[base+5].imshow(cv2.cvtColor(sat, cv2.COLOR_BGR2RGB))
        axes[base+5].set_title("saturation")

        # Hue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-30, 30)) % 180
        hue = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        axes[base+6].imshow(cv2.cvtColor(hue, cv2.COLOR_BGR2RGB))
        axes[base+6].set_title("hue")

        # Noise
        std = np.random.randint(5, 20)
        noise = np.random.normal(0, std, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        axes[base+7].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
        axes[base+7].set_title("noise")


    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    img_path = os.path.join(output_path, f"augmented_image_{count+1}.png") if output_path else "augmented_image.png"
    plt.savefig(img_path)
    


def main(output_folder = 'augmented_images'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "test_images_aug")

    count = 0
    if not os.path.exists(input_path):
        print(f"Input folder does not exist: {input_path}")
        return
    
    if output_folder:
        output_path = os.path.join(script_dir, output_folder)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        print(f"Created output folder: {output_path}")

    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        print(f"Augmenting image: {img_name}")
        augment_image(image, count, iterations=5, fig_size=3, output_path=output_path)
        count += 1

if __name__ == "__main__":
    
    main()