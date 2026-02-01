import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2



def pencil_sketch(image, saturation_factor = 0.8, blur_kernel=99 ,noise_std = 5):
  
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray

    blur_inv_gray = cv2.GaussianBlur(inv_gray,(blur_kernel,blur_kernel),0)
    inv_blur_inv_gray = 255 - blur_inv_gray

    dodge = (gray/(inv_blur_inv_gray + 1e-6))*256
    sketch = np.clip(dodge,0,255).astype(np.uint8)

    noise = np.random.normal(0, noise_std, sketch.shape).astype(np.int16)
    sketch = np.clip(sketch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return sketch


def color_sketch(image, saturation_factor = 0.8, blur_kernel = 99, noise_std = 5) :

    image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hue,sat,val = cv2.split(image_hsv)

    inv_gray = 255 - val
    blur_inv_gray = cv2.GaussianBlur(inv_gray,(blur_kernel,blur_kernel),0)
    inv_blur_inv_gray = 255 - blur_inv_gray

    dodge = (val/(inv_blur_inv_gray + 1e-6))*256
    sketch_val = np.clip(dodge,0,255).astype(np.uint8)

    noise = np.random.normal(0,noise_std,sketch_val.shape)
    sketch_val = np.clip(sketch_val + noise,0,255).astype(np.uint8)

    final_sat = np.clip(sat.astype(np.float32) * saturation_factor,0, 255).astype(np.uint8)

    final_image_hsv = cv2.merge([hue,final_sat.astype(np.uint8),sketch_val])
    final_image = cv2.cvtColor(final_image_hsv,cv2.COLOR_HSV2BGR)

    return final_image


def display_results(original, pencil_sketch ,color_sketch):

    images = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB), pencil_sketch, cv2.cvtColor(color_sketch, cv2.COLOR_BGR2RGB)]
    titles = ["Original Image", "Pencil Sketch", "Color Pencil Sketch"]

    fig,axes = plt.subplots(1,3,figsize=(15,5))
    axes = axes.ravel()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i].set_title(title)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
   

def main_image() :

    image_path = input("Enter image path: ").strip()
    blur_kernel = int(input("Blur kernel (odd, e.g. 99): "))
    saturation_factor = float(input("Saturation factor (0.4–0.7): "))
    noise_std = int(input("Noise - standard deviation: "))

    image = cv2.imread(image_path)
    gray_sketch = pencil_sketch(image, blur_kernel ,noise_std)
    color = color_sketch(image, saturation_factor, blur_kernel ,noise_std)

    display_results(image ,gray_sketch, color)


def main_video_live() :
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pencil_sketch_img = pencil_sketch(frame)
        color_sketch_img = color_sketch(frame)

        cv2.imshow("Pencil Sketch", pencil_sketch_img)
        cv2.imshow("Color Sketch", color_sketch_img)

        if cv2.waitKey(1) == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def process_folder(folder_path, output_path = None, effect = 'pencil', saturation_factor = 0.8, blur_kernel = 99, noise_std = 5) :
    if output_path :
        Path ( output_path ). mkdir ( parents =True , exist_ok = True )
        print (f" Created output folder : { output_path }")
    image_extensions = ['.jpg', '. jpeg', '.png']
    image_files = [f for f in os. listdir ( folder_path ) if Path (f).suffix.lower() in image_extensions]
    if not image_files :
        print (f"No images found in { folder_path }")
        return
    print (f'Found {len( image_files )} images')
    
    for i,image in enumerate(image_files) :
        function = pencil_sketch if effect == 'pencil' else color_sketch
        img_path = os. path . join ( folder_path , image )
        img = cv2.imread(img_path)
        sketch = function(img, saturation_factor, blur_kernel, noise_std)
        img_output_path = output_path + f'/sketch_{i+1}{Path (img_path).suffix.lower()}'
        cv2.imwrite(img_output_path,sketch)


def batch_img() :

    effect = int(input('Enter 0 for pencil sketch and 1 for color sketch'))
    blur_kernel = int(input("Blur kernel (odd, e.g. 99): "))
    saturation_factor = float(input("Saturation factor (0.4–0.7): "))
    noise_std = int(input("Noise - standard deviation: "))

    if effect in (0,1) :
        process_folder(r'./test_images',r'./output_sketches','pencil' if effect == 0 else 'color',saturation_factor, blur_kernel, noise_std)
    else :
        print('Enter valid effect')


if __name__ == "__main__":
    choice = int(input('Enter 0 for batch img processing, 1 for 1 for individual image and 2 for live'))
    if choice == 0 :
        batch_img()
    elif choice == 1:
        main_image()
    elif choice == 2:
        main_video_live()