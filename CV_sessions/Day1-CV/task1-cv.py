import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'dog.jpg'
image  = cv2.imread(path)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

gaussian_blurred = cv2.GaussianBlur(image,(7,7),0)

def canny(image_file,low,high) :
    gray = cv2.cvtColor(image_file,cv2.COLOR_RGB2GRAY)
    #gray_blurred = cv2.GaussianBlur(gray,(7,7),0)
    gray_blurred = cv2.bilateralFilter(gray,9,75,75)
    edges = cv2.Canny(gray_blurred,low,high)

    return edges

def binary_thresh(image_file,threshold) :
    gray = cv2.cvtColor(image_file,cv2.COLOR_RGB2GRAY)
    gray_blurred = cv2.GaussianBlur(gray,(5,5),0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.threshold(gray_blurred, threshold, 255, cv2.THRESH_BINARY)[1]
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return clean

def otsu_thresh(image_file) :
    gray = cv2.cvtColor(image_file,cv2.COLOR_RGB2GRAY)
    gray_blurred = cv2.GaussianBlur(gray,(5,5),0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return clean

canny_image = canny(image,50,150)

T = np.clip(np.median(gaussian_blurred), 50, 200)
binary_image = binary_thresh(image,T)
#binary_image = otsu_thresh(image)

images = [image,gaussian_blurred,canny_image,binary_image]
titles = ['Normal','Gaussian_blur','Canny_edge','Binary_thresh']
ids = [0,1,2,3]

fig,axes = plt.subplots(2,2,figsize = (14,14))
axes = axes.ravel()
for a,b,c in zip(images,titles,ids) :
    if c == 2 or c== 3 :
        axes[c].imshow(a,cmap = 'gray')
    else :
        axes[c].imshow(a)

    axes[c].set_title(b)
    axes[c].axis('off')

plt.tight_layout()
plt.show()