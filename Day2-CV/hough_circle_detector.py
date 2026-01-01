import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path,ksize = 5):
    """
    Load and preprocess image for circle detection.
    
    Args:
        image_path: Path to input image
    
    Returns:
        tuple: (original_color, preprocessed_gray) or (None, None)
    """

    image = cv2.imread(image_path)
    if image is None :
        return None,None
    
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    median_blurred = cv2.medianBlur(gray , ksize)
    gaussian_blurred = cv2.GaussianBlur(gray,(ksize,ksize),0)

    return (image,gaussian_blurred)


def detect_circles(gray_image, dp=1, minDist=50, param1=50, 
                   param2=30, minRadius=10, maxRadius=100):
    """
    Detect circles using Hough Circle Transform.
    
    Args:
        gray_image: Preprocessed grayscale image
        dp: Inverse accumulator resolution ratio
        minDist: Minimum distance between circle centers
        param1: Upper Canny threshold
        param2: Accumulator threshold
        minRadius: Minimum circle radius
        maxRadius: Maximum circle radius
    
    Returns:
        numpy array of circles (x, y, radius) or None
    """
    # TODO: Apply HoughCircles
    circles = cv2.HoughCircles(gray_image,cv2.HOUGH_GRADIENT,dp,minDist=minDist,param1=param1,param2 = param2, minRadius = minRadius, maxRadius = maxRadius)
    circles = np.uint16(np.around(circles))

    return circles


def visualize_circles(image, circles, save_path=None):
    """
    Draw detected circles on image and display.
    
    Args:
        image: Original color image
        circles: Array of detected circles
        save_path: Optional path to save annotated image
    """
    # TODO: Draw circles and labels
    for cx,cy,r in circles[0] :
        cv2.circle(image,(cx,cy),r,(0,0,0),2)
        cv2.putText(image,'circle detected',(cx-r,cy-r),fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale= 0.6,color=(0,0,0),thickness=2)

    return image


def calculate_statistics(circles):
    """
    Calculate and display statistics about detected circles.
    
    Args:
        circles: Array of detected circles
    
    Returns:
        dict: Statistics dictionary
    """
    # TODO: Compute statistics
    pass


def main():
    """Main function."""
    # TODO: Implement main workflow
    script_path = os.path.dirname(os.path.abspath(__file__))
    path = input('Enter relative path of image  ')
    img_path = os.path.join(script_path,path)
    image, gray =preprocess_image(img_path,ksize = 9)
    circles = detect_circles(gray,param2 = 30)

    image_copy = image.copy()
    annotated = visualize_circles(image_copy,circles)
    plt.imshow(annotated)
    plt.title('Circle detection')
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    main()