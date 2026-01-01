import cv2
import numpy as np
import matplotlib.pyplot as plt

def feature_matching_orb(image1_path, image2_path):

    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One of the images could not be read.")

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures = 500)

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    req_matches = [ i for i in matches if i.distance < 40]
    # Draw first 10 matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, req_matches , None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print(len(req_matches))

    # Display the matches
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    path1 = r'FOV1.jpg'
    path2 = r'FOV2.jpg'
    feature_matching_orb(path1, path2)