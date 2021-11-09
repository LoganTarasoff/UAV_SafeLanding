#Import Libraries
#
# Install OpenCV: pip install opencv-python
#
import cv2 as cv
import argparse
import glob
import numpy as np
from PIL import Image

#Constants
safe_img_path = "../Dataset/Safe/"
unsafe_img_path = "../Dataset/Unsafe/"

#%% Load images and convert them to 64x64 and grayscale
def resize_images(option):

    if option == 1 or option == 3:
        #Load safe data
        for image_path_safe in glob.glob(unsafe_img_path + "*.png"): # Note must use local path for image loading
             image_safe = Image.open(image_path_safe)                   #loads all safe images
             image_safe = image_safe.resize((64,64))                    #resizes all images to 64x64
             image_safe = image_safe.convert('L')                       #converts all the resized images to grayscale
             #image_safe.save(image_path_safe, .png)                    #saves the images as a .png
    elif option == 2 or option == 3:
        #Load unsafe data
        for image_path_unsafe in glob.glob(safe_img_path + "*.png"): #Note: Must use local path for image loading!
             image_unsafe = Image.open(image_path_unsafe)               #loads all unsafe images
             image_unsafe = image_unsafe.resize((64,64))                #resizes all images to 64x64
             image_unsafe = image_unsafe.convert('L')                   #converts all the resized images to grayscale
             #image_unsafe.save(''.png)                                 #saves the images as a .png

#Parse arguments
#
# Usage: python .\main.py --resize all
#
parser = argparse.ArgumentParser(description='UAV Safe Landing Project Script')
parser.add_argument("--resize", type=str, default='all', help='Resize dataset to 64x64. Default value = \'all\'. Options = \'safe\', \'unsafe\', \'all\'' )
args = parser.parse_args()

#Resize images if argument passed
resize = args.resize
if resize == "all":
    resize_images(3)
elif resize == "safe":
    resize_images(1)
elif resize == "unsafe":
    resize_images(2)
else:
    print("Invalid resize argument!")

#Edge Detection Using OpenCV
img = cv.imread(unsafe_img_path + "img_17.png")
cv.imshow('Original', img)
cv.waitKey(0)

img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv.GaussianBlur(img_gray, (3,3), 0)

# Canny Edge Detection
edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

# Display Canny Edge Detection Image
cv.imshow('Canny Edge Detection', edges)
cv.waitKey(0)
cv.destroyAllWindows()


#%% Cross Entropy Cost Function
