#Import Libraries

import glob
import numpy as np
from PIL import Image
#%% Load images and convert them to 64x64 and grayscale

#Load safe data
for image_path_safe in glob.glob("C:/Users/alexm/UAV_SafeLanding/Dataset/Safe/*.png"): # Note must use local path for image loading
     image_safe = Image.open(image_path_safe)                   #loads all safe images
     image_safe = image_safe.resize((64,64))                    #resizes all images to 64x64
     image_safe = image_safe.convert('L')                       #converts all the resized images to grayscale
     #image_safe.save(image_path_safe, .png)                    #saves the images as a .png
#Load unsafe data
for image_path_unsafe in glob.glob("C:/Users/alexm/UAV_SafeLanding/Dataset/Safe/*.png"): #Note: Must use local path for image loading!
     image_unsafe = Image.open(image_path_unsafe)               #loads all unsafe images
     image_unsafe = image_unsafe.resize((64,64))                #resizes all images to 64x64
     image_unsafe = image_unsafe.convert('L')                   #converts all the resized images to grayscale
     #image_unsafe.save(''.png)                                 #saves the images as a .png

#%% Cross Entropy Cost Function


