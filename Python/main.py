import numpy as np

import os, glob

print("This is a test to see if I can add a file to the repo")

#%% Import Photos, rescale to 64x64, and greyscale them

import os
import glob
from PIL import Image
import imageio
#%% load safe and unsafe images and convert to 64x64

#Load safe data
for image_path_safe in glob.glob("UAV_SafeLanding/Dataset/Safe/*.png"):
     image_safe = imageio.imread(image_path_safe)               #loads all safe images
     image_safe = image_safe.resize((64,64))                    #resizes all images to 64x64
     image_safe = image_safe.convert('L')                       #converts all the resized images to grayscale

#Load unsafe data
for image_path_unsafe in glob.glob("UAV_SafeLanding/Dataset/Unsafe/*.png"):
     image_unsafe = imageio.imread(image_path_unsafe)           #loads all unsafe images
     image_unsafe = image_unsafe.resize((64,64))                #resizes all images to 64x64
     image_unsafe = image_unsafe.convert('L')                   #converts all the resized images to grayscale

