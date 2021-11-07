#Import Libraries
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



#%% Model
def model (x_p,w):
    x1 = x_p[:,0]
    x2 = x_p[:,1]
    x3 = x_p[:,2]
    x4 = x_p[:,3]
    x5 = x_p[:,4]
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    x3 = x3.reshape(-1,1)
    x4 = x4.reshape(-1,1)
    x5 = x5.reshape(-1,1)
    a = w[0] + np.dot(x1,w[1])+ np.dot(x2,w[2])+ np.dot(x3,w[3])+ np.dot(x4,w[4])+ np.dot(x5,w[5])
    return a

#%% Sigmoid Function
def sigmoid(t):
    return 1/(1+np.exp(-t))

#%% Cross Entropy Cost Function
def cross_entropy(w,x,y):
    a = sigmoid(model(x,w))
    #print('sigmoid result is ', a)
    #print(np.shape(a))
    ind = np.argwhere(y==0)[:,0]
    cost = -np.sum(np.log(1-a[ind]))
    #print(cost)
    ind = np.argwhere(y==1)[:,0]
    cost -= np.sum(np.log(a[ind]))
   # print(cost)
    return cost/y.size

#%% Gradient Descent
def gradient_descent(g, step, max_its, w):
    
    gradient = grad(g) # compute gradient of cost function

# gradient descent loop
    weight_history = [w] # weight history container
    cost_history = [g(w)] # cost history container
    for k in range(max_its):
        # eval gradient
        grad_eval = gradient(w)
        grad_eval_norm = grad_eval /np.linalg.norm(grad_eval)
    
    # take gradient descent step
        if step == 'd': # diminishing step
            alpha = 1/(k+1)
        else: # constant step
            alpha = step
        w = w - alpha*grad_eval_norm
    
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history, cost_history

#%% Confusion Matrix
prediction = sigmoid(model(x,weights))
actual = y
a = 0
b = 0
c = 0
d = 0
for i in range(20):
    if actual[i] == 1 :
        if (actual[i] - prediction[i]) < 0.5: # this increments if  correctly predicting a 1 (actual = 1, predict = 1)
            a = a + 1
        if (actual[i] - prediction[i]) > 0.5: # this increments if incorrectly predicting a 1 as a 0 (actual = 1, predict = 0)
            b = b + 1
    if actual[i] == 0:
        if (actual[i] - prediction[i]) > -0.5: # this increments if correctly predicting a 0 (actual = 0, predict = 0)
            d = d + 1
        if (actual[i] - prediction[i]) < -0.5: # this increments if incorrectly predicting a 0 (actual = 0, predict = 1)
            c = c + 1
e = np.zeros((2,2))
e[0][0] = a
e[0][1] = b
e[1][0] = c
e[1][1] = d
confusion_matrix = e
print(confusion_matrix)
