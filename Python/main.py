
import cv2 as cv
import argparse
import os
import glob
from autograd import grad
import autograd.numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Constants
safe_img_path = "../Dataset/Cropped_Safe/"
unsafe_img_path = "../Dataset/Cropped_Unsafe/"
resized_safe_img_path = "../Dataset/Resized_Safe/"
resized_unsafe_img_path = "../Dataset/Resized_Unsafe/"
edgemap_safe_img_path = "../Dataset/Edgemap_Safe/"
edgemap_unsafe_img_path = "../Dataset/Edgemap_Unsafe/"

test_safe_img_path = "../Dataset/Safe/" #"Dataset/Test_Cropped_Safe/"
test_unsafe_img_path = "../Dataset/Unsafe/"
test_resized_safe_img_path = "../Dataset/Test_Resized_Safe/"
test_resized_unsafe_img_path = "../Dataset/Test_Resized_Unsafe/"
test_edgemap_safe_img_path = "../Dataset/Test_Edgemap_Safe/"
test_edgemap_unsafe_img_path = "../Dataset/Test_Edgemap_Unsafe/"

x = np.empty((0,2), int)
y = np.empty((0,1), int)
x_test = np.empty((0,2), int)
y_test = np.empty((0,1), int)

#%% Load images and convert them to 64x64 and grayscale
def resize_images(option):

    if option == 1 or option == 3:
        #Load safe data
        index = 1
        if not os.path.isdir(resized_safe_img_path):
            os.mkdir(resized_safe_img_path)
            print("Making " + resized_safe_img_path)
        for image_path_safe in glob.glob(safe_img_path + "*.png"): # Note must use local path for image loading
            image_safe = Image.open(image_path_safe)                   #loads all safe images
            image_safe = image_safe.resize((64,64))                    #resizes all images to 64x64
            image_safe = image_safe.convert('L')                       #converts all the resized images to grayscale
            resized_img_name = resized_safe_img_path + "img_" + str(index)
            image_safe.save(resized_img_name+ ".png", "PNG")                    #saves the images as a .png
            index = index + 1
        print("Resize safe images: DONE")

    if option == 2 or option == 3:
        #Load unsafe data
        index = 1
        if not os.path.isdir(resized_unsafe_img_path):
            os.mkdir(resized_unsafe_img_path)
            print("Making " + resized_unsafe_img_path)
        for image_path_unsafe in glob.glob(unsafe_img_path + "*.png"):  #Note: Must use local path for image loading!
            image_unsafe = Image.open(image_path_unsafe)
            image_unsafe = image_unsafe.resize((64,64))
            image_unsafe = image_unsafe.convert('L')
            resized_img_name = resized_unsafe_img_path + "img_" + str(index)
            image_unsafe.save(resized_img_name + ".png", "PNG")
            index = index + 1
        print("Resize unsafe images: DONE")

    if option == 4 or option == 3:
        #Load safe test data
        index = 1
        if not os.path.isdir(test_resized_safe_img_path):
            os.mkdir(test_resized_safe_img_path)
            print("Making " + test_resized_safe_img_path)
        for test_image_path_safe in glob.glob(test_safe_img_path + "*.png"):  #Note: Must use local path for image loading!
            test_image_safe = Image.open(test_image_path_safe)
            test_image_safe = test_image_safe.resize((64,64))
            test_image_safe = test_image_safe.convert('L')
            test_resized_img_name = test_resized_safe_img_path + "img_" + str(index)
            test_image_safe.save(test_resized_img_name + ".png", "PNG")
            index = index + 1
        print("Resize safe test images: DONE")

    if option == 4 or option == 3:
        #Load unsafe test data
        index = 1
        if not os.path.isdir(test_resized_unsafe_img_path):
            os.mkdir(test_resized_unsafe_img_path)
            print("Making " + test_resized_unsafe_img_path)
        for test_image_path_unsafe in glob.glob(test_unsafe_img_path + "*.png"): #Note: Must use local path for image loading!
            test_image_unsafe = Image.open(test_image_path_unsafe)
            test_image_unsafe = test_image_unsafe.resize((64,64))
            test_image_unsafe = test_image_unsafe.convert('L')
            test_resized_img_name = test_resized_unsafe_img_path + "img_" + str(index)
            test_image_unsafe.save(test_resized_img_name + ".png", "PNG")
            index = index + 1
        print("Resize unsafe test images: DONE")

#Load resized images and generate edge maps for each using OpenCV's Canny detector
def generate_edge_maps():

    #Load safe data
    index = 1
    if not os.path.isdir(edgemap_safe_img_path):
        os.mkdir(edgemap_safe_img_path)
        print("Making " + edgemap_safe_img_path)
    for image_path_safe in glob.glob(resized_safe_img_path + "*.png"): # Note must use local path for image loading
        img = cv.imread(image_path_safe)
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
        edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
        edgemap_img_name = edgemap_safe_img_path + "img_" + str(index) + ".png"
        cv.imwrite(edgemap_img_name, edges)
        index = index + 1

    print("Safe image edgemap generation: DONE")

    #Load unsafe data
    index = 1
    if not os.path.isdir(edgemap_unsafe_img_path):
        os.mkdir(edgemap_unsafe_img_path)
        print("Making " + edgemap_unsafe_img_path)
    for image_path_unsafe in glob.glob(resized_unsafe_img_path + "*.png"): #Note: Must use local path for image loading!
        img = cv.imread(image_path_unsafe)
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
        edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
        edgemap_img_name = edgemap_unsafe_img_path + "img_" + str(index) + ".png"
        cv.imwrite(edgemap_img_name, edges)
        index = index + 1

    print("Unsafe image edgemap generation: DONE")

    #Load safe test data
    index = 1
    if not os.path.isdir(test_edgemap_safe_img_path):
        os.mkdir(test_edgemap_safe_img_path)
        print("Making " + test_edgemap_safe_img_path)
    for test_image_path_safe in glob.glob(test_resized_safe_img_path + "*.png"): # Note must use local path for image loading
        img = cv.imread(test_image_path_safe)
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
        edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
        edgemap_img_name = test_edgemap_safe_img_path + "img_" + str(index) + ".png"
        cv.imwrite(edgemap_img_name, edges)
        index = index + 1
    print("Safe test image edgemap generation: DONE")

    #Load unsafe test data
    index = 1
    if not os.path.isdir(test_edgemap_unsafe_img_path):
        os.mkdir(test_edgemap_unsafe_img_path)
        print("Making " + test_edgemap_unsafe_img_path)
    for test_image_path_unsafe in glob.glob(test_resized_unsafe_img_path + "*.png"): #Note: Must use local path for image loading!
        img = cv.imread(test_image_path_unsafe)
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
        edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
        edgemap_img_name = test_edgemap_unsafe_img_path + "img_" + str(index) + ".png"
        cv.imwrite(edgemap_img_name, edges)                              #saves the images as a .png
        index = index + 1
    print("Unsafe test image edgemap generation: DONE")

def get_data():
    status = 1
    x = np.empty((0,2), int)
    y = np.empty((0,1), int)
    x_test = np.empty((0,2), int)
    y_test = np.empty((0,1), int)

    if os.path.isdir(edgemap_safe_img_path):
        for image_path_safe in glob.glob(edgemap_safe_img_path + "*.png"):
            img = cv.imread(image_path_safe)
            count = np.count_nonzero(img >= 56)
            edge_score = count/4096
            x = np.append(x, np.array([[edge_score, 0]]), axis=0)
            y = np.append(y, np.array([[0]]), axis=0)
    else:
        print(edgemap_safe_img_path + " does not exist")
        status = 0

    if os.path.isdir(edgemap_unsafe_img_path):
        for image_path_unsafe in glob.glob(edgemap_unsafe_img_path + "*.png"):
            img = cv.imread(image_path_unsafe)
            count = np.count_nonzero(img >= 56)
            edge_score = count/4096
            x = np.append(x, np.array([[edge_score, 1]]), axis=0)
            y = np.append(y, np.array([[1]]), axis=0)
    else:
        print(edgemap_unsafe_img_path + " does not exist")
        status = 0

    if os.path.isdir(test_edgemap_safe_img_path):
        for test_image_path_safe in glob.glob(test_edgemap_safe_img_path + "*.png"):
            img = cv.imread(test_image_path_safe)
            count = np.count_nonzero(img >= 56)
            edge_score = count/4096
            x_test = np.append(x_test, np.array([[edge_score, 0]]), axis=0)
            y_test = np.append(y_test, np.array([[0]]), axis=0)
    else:
        print(test_edgemap_safe_img_path + " does not exist")
        status = 0

    if os.path.isdir(test_edgemap_unsafe_img_path):
        for test_image_path_unsafe in glob.glob(test_edgemap_unsafe_img_path + "*.png"):
            img = cv.imread(test_image_path_unsafe)
            count = np.count_nonzero(img >= 56)
            edge_score = count/4096
            x_test = np.append(x_test, np.array([[edge_score, 1]]), axis=0)
            y_test = np.append(y_test, np.array([[1]]), axis=0)
    else:
        print(test_edgemap_unsafe_img_path + " does not exist")
        status = 0

    print("Collected data from edgemaps")
    return status,x,y,x_test,y_test

status,x,y,x_test,y_test = get_data();  #You may need to comment lines 157-161 out until your images are populuted in their respective folders
if status:
    x = x[:,0]/max(x[:,0])
    x = x.reshape(-1,1)
    x_test = x_test[:,0]/max(x_test[:,0])
    x_test=x_test.reshape(-1,1)
else:
    print("Data not loaded correctly")

#%% Model
def model (x_p,w):
    x1 = x_p[:,0]
    x1 = x1.reshape(-1,1)
    a = w[0] + np.dot(x1,w[1])
    return a

#%% Sigmoid Function
def sigmoid(t):
    return 1/(1+np.exp(-t))

#%% Cross Entropy Cost Function
def cross_entropy(w,x,y):
    a = sigmoid(model(x,w))
    ind = np.argwhere(y==0)[:,0]
    cost = -np.sum(np.log(1-a[ind]))
    ind = np.argwhere(y==1)[:,0]
    cost -= np.sum(np.log(a[ind]))
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

def c(t):
    c = cross_entropy(t,x,y)
    return c

def train_model():
    #Get data

    max_iter = 1000
    w = np.array([[-100],[1.]])
    [weightings,cost] = gradient_descent(c,1,max_iter,w)

    weights = weightings[max_iter]
    # plt.plot(sigmoid(weights[0] + weights[1]*x))
    # plt.show()
    plt.scatter(x,y)
    xp=np.array([np.linspace(0,1,200)])
    xp = xp.reshape(-1,1)
    plt.plot(xp,sigmoid(model(xp,weights)))
    plt.show()
    return x,y,weights


    #%% Confusion Matrix
def test_model(x_test,y_test,weights):
    weights = np.array([-14.587, 97.3]) #This is hard coded in right now
    weights=weights.reshape(-1,1)
    prediction = sigmoid(model(x_test,weights))
    # prediction = prediction.reshape(-1,1)
    actual = y_test
    # actual=actual.reshape(1,-1)

    a = 0
    b = 0
    c = 0
    d = 0

    for i in range(x_test.size):
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

#%%Parse arguments
#
# Usage: python .\main.py [-h] [--resize RESIZE] [--edges EDGES]
#
parser = argparse.ArgumentParser(description='UAV Safe Landing Project Script')
parser.add_argument("-r","--resize", type=str, default='none', help='Resize dataset to 64x64. Default value = \'none\'. Options = \'safe\', \'unsafe\', \'all\'' )
parser.add_argument("-e","--edges", type=str, default='none', help='Generate edgemaps for resized images. Default value = \'none\'. Options = \'all\'' )
parser.add_argument("-t","--train", action='store_true', help="Train model using generated edgemaps")
args = parser.parse_args()

#Resize images if argument passed
resize = args.resize
if resize == "all":
    resize_images(3)
elif resize == "safe":
    resize_images(1)
elif resize == "unsafe":
    resize_images(2)
elif resize == "test_data":
    resize_images(4)
elif resize != "none":
    print("Invalid resize argument!")

#Generate Edgemaps
edges = args.edges
if edges == "all":
    generate_edge_maps()
elif edges != "none":
    print("Invalid edges argument!")

#Train Model
train = args.train
if train == True:
    x,y,w = train_model()
    test_model(x,y,w)
    #print(x)
    #print(y)
