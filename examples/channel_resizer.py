import sys
sys.path.insert(1,'submodules/LightGlue/')
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from multiprocessing import cpu_count, Process, Value, Array, Pool, TimeoutError
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import tifffile
import numpy as np
np.set_printoptions(threshold=np.inf)


def load_hsi(file_name):
    # load hyperspectral image
    _, extension = os.path.splitext(file_name)

    if extension == '.tiff': 
        #below is a way to load hyperspectral images that are tiff files
        mylist = []
        loaded,mylist = cv2.imreadmulti(mats = mylist, filename = file_name, flags = cv2.IMREAD_ANYCOLOR )
        cube=np.array(mylist)
        cube = cube[:, :, :]  
    else :
        print("Error: file type not supported")
        return 
    return cube 

def load_rgb(file_name):
    # load RGB image
    img = cv2.imread(file_name)
    rgbArray = np.array(img)
    rgbArray = np.transpose(rgbArray, (2, 0, 1))  # Transpose the array to make the third dimension the first dimension
    return rgbArray

def load_dat(file_name):
    # load .dat file into a 3-dimensional numpy array
    cube = np.fromfile(file_name, dtype=np.float32)
    cube = cube.reshape((-1, 512, 512))  # adjust the shape according to your data dimensions
    return cube

def reduce_channels(hsi_array):
    # keep the first and last layer
    # reduce to 64 total
     
    indicesKeep = np.linspace(1, hsi_array.shape[0] - 2, num=62, dtype=int) # create a numpy array of 62 evenly spaced indicies to keep
    indicesDel = np.setdiff1d(np.arange(1, hsi_array.shape[0] - 1), indicesKeep) #create an array of indicies to delete

    hsi_array = np.delete(hsi_array, indicesDel, axis=0) # delete the indicies from the array so that 64 are remaining

    return hsi_array

def fill_channels(hsi_array):
    # fill the channels to 64
    # fill with a copy of the previous layer
    layersToAdd = 64 - hsi_array.shape[0]
    indices = np.linspace(1, hsi_array.shape[0] - 2, num=layersToAdd, dtype=int) #find where to add layers

    for i in indices:
        prev_layer = hsi_array[i] / 2
        next_layer = hsi_array[i + 1] / 2
        interpolated_layer = (prev_layer + next_layer)
        hsi_array = np.insert(hsi_array, i+1, interpolated_layer, axis=0)

    return hsi_array

def save_hsi(file_name, hsi_array):
    # save hyperspectral image as tiff file
    tifffile.imwrite(file_name, hsi_array)


#------------Main------------
    
image_name = input("Enter the file name: ")
image_path = f"../../HyperImages/{image_name}"

imgArray = load_hsi(image_path)

print("Original array shape: \n", imgArray.shape)
layers = len(imgArray)

print("Number of layers:", layers)

if(layers > 64):
    print("Reducing channels...")
    resizedArray = reduce_channels(imgArray)
elif(layers < 64):
    print("Filling channels...")
    resizedArray = fill_channels(imgArray)


print("Resized array shape: \n", resizedArray.shape)

image_name = image_name[:-5] + f"_{layers}_to_64.tiff"
save_hsi(f"../../HyperImages/{image_name}", resizedArray)
print("Saved as:", image_name)