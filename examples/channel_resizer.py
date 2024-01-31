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

def reduce_channels(hsi_array):
    # keep the first and last layer
    # reduce to 64 total
     
    indicesKeep = np.linspace(1, hsi_array.shape[0] - 2, num=62, dtype=int) # create a numpy array of 62 evenly spaced indicies to keep
    indicesDel = np.setdiff1d(np.arange(1, hsi_array.shape[0] - 1), indicesKeep) #create an array of indicies to delete

    print(indicesKeep)
    hsi_array = np.delete(hsi_array, indicesDel, axis=0) # delete the indicies from the array so that 64 are remaining

    return hsi_array

image_path = "../../HyperImages/img1.tiff"

imgArray = load_hsi(image_path)
layers = len(imgArray)

print("Number of layers:", layers)

if(layers > 64):
    print("Reducing channels...")
    resizedArray = reduce_channels(imgArray)

print("Resized array shape: \n", resizedArray.shape)
