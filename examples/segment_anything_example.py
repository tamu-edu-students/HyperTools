import sys
sys.path.insert(2,'submodules/segment-anything/')
sys.path.append("submodules/segment-anything/segment_anything/")


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import time 
from matplotlib import animation
import json 
from multiprocessing import cpu_count, Process, Value, Array, Pool, TimeoutError
import math

# below dependencies are maybe future work for integration with hyperspectral data
# from scipy import stats as st
# from pyopencl.tools import get_test_platforms_and_devices
# import pyopencl as cl
# import cuvis
# lib_dir = os.getenv("CUVIS")
# data_dir = os.path.normpath(os.path.join(lib_dir, os.path.pardir, "sdk", "sample_data", "set1"))







if __name__ == "__main__":
    
    
    # this file is a work in progress and is still not working. 
    
    print("start")
    torch.cuda.empty_cache()
    sam = sam_model_registry["vit_b"](checkpoint="../HyperImages/segment_anything_model/sam_vit_b_01ec64.pth")
    sam.to(device='cuda')
