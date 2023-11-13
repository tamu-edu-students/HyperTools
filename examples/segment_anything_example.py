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

def extract_rgb(cube, red_layer=78 , green_layer=40, blue_layer=25,  visualize=False):

    
    try: #for loading using opencv
        red_img = cube[:,:, red_layer]
        green_img = cube[:,:, green_layer]
        blue_img = cube[:,:, blue_layer]
    except: #for loading using cuvis
        red_img=cube.array[:,:, red_layer]
        green_img=cube.array[:,:, green_layer]
        blue_img=cube.array[:,:, blue_layer]
        
    data=np.stack([red_img,green_img,blue_img], axis=-1)
    #print(image.shape)
    #print(type(image))

    #convert to 8bit
    x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
    image=(x_norm*255).astype('uint8')
    if visualize:
        #pass
        plt.imshow(image)
        plt.show()
    return image  

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)




if __name__ == "__main__":
    
    
    
    # print("start")
    torch.cuda.empty_cache()
    # model needs to be downloaded from online (https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    # there are other models that can be used as well.
    sam = sam_model_registry["vit_b"](checkpoint="../HyperImages/segment_anything_model/sam_vit_b_01ec64.pth")
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # load rgb image into numpy array
    rgb_img = cv2.imread("images/lena3.png")
    rgb_img = cv2.cvtColor(rgb_img,  cv2.COLOR_BGR2RGB)
    
    #load hyperspectral image into numpy array
    mylist = []
    loaded,mylist = cv2.imreadmulti(mats = mylist, filename = "../HyperImages/img1.tiff", flags = cv2.IMREAD_ANYCOLOR )
    cube1=np.array(mylist)
    cube1 = cube1[:, :, :]  # channels, x, y
    cube1 = np.transpose(cube1, (1, 2, 0)) # x, y, channels 
    # print(cube1.shape)
    
    # create rgb image from hyperspectral image
    rgb_img= extract_rgb(cube1, 163 , 104, 65,  False)

    
    
    #generate masks on rgb image 
    startTime = time.time()
    masks = mask_generator.generate(rgb_img)
    endTime = time.time()
    print("time to generate masks: ", endTime-startTime)
    
    
    
    #tune SAM parameters to generate more masks
    #parameters were found online and havent been tuned 
    startTime = time.time()
    mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    
    
    masks2 = mask_generator_2.generate(rgb_img)
    endTime = time.time()
    print("time to generate masks: ", endTime-startTime)
    
    #show masks on rgb image
    plt.imshow(rgb_img)
    show_anns(masks2)
    plt.show()
