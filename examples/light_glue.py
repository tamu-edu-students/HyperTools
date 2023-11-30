import sys
sys.path.insert(1,'submodules/LightGlue/')

from lightglue import LightGlue, SuperPoint, DISK, match_pair, SIFT, ALIKED
from lightglue.utils import load_image, rbd
from lightglue.viz2d import plot_images, plot_keypoints, plot_matches, save_plot
import torch
import pathlib
import os
from tqdm import tqdm
import json 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from multiprocessing import cpu_count, Process, Value, Array, Pool, TimeoutError
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

def extract_rgb(cube, red_layer=78 , green_layer=40, blue_layer=25,  visualize=False):

    
    red_img = cube[ red_layer,:,:]
    green_img = cube[ green_layer,:,:]
    blue_img = cube[ blue_layer,:,:]

        
    data=np.stack([red_img,green_img,blue_img], axis=-1)
    # print(data.shape)
    #print(type(image))

    #convert to 8bit
    x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
    image=(x_norm*255).astype('uint8')
    if visualize:
        #pass
        plt.imshow(image)
        plt.show()
    return image  

def extract_single_layer(cube, layer):
    return cube[layer, :, :]

def sum_of_spectrum(cube):
    data = np.sum(cube, axis=0)
    return data

def avg_of_spectrum(cube):
    data = np.average(cube, axis=0)
    return data

def read_spec_json(file_name):
    
    f=open(file_name)
    data=json.load(f)
    
    color_array=np.array([])
    spectral_array=np.array([])
    for i in data['Color_Information']:
        temp_array=np.array([])
        for j in data['Color_Information'][i]:
            #print(i,j,data['Color_Information'][i][j])
            temp_array=np.hstack((temp_array, np.array([data['Color_Information'][i][j]]))) 
        if (len(color_array)==0):
            color_array =np.hstack((color_array,temp_array))
        else:
            color_array =np.vstack((color_array,temp_array))
    for i in data['Spectral_Information']:
        temp_array=np.array([])
        for j in data['Spectral_Information'][i]:
            #print(i,j,data['Spectral_Information'][i][j])
            temp_array=np.hstack((temp_array, np.array([data['Spectral_Information'][i][j]])))
        if (len(spectral_array)==0):
            spectral_array =np.hstack((spectral_array,temp_array))
        else:
            spectral_array =np.vstack((spectral_array,temp_array))
               
    f.close()
    return color_array, spectral_array   

def SpectralAngleMapper(cur_pixel, ref_spec):
        dot_product = np.dot( cur_pixel, ref_spec)
        norm_spectral=np.linalg.norm(cur_pixel)
        norm_ref=np.linalg.norm(ref_spec)
        denom = norm_spectral * norm_ref
        if denom == 0:
            return 3.14
        alpha_rad=math.acos(dot_product / (denom)); 
        return alpha_rad*255/3.1416 

def perform_similarity(cube1, spectral_array, index=0):
    #cube 1 x,y,channels
    #spectral array item_num,channels ie 3,164
    cube1 = np.transpose(cube1, (1, 2, 0)) # x, y, channels 
    # print(cube1.shape)
    if (cube1.shape[2] != spectral_array.shape[1]):
         print("Error: cube and spectral array have different number of channels")
         return
    else :
        
        #perform spectral angle mapper
        spec_sim_arr_1=np.zeros((cube1.shape[0], cube1.shape[1]))
        # print(len(spectral_array))
        k=index
        # for k in range(len(spectral_array)):
        # print(k)
        ref_spec =  spectral_array[k,:]
        arr=np.zeros((cube1.shape[0], cube1.shape[1])) #y,x
        
        #thread pool to speed up computation
        with Pool(processes=cpu_count()) as pool:
                for  i in  range (cube1.shape[0]):
                    itr=0
                    #for result in pool.map(SpectralCorrelationMapper, ( data[i,j,:] for j in range((cube.height)) ) ):
                    for result in pool.starmap(SpectralAngleMapper, ((cube1[i, j, :], ref_spec) for j in range(cube1.shape[1]))):
                        arr[i,itr]=result  
                        itr+=1
                    spec_sim_arr_1[:,:]=arr
        # plt.imshow(spec_sim_arr_1[:,:,k]/3.15) 
        # plt.show()

        return spec_sim_arr_1

def perform_pca(cube):
    # print(cube.shape) #channels, x, y
    #convert to x,y,channels
    cube = np.transpose(cube, (1, 2, 0)) # x, y, channels
    # print(cube.shape)
    # convert to 2d array
    cube = cube.reshape(cube.shape[0]*cube.shape[1], cube.shape[2])
    print(cube.shape)
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 3) # rgb channels when 3 is used
    result = pca.fit_transform(cube[1:1000,:])
    print(result.shape)
    # https://www.geeksforgeeks.org/principal-component-analysis-with-python/
    # need to reshape result to x,y,channels
    # running into issues right now need to test on different platform 
    return 

    
if __name__ == "__main__":    
    
    # SuperPoint+LightGlue # not good for hyperspectral images not many matched points neither is sift or aliked
    # extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    # matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
    
    # extractor = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
    # matcher = LightGlue(features='sift').eval().cuda()  # load the matcher
    
    # extractor = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
    # matcher = LightGlue(features='aliked').eval().cuda()  # load the matcher

    # or DISK+LightGlue # getting decent results for hyperspectral images
    extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    # image0 = load_image('submodules/LightGlue/assets/sacre_coeur1.jpg').cuda()
    # image1 = load_image('submodules/LightGlue/assets/sacre_coeur2.jpg').cuda()
    # image0 = load_image('images/pan1.tiff').cuda()
    # image1 = load_image('images/pan2.tiff').cuda()



    #below is a way to load hyperspectral images that are tiff files
    mylist = []
    loaded,mylist = cv2.imreadmulti(mats = mylist, filename = "../HyperImages/img1.tiff", flags = cv2.IMREAD_ANYCOLOR )
    cube1=np.array(mylist)
    cube1 = cube1[:, :, :]  
    # print(cube1.shape)

    mylist = []
    loaded,mylist = cv2.imreadmulti(mats = mylist, filename = "../HyperImages/img2.tiff", flags = cv2.IMREAD_ANYCOLOR )
    cube2=np.array(mylist)
    cube2 = cube2[:, :, :]  
    # print(cube2.shape)
    
    #use below to set the both images to be the same
    # cube2 = cube1

    #single channel from hyperspectral image
    # image0 = extract_single_layer(cube1, 100)
    # image1 = extract_single_layer(cube2, 100)

    #rgb image from hyperspectral image
    image0=extract_rgb(cube1)
    image1=extract_rgb(cube2)

    # spectral similarity image from hyperspectral image
    # color_array, spectral_array= read_spec_json('json/spectral_database_U20.json')
    # image0 = perform_similarity(cube1, spectral_array,0)
    # image1 = perform_similarity(cube2, spectral_array,0)

    # sum of sprectum 
    # image0= sum_of_spectrum(cube1)
    # image1= sum_of_spectrum(cube2)

    # average of spectrum  
    # #visually similar to sum of spectrum after normalization
    # image0= avg_of_spectrum(cube1)
    # image1= avg_of_spectrum(cube2)


    # pca of hyperspectral image
    # image2= perform_pca(cube1)




    # below is to process the image to be in the correct format for the lightglue library

    # make sure it is chann, x, y
    # print(image0.shape)
    # print(len(image0.shape))
    
    # if only x,y then add channel dimension    
    if (len(image0.shape)==2):
        image0 = np.expand_dims(image0, axis=0)
    if (len(image1.shape)==2):
        image1 = np.expand_dims(image1, axis=0)
    # print(image0.shape)
    
    # if in x,y,channel then convert to channel,x,y
    if(image0.shape[2]==3):
        image0 = np.transpose(image0, (2, 0, 1)) # channels, x, y
    if(image1.shape[2]==3):
        image1 = np.transpose(image1, (2, 0, 1))
    
    if (image0.shape[0] == 3 or image0.shape[0] == 1) :
        pass
        # has correct dimensions
    else:
        print("Error: image0 does not have 1 or 3 channels")
        sys.exit()

    if (image1.shape[0] == 3 or image1.shape[0] == 1) :
        pass
        # has correct dimensions    
    else:
        print("Error: image0 does not have 1 or 3 channels")
        sys.exit()

    # convert values to be between 0 and 1
    if (np.max(image0) > 1):
        image0 = image0/255
    if (np.max(image1) > 1):
        image1 = image1/255
    
    # convert to torch tensor for lightglue library    
    if (type(image0) != torch.Tensor):
        # print("Error: image0 is not a torch tensor")
        # sys.exit()
        image0 = torch.from_numpy(image0).float().cuda()
        # image0 = image0.cpu()
    if (type(image1) != torch.Tensor):  
        # print("Error: image1 is not a torch tensor")
        # sys.exit()
        image1 = torch.from_numpy(image1).float().cuda()
        # image1 = image1.cpu()
    
    # print(image0.shape)
    # print(image1.shape)
    
    # extract local features
    feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)
    # print(feats0) # this is a dictionary with keys 'keypoints' and 'descriptors'   and keypoint scores and other information about the image, device
    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    print('number of matched features: ', len(matches))
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    plot_images([image0.cpu(), image1.cpu()])
    # plot_images([image0, image1])
    #plot_keypoints(points0, points1)
    plot_matches(points0[:, :], points1[:, :])
    plt.show()


    # tradtional way with opencv
    # img1 = cube1[100, :, :]
    # img2 = cube2[100, :, :]
    # # Initiate SIFT detector
    # sift = cv2.SIFT_create()
    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)
    # # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    # # Apply ratio test
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])
    # # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()

    # # Initiate ORB detector
    # orb = cv2.ORB_create()
    # # find the keypoints and descriptors with ORB
    # kp1, des1 = orb.detectAndCompute(cube1[100, :, :],None)
    # kp2, des2 = orb.detectAndCompute(cube2[100, :, :],None)
    # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # # Match descriptors.
    # matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)
    # # Draw first 10 matches.
    # img3 = cv2.drawMatches(cube1[100, :, :],kp1,cube2[100, :, :],kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()