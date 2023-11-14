import sys
sys.path.insert(1,'submodules/LightGlue/')

from lightglue import LightGlue, SuperPoint, DISK, match_pair
from lightglue.utils import load_image, rbd
from lightglue.viz2d import plot_images, plot_keypoints, plot_matches, save_plot
import torch
import pathlib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')

# SuperPoint+LightGlue # not good for hyperspectral images not many matched points neither is sift or aliked
# extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

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


image0 = cube1[100, :, :]
# image0 = np.expand_dims(image0, axis=0)
# print(image0.shape)

image1 = cube2[100, :, :]
# image1 = np.expand_dims(image1, axis=0)

image0 = torch.from_numpy(image0).unsqueeze(0).float().cuda()
image1 = torch.from_numpy(image1).unsqueeze(0).float().cuda()


# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)
# print(feats0) # this is a dictionary with keys 'keypoints' and 'descriptors'   and keypoint scores and other information about the image, device
# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
print(len(matches))
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

plot_images([image0.cpu(), image1.cpu()])
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