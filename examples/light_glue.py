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

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# or DISK+LightGlue
#extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
#matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image('submodules/LightGlue/assets/sacre_coeur1.jpg').cuda()
image1 = load_image('submodules/LightGlue/assets/sacre_coeur2.jpg').cuda()

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

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