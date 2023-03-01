import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd

file_hyper = input("Please enter the hyperspectal image name: ")
file_gt = input("Please enter the ground truth image name: ")
current_dir=(os.getcwd())
os.chdir('../GroundTruthImgs') # change this to where you have stored your .mat, .tiff files

img_hyper = loadmat(file_hyper + '.mat')
img_gt = loadmat(file_gt + '.mat')
key_hyper = list(img_hyper.keys())[3]
key_gt = list(img_gt.keys())[3]
    
def plot_dataset(dataset):
    plt.figure(figsize=(8, 6))
    band_no = np.random.randint(dataset.shape[2])
    plt.imshow(dataset[:,:, band_no], cmap='jet')
    plt.title(f'Band-{band_no}', fontsize=14)
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
def plot_gt(ground_truth):
    plt.figure(figsize=(8, 6))
    plt.imshow(ground_truth, cmap='jet')
    plt.axis('off')
    plt.colorbar(ticks= range(0,16))
    plt.show()

ground_truth = img_gt[key_gt]
print(f'Ground Truth: {ground_truth.shape}')
#plot_gt(ground_truth)

dataset = img_hyper[key_hyper]
print(f'Dataset: {dataset.shape}')
#plot_dataset(dataset)

def extract_pixels(dataset, ground_truth):
    df = pd.DataFrame()
    for i in tqdm(range(dataset.shape[2])):
        df = pd.concat([df, pd.DataFrame(dataset[:, :, i].ravel())], axis=1)
    df = pd.concat([df, pd.DataFrame(ground_truth.ravel())], axis=1)
    df.columns = [f'band-{i}' for i in range(1, 1+dataset.shape[2])]+['class']
    return df

df = extract_pixels(dataset, ground_truth)

df.to_csv('Dataset.csv', index=False)

df = pd.read_csv('Dataset.csv')

print(df.loc[:, 'class'].value_counts().sort_index())

