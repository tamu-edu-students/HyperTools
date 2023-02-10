# HyperTools
## Overview
This package can be used to quickly analyze a hyperspectral image. The package was developed in a way to be agnostic of the camera manufacturer. The data structure that is at the foundation of the code for the hyperspectral image analysis is a vector<Mat>. There is some limited support for using a .cu3 with this code. A user interface was developed to support the usage of some of the functions that were developed, but it is a work in progress. Some of the capabilities of this package include: semantic segmentation, feature matching between single layers of two hyperspectral images, generating spectral similarity images, and extracting objects from a semantic classification image. The code was developed for data from the Ultris X20 Plus, but other Cubert images should work for the majority of the capabilities. 

## Installation Instructions
This code was developed with Ubuntu 20.04 with a x86_64 architecture, but may work with other versions of Ubuntu.
Below are the installation instructions to install the dependencies for this package. 

`sudo apt update`

`sudo apt upgrade`

`sudo apt  install cmake build-essential  g++ wget unzip libgtk2.0-dev pkg-config  libjsoncpp-dev libcanberra-gtk-module libgtk2.0-dev libgtk-3-dev libboost-all-dev glade -y`

### To install Nvidia packages:
 This package has some dependencies on NVIDIA, however is not required. This package speeds up some optional functions through the use of a NVIDIA GPU. The required packages can be installed with the below commands. 

`sudo apt install nvidia-driver-515 nvidia-dkms-515 -y `

`reboot`

`sudo apt install nvidia-cuda-toolkit -y`

### To install OpenCV:
The most recent version of OpenCV can be installed with the below commands. However, Cuvis requires version 4.2 to operate properly. The required version can be downloaded from the OpenCV Github. 

#### To install the most recent version of OpenCV:

`cd ~`

`mkdir opencv_build`

`cd opencv_build`


`wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip`

`wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip`

`unzip opencv.zip`

`unzip opencv_contrib.zip`

`mkdir -p build && cd build`

`cmake -DOPENCV_ENABLE_NONFREE:BOOL=ON  -DOPENCV_EXTRA_MODULES_PATH=/home/$USER/opencv_build/opencv_contrib-4.x/modules /home/$USER/opencv_build/opencv-4.x/     -D OPENCV_GENERATE_PKGCONFIG=ON `

`make -j$(nproc)`

`sudo make install`

#### To install the 4.2 version of OpenCV:

`cd ~`

`mkdir opencv_build`

`cd opencv_build`


`wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.zip`

`wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.zip`

`unzip opencv.zip`

`unzip opencv_contrib.zip`

`mkdir -p build && cd build`

`cmake -DOPENCV_ENABLE_NONFREE:BOOL=ON  -DOPENCV_EXTRA_MODULES_PATH=/home/$USER/opencv_build/opencv_contrib-4.2/modules /home/$USER/opencv_build/opencv-4.2/     -D OPENCV_GENERATE_PKGCONFIG=ON `

`make -j$(nproc)`

`sudo make install`




## Usage
There are a couple of example files that were created to display some of the capabilities of this package. In order to build the files to run, a build directory can be created. Below are the commands to create a build directory and the example files. There are two variables in the CMakeLists.txt file that need to be set if CUDA and Cuvis are to be used. They are "use_CUDA" and "use_cuvis". Set the variable to true to include those capabilities or false to not include the relate capabilities. 

`mkdir build`

`cd build`

`cmake ..`

`make -j$(nproc)`


### Semantic Interface
This example assumes a classified image as input. The capability to generate the classified image with the Image Tool is a work in progress. 

The sample file can be run with the following command:
`./semantic_interface`

### Feature Tool
This tool takes in two hyperspectral images and performs feature matching on a single wavelength range for the two images. 

The sample file can be run with the following command:
`./feature_tool`

### Image Tool
This is still a work in progress. Parts of the user interface are not operational. 

The sample file can be run with the following command:
`./image_tool`

### CUDA and Cuvis Examples
Three example files were created to show how .cu3 images and CUDA can be used.  

The sample files can be run with the following commands:

`./cuda_example`

`./cubert_example`

`./cuda_cubert_example`




## Credits
This package was developed by Anthony Medellin with help from Jacqueline Aleman, Anant Bhamri, Leo Feng, Adam Garsha, Sarah Lee, Albert Ma, Sudiksha Pradhan, Rijul Ranjan, and Alex Tran. 
