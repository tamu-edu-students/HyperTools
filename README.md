# HyperTools
## Overview
## Installation Instructions
This code is operates with Ubuntu 20.04, but may work with other versions of Ubuntu.
Below are the installation instructions to install the dependencies for this package. Architecture to use is x86_64.

`sudo apt update`

`sudo apt upgrade`

`sudo apt  install cmake build-essential  g++ wget unzip libgtk2.0-dev pkg-config  libjsoncpp-dev libcanberra-gtk-module libgtk2.0-dev libgtk-3-dev libboost-all-dev glade -y`

### To install Nvidia packages:
 This package has some dependicies on Nvidia, however is not required.

`sudo apt install nvidia-driver-515 nvidia-dkms-515 -y `

`reboot`

`sudo apt install nvidia-cuda-toolkit -y`

### To install opencv:

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

## Usage

### Semantic Interface
### Feature Tool
### Image Tool

