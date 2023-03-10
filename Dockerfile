FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install cmake build-essential  g++ wget unzip libgtk2.0-dev pkg-config  libjsoncpp-dev libcanberra-gtk-module libgtk2.0-dev libgtk-3-dev libboost-all-dev glade  git -y 


RUN mkdir -p opencv_build cd opencv_build && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip && \
    unzip opencv.zip &&\
    rm opencv.zip && \
    unzip opencv_contrib.zip &&\
    rm opencv_contrib.zip && \
    mkdir -p build && \ 
    cd build 


RUN cmake \
    -DOPENCV_ENABLE_NONFREE:BOOL=ON  \
    -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules  ../opencv-4.x/    \
    -D OPENCV_GENERATE_PKGCONFIG=ON 



RUN make -j$(nproc) &&\
    make install && \
    ldconfig
 

WORKDIR /home

RUN cd /home &&\
    bash

