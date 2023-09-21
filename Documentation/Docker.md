# Docker
## Overview
Docker allows this code to be set up in a container. This can be used to make sure all the dependencies are installed without affecting the base-system. The included Dockerfile is for the most recent version of OpenCV, but it can be edited to install a different version. This allows a different version of OpenCV to be used than what is installed on a base-system. For example, the Cubert SDK requires OpenCV 4.2. Docker allows this version of OpenCV to be located within a container and not interfere with what is located on the base-system. It is expected that a Docker Volume will be used to pass hyperspectral images to be mounted in the Docker Container. This allows files to be shared between the base-system and container. It is also expected that visualization will be needed with the example files. This file will detail how to set up your base-system to use HyperTools with Docker. 

## Downloading Docker
Docker Engine can be installed by following the directions at the below link. 
https://docs.docker.com/engine/install/

Docker Desktop can be downloaded from the below link. 
https://www.docker.com/products/docker-desktop/

## Building Docker Image 
The included Dockerfile installs the dependencies for the CPU aspects of the code. The GPU dependencies are not installed. This means that the CUDA files will not function properly with Docker. HyperTools is not included in the Dockerfile, because it is passed as part of the Docker Volume. The Dockerfile can be built with the below command. Make sure that you are in the same directory as the Dockerfile. More information about this command can be found at the below link. 
https://docs.docker.com/engine/reference/commandline/build/ 

`docker build -t hypercode_base -f Dockerfile . ` 

## Visualization with Docker
Different operating systems have different methods for allowing the user interfaces in the example files to be visualized. Below are some helpful tips in order to allow a user to visualize the examples within a Docker Container.

### Ubuntu 
1. Make sure X11 is installed

    `sudo apt-get install xorg openbox`
2. Allow connections to the X server. 

    `xhost +local:*`

3. In order to stop allowing connections. This can be done after your Docker container stops, but is not necessary. 

    `xhost +local:*`

### Windows
1. It is assumed that Choclatey is already installed. 
Install directions can be found at the below link. https://chocolatey.org/install 

2. The below command installs VcXsrv Windows X Server. 

    `choco install vcxsrv`

3. After vcsrv is done installing, run Xlaunch. On the Display settings page, select "Multiple windows". On the Client startup page, select "Start no client". On the Extra settings, select all the checkboxes. Lastly, save and finish the configuration. 

4. In the terminal

    `set-variable -name DISPLAY -value YOUR-IP:0.0`
5. The saved configuration will have to be loaded when a Docker Container is running, in order for visualization. 

The directions for this are based on the below link. 
https://dev.to/darksmile92/run-gui-app-in-linux-docker-container-on-windows-host-4kde


### Mac 

1. Install XQuartz: https://www.xquartz.org/
2. Launch XQuartz. Under the XQuartz menu, select Preferences
3. Go to the security tab and ensure "Allow connections from network clients" is checked.
4. Run xhost + ${hostname} to allow connections to the macOS host 
5. Setup a HOSTNAME env var export HOSTNAME=`hostname`
Add the following to your docker-compose.
In the terminal:

    `% IP=$(/usr/sbin/ipconfig getifaddr en0)`

    `echo $IP`

    `% /opt/X11/bin/xhost + "$IP"`

*cd to the Hypertools folder in the terminal and then run*

`pwd`


*the location of the folder can be traced with the following input into the terminal to replace into the next line of code accordingly*

## Running HyperTools with Docker 

The below commands can be used to mount a Docker Volume that contains HyperTools and the Images to be analyzed. More information on the docker run command can be found at the below link. https://docs.docker.com/engine/reference/commandline/run/


### Windows

`docker run -ti --rm -e DISPLAY=$DISPLAY -v /path/to/code/and/images:/home  hypercode_base `


### Ubuntu 
`docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -v /path/to/code/and/images:/home  -e DISPLAY=unix$DISPLAY  hypercode_base`

### Mac


`docker run -it --rm -e DISPLAY="${IP}:0" -v /tmp/.X11-unix:/tmp/.X11-unix hypercode_base`
