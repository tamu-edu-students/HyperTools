# HyperTools
## Overview
This package can be used to quickly analyze a hyperspectral image. The package was developed in a way to be agnostic of the camera manufacturer. The data structure that is at the foundation of the code for the hyperspectral image analysis is a vector<Mat>. There is some limited support for using a .cu3 with this code. The associated Github repo can be found [here](https://github.com/cubert-hyperspectral/cuvis.sdk). A user interface was developed to support the usage of some of the functions that were developed, but it is a work in progress. Some of the capabilities of this package include: semantic segmentation, feature matching between single layers of two hyperspectral images, generating spectral similarity images, and extracting objects from a semantic classification image.

## Installation Instructions
This code has associated Dockerfiles, depending on the usage. The Dockerfiles can be found in the docker directory. The Dockerfiles are not required to run this code, but are provided for convenience. Information about installing Docker can be found [here](https://docs.docker.com/get-docker/). The reason for the multiple Dockerfiles is to ensure that only the needed code is installed to reduce the amount of space used. Some of the Docker images can get rather large with the addition of CUDA. 

Make sure the Docker Daemon is running. Then in the terminal, navigate to the docker directory. There are five different Dockerfiles. The Dockerfiles are named based on the intended usage. The Dockerfiles are named as follows:
- Dockerfile_4_2_OpenCV : This is the basis for the cubert images. This uses OpenCV 4.2.0. This is recommended if you are not using CUDA code or cuvis.
- Dockerfile_4_2_OpenCV_cuda : This is the basis for the cubert images. This uses OpenCV 4.2.0. This is recommended if you are using CUDA code but not cuvis. 
- Dockerfile_cubert : This uses OpenCV 4.2.0. This is recommended if you are not using CUDA code, but using cuvis.
- Dockerfile_cubert_cuda : This uses OpenCV 4.2.0. This is recommended if you are using CUDA code and cuvis.
- Dockerfile_current_OpenCV : This uses the most recent version of OpenCV. This is recommended if you are not using CUDA code or cuvis.

The Dockerfiles can be built with the following command (Note: They only need to built once or after changes are made to the Dockerfile):

`docker build -t <image_name> -f <Dockerfile_name> .`

The recommended image_names file name paris are as follows to be replaced in the above command: 
- hypertools_4_2, Dockerfile_4_2_OpenCV
- hypertools_4_2_cuda, Dockerfile_4_2_OpenCV_cuda
- hypertools_cubert, Dockerfile_cubert
- hypertools_cubert_cuda, Dockerfile_cubert_cuda
- hypertools_current_opencv, Dockerfile_current_OpenCV

For example: 

`docker build -t hypertools_4_2 -f Dockerfile_4_2_OpenCV .`


## Usage
There are many example files in the corresponding directory. The CPP and CUDA files are built with CMakeLists.txt. Make sure to set the use_CUDA and use_cuvis variables to true if CUDA and Cuvis are to be used. Some of the executables are commented out to improve build time. Make sure to uncomment the desired executable and related lines to build the desired executable. The below commands assume that an associated Dockerfile was used. The code was also integrated with VS Code for development. The below commands assume that VS Code is being used. More information about the integration can be found [here](https://code.visualstudio.com/docs/devcontainers/create-dev-container). 

Make sure the .devcontainer/devcontainer.json file is set to the desired Dockerfile. Just uncomment the associated Dockerfile and comment out the other Dockerfiles. An example line from the file is:

`"image": "hypertools_4_2:latest"`

Mounts are used to be able to access a directory on the host. This is used for analyzing the hyperspectral images. The assumed directory format is: 
 
    .
    ├── HyperWorkspace                   
    │   ├── HyperTools          
    │   └── HyperImages                
    └── ...
In order to make sure HyperImages is mounted into the docker container make sure the following line is uncommented from the .devcontainer/devcontainer.json file:

    "mounts": [
    
    "source=${localWorkspaceFolder}/HyperImages,target=/HyperImages,type=bind,consistency=cached"
    
    ]

If you are using CUDA, make sure to uncomment the following line from the .devcontainer/devcontainer.json file to make the GPU accessible by the Docker container:

    "runArgs": [
    
    "--gpus=all"
    
    ]

Additional information about this file can be found [here](https://containers.dev/implementors/json_reference/). 


Next make sure to open the HyperTools folder in VS Code. It is important to make sure that this folder is opened in VS Code and not HyperWorkspace. This is because the .devcontainer/devcontainer.json file has to be at the top level of the workspace. In the terminal, you can navigate to the Hypertools folder, then use the following command to accomplish this:

`code .`

Once the folder is opened in VS Code, the Docker container can be built and run. This can be done by clicking the green button in the bottom left corner of VS Code and selecting "reopen in container". Sometimes there will be a pop up that asks if you want to reopen in container. If this is the case, click reopen in container. 

If changes are made to the devcontainer/devcontainer.json file, the container will need to be rebuilt. This can be done by clicking the green button in the bottom left corner of VS Code and selecting "Rebuild Container".

The terminal in the new window will be in the HyperTools directory. The code can be built with the following commands:

The followind commands need to be only done one time at the start.

`mkdir build`

`cd build`

**(Make sure you are in the HyperTools/build directory for the following commands)**

If changes are made to the CMakeLists.txt file, the code will need to be rebuilt. This can be done with the following command. This will also need to be done at the start:

`cmake ..`

The code can than be built with the following command. This will need to be redone if changes are made to any of the files:

`make -j$(nproc)`

After the executables are built they can be run with the following commands:

`./<executable_name>`

For example:

`./image_tool`





<!-- ### Semantic Interface
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


## Video
Link to video demoing some of the code: https://youtu.be/dIzrb7cCqlA -->

## Credits
This package was developed by Anthony Medellin with help from Anant Bhamri and undergraduate students at Texas A&M University through the Aggie Research Program. 