# Semantic Segmentation Tool: 
## Overview

This tool facilitates processing of a semantically segmented image. The aim is to process and extract information from the image for input into other repos. It converts images that were classified at the pixel-level to contours (polygonal features) with associated class labels. This tool also has capabilities to simplify contours and filter contours below a user-defined size. 

The sample file can be run with the following command after it is built. Instructions for building the code are found on the main ReadMe file for the repo. 

`./semantic_segmentation`

![IMAGETOOL](../images/semantic1)
**Replace sample name variables inside the semantic_segmentation.cpp file**
These variables are for the associated semantically segmented image and the corresponding json file that contains information about the image such as the FOV of the camera, GPS location of the center of the image, and height of the camera from the ground.

  `string file_name2="../images/`*lena3*`.png";`
  `string file_name3="../json/`*lena3*`.json";`


### Show Original Image:
Displays the original input Segmantically Segmented Image that is processed by the rest of the Semantic Segmatation tool.


### Edge Detection: 
Finds the edges of all objects by determining where the color changes in comparison to the surrounding pixels. A binary image then generated where edge pixels are shown in white and non-edge pixels are displayed in black. 


### Min Countour Area (m^2):
Changes the minimum amount of square meters in relation to pixels to be rendered in the updated image. Increasing the minimum(+) helps to reduce noise in the image by filtering out polygons below the specified size.
In order to convert pixels to area (meters squared) the user needs to know the average distance between the image taken to the ground. The associated parameter is "double avgDist".


### Polygon Approximation Coefficient:
Approximate polygons through increasing the approximation coefficient(+) which reduces the number of vertices in a polygon.
The associated parameter is "double polygon_approx_coeff"


### Show Contours:
Displays contour image. This image shows the contours after filtering by size and simplifying contours. 

### Fidelity Levels
These fidelity levels have different values for size filtering and polygon approximation coefficients. The result of the fidelity levels are shown in the contour image. The different fidelity levels can be selected by clicking the corresponding button. 


### Show Differences
This shows the differences between the input image and the contour image. Differences are defined as a different in class between the contour image and the input image. Pixels that are of a different class in the input and contour image are shown in white. Pixels that are the same class in the input and contour image are shown in black. 


**Quit:**
Closes the Semantic Segmantation Tool.



