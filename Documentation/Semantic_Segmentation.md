# Semantic Segmentation Tool: 
## Overview
New contour approximation method which provides a label to pixels in the image based on vegetation, land, and water. The classification is determined by the spectral library, if there is no classifications then it is just rendered as the closest assumed segmentation.
`./semantic_segmentation`

![IMAGETOOL](../images/semantic1)

### Show Original Image:
Displays the original input Segmantic Image that is processed by the rest of the Semantic Segmatation tool.

### Min Countour Area (m^2):
Changes the minimum amount of square meters in relation to pixels to be rendered in the updated image. Increasing the minimum(+) helps to reduce noise in the image by filtering out polygons below a certain size.
In order to convert pixels to area(meters squared) the user needs to know the average distance between the image taken to the ground. Parameter "double avgDist", average space to area

### Polygon Approximation Coefficient:
Approximate polygons through increasing the approximation coefficient(+) which reduces the number of vertices in a polygon.
Parameter "double polygon_approx_coeff"


### Show Differences


### Edge Detection: 
Finds the edge of all objects by determining where the contour color changes in comparison to the surrounding pixels. The image is then reproduced with the edge highlighted in white demonstrating high contrast against the black background.


### Show Contours:
Displays image of contour image.

**Quit:**
Closes the Semantic Segmantation Tool.

**Replace sample name inside the cpp folder**

  `string file_name2="../images/`*lena3*`.png";`
  `string file_name3="../json/`*lena3*`.json";`

