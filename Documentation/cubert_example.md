# Cubert Example
## Overview
Using Raw data:
This C++ code processes a hyperspectral image for reflectances using the cornfields dataset and applies hyperspectral image classification to the resulting image. 

The code uses the OpenCV and cuvis libraries.

This is still a work in progress. Parts of the user interface may have bugs. Please create an issue for any bugs, so they can be fixed. 

### Reflectance Processing:
Loads a hyperspectral image from the dataset with the corresponding dark, white, and distance calibration images. It then uses the cuvis library to apply reflectance correction to the hyperspectral image using the provided calibration data. The resulting image is then converted to an OpenCV Mat object, and each spectral band is extracted and displayed as a separate grayscale image.

### Hyperspectral Image Classification:
The HyperFunctions class reads a JSON file with spectral signatures for different materials, applies semantic segmentation to the image, and displays the resulting classified image.

### Key code methods
The `read_ref_spec_json` method of the HyperFunctions class is called to load a spectral database file

`SemanticSegmenter` method is used to perform semantic segmentation on the spectral bands.

 The `DispClassifiedImage` method is called to display the resulting classified image.

