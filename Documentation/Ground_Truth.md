# Ground Truth Example: 
## Overview

This code loads a hyperspectral image and its corresponding ground truth image, and then performs spectral averaging to find the average spectrum for each semantic class in the ground truth image.

The sample file can be run with the following command after it is built. Instructions for building the code are found on the main ReadMe file for the repo. 

`./ground_truth_example`

### Variables:
These variables are for the associated hyperspectral image and ground truth image. It assumes that that the ground truth image is a single channel image with pixel values that are integers from 0-N, where N is the number of semantic classes. 

`string file_name1="../../HyperImages/`*Indian_pines_corrected*`.tiff";`
`string file_name2="../../HyperImages/`*Indian_pines_gt*`.tiff";`

### Pixel Coordinates:
The code finds the pixel coordinates of each semantic class by looping through the ground truth image and adding each pixel's coordinates to the appropriate class vector in a 2D vector, `class_coordinates`.


### Average Spectrum Array: 
2D array `avgSpectrums` stores the average spectra for each semantic class. The average spectrum for each class is found by looping through the pixels in the class, finding the spectrum for each pixel in the hyperspectral image, and then taking the average of the spectra for all pixels in the class. 


### Average Spectrum Vector:
 Used as a spectral database for classification algorithms that use spectral signatures to identify different materials or land cover types. Storing the average spectra in `avgSpectrums_vector`.

The code ignores the unknown class with a value of 0.
