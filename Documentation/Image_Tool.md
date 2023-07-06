# Image Tool
## Overview

This tool can be used to quickly analyze a hyperspectral image. There are many capabilities of this tool. It was designed for images from an Ultris X20 Plus, but other hyperspectral images should be able to be used. The usage and description of the capabilities are detailed below. 

This is still a work in progress. Parts of the user interface may have bugs. Please create an issue for any bugs, so they can be fixed. 

The sample file can be run with the following command after it is built. Instructions for building the code are found on the main ReadMe file for the repo. 

`./image_tool`

The below user interface should appear on startup. Sometimes it may be slow to start, due to large file sizes associated with loading the hyperspectral image. 

<img src = docimages/Startingpage.png>


### Variables:
This is the hyperspectral image that is loaded on startup. It is not necessary, but recommended. A hyperspectral image can be loaded from the user interface during run-time. If a hyperspectral image is not loaded, errors may occur.  


`string file_name2="../../HyperImages/(img1).tiff";`


### False Image:

 This capability allows specific layers to be set as the red, blue, and green channels of a false image. The individual layers are combined together to generate the false image. This is helpful for visualizing hyperspectral images and looking at specific wavelengths to create custom filters. 
 

**Image Layer:**

	Red(-/+), Green(-/+), Blue(-/+) spin buttons allow the user to select which layers should correspond to which color channel.

**Display False Image:**

This button can be clicked if a non-false image is being displayed in the user interface. The result is that the false image will be shown. 

**Standard RGB Image:**

This sets the blue channel to 65, green channel to 104, and the red channel to 163. These values were chosen to highlight vegetation and water in hyperspectral images. The corresponding wavelengths are camera dependent. 

**Reset False Image:**

This button resets the Image Layer values to 0. 

### Semantic Image: 
Different spectral similarity algorithms can be used to generate a semantic segmentation image. The radio button for the desired spectral simialarity button for analysis should be selected. After, click the "Update Semantic Image" button. This will perform the analysis and show the result in the user interface. 


### Spectral Similarity Image:
Different spectral similarity algorithms can be used to generate a semantic segmentation image. The radio button for the desired spectral similarity button for analysis should be selected. The items in the reference spectral database are loaded. The item and related reference spectrum can be selected with the drop down box under "Object Being Analyzed". After both the algorithm and object are selected, click the "Update Spectral Similarity Image" button. This will perform the analysis and show the result in the user interface. 


### Spectral Database: 

This tab has multiple capabilities dealing with a spectral database. A spectral database can be loaded with the file chooser in the tab. A new spectral database can be created with the "Create Database" button. The name of the database will be what is entered in the text box next to the button. The created database is blank, but can be filled by save spectrums. A spectrum can be saved by clicking the "Save Spectrum to Database" button. This should be selected after entering a name of the spectrum. The spectrum that is saved will be the one that is displaying in the user interface. In order to select a new spectrum, click a new point in the hyperspectral image. The last capability in this tab is to clear the database. This will remove all spectral and color information associated with the spectral database. 


### Tiled Image: 

Displays tiled image of all the layers of the hyperspectral image in increasing order of electromagnetic range. This was hard-coded for 164 layers from the Ultris X20 Plus, so hyperspectral images with a different number of layers may lead to unexpected results. 

**Pixel Spectrum:**

The spectral curve of a pixel is displayed. The pixel can be selected by clicking a point in the hyperspectral image. This will update each time a new point the hyperspectral image is clicked. Only one spectral curve is shown in the graph. 

**Quit:** 

Closes the Hyperspectral Image Analysis Tool.

### File Chooser in top left corner: 
Imports a new hyperspectral image to be analyzed. 


