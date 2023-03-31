# Image Tool
## Overview
Displays the Pixel Spectrum instance at the chosen pixel of the Output Image. As well as displays the Hyperspectral Image and can be manipulated to.

This is still a work in progress. Parts of the user interface are not operational. 

The sample file can be run with the following command:
`./image_tool`

![IMAGETOOL](../images/startingpage)

### False Image:
 Combines and rearranges the ratios of the rgb from one or multiple source images that results in a final image. Helping to visualize information not easily seen by the human eye and approach the image from a different perspective.


**Image Layer:**
	Red(+), Green(+), Blue(+): Added correlated color values to the False Image.


### Semantic Image: 
Displays the contour image of the semantic segmentation image. *(Outlined in black)*

**Interactive bullet points:** *Spectral Angle Mapper, Spectral Correlation Mapper, Spectral Information Divergence*<br />
:Demonstates the different spectral similarities algorithms to specify classification.

### Spectral Similarity Image:

### Spectral Database: 
Import old/new databases, saving reference curves for classification and spectral/semantation.

### Tiled Image: 
Displays json image of tiled image of the semantic image at various points of the day.

**Pixel Spectrum:**

**Display False Image:**
**Standard RGB Image:**
**Quit:** Closes the Hyperspectral Image Analysis Tool.

### (None): 
Import a new hyperspectral image

**Replace sample name inside the cpp folder**

`string file_name2="../../HyperImages/(img1).tiff";`

