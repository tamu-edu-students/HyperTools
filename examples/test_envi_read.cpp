#include "gdal/gdal.h"
#include "gdal/gdal_priv.h"
#include "gdal/cpl_conv.h"  // for CPLMalloc()
#include "opencv2/opencv.hpp"

int main() {

// install gdal on ubuntu
// apt-get update & apt-get install libgdal-dev -y
// run ./test_envi_read

    // Register GDAL drivers
    GDALAllRegister();

    // Path to the ENVI header file (.hdr)
    const char* enviHeaderPath = "../../HyperImages/test.dat";

    // Open the ENVI file
    GDALDataset *poDataset = (GDALDataset *) GDALOpen(enviHeaderPath, GA_ReadOnly);
    
    if (poDataset != nullptr) {
        // Get information about the dataset
        int width = poDataset->GetRasterXSize();
        int height = poDataset->GetRasterYSize();
        int numBands = poDataset->GetRasterCount();

        printf("Width: %d, Height: %d, Bands: %d\n", width, height, numBands);

        std::vector<cv::Mat> imageBands;

        // Loop through bands and read data
        for (int bandNum = 1; bandNum <= numBands; ++bandNum) {
            GDALRasterBand *poBand = poDataset->GetRasterBand(bandNum);

            // Allocate memory to store pixel values
            int *bandData = (int *) CPLMalloc(sizeof(int) * width * height);

            // Read band data
            poBand->RasterIO(GF_Read, 0, 0, width, height, bandData, width, height, GDT_Int32, 0, 0);

            // Create an OpenCV Mat from the band data
            cv::Mat bandMat(height, width, CV_32SC1, bandData);

            cv::imshow("bandMat", bandMat);
            cv::waitKey(30);

            // Add the Mat to the vector
            imageBands.push_back(bandMat);

            // Release allocated memory
            CPLFree(bandData);
        }

        // Close the dataset
        GDALClose(poDataset);

    } else {
        printf("Failed to open the dataset.\n");
    }

    return 0;
}