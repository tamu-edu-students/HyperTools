#include "gdal/gdal.h"
#include "gdal/gdal_priv.h"
#include "gdal/cpl_conv.h"  // for CPLMalloc()
#include "opencv2/opencv.hpp"


using namespace std;
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
            // int *bandData = (int *) CPLMalloc(sizeof(int) * width * height);
            float *bandData = (float *) CPLMalloc(sizeof(float) * width * height);

            // Read band data
            poBand->RasterIO(GF_Read, 0, 0, width, height, bandData, width, height, GDT_Float32, 0, 0);

            // Create an OpenCV Mat from the band data
            cv::Mat bandMat(height, width, CV_32FC1, bandData);
            bandMat = bandMat.t(); 
            cv::flip(bandMat, bandMat, 1); 

            double minVal, maxVal;
            cv::minMaxLoc(bandMat, &minVal, &maxVal);
            bandMat = (bandMat - minVal) / (maxVal - minVal);
            std::cout << "Before normalization: Min = " << minVal << ", Max = " << maxVal << std::endl;

            cv::minMaxLoc(bandMat, &minVal, &maxVal);
            std::cout << "After normalization: Min = " << minVal << ", Max = " << maxVal << std::endl;


            cv::normalize(bandMat, bandMat, 0.0, 1.0, cv::NORM_MINMAX);
            bandMat.convertTo(bandMat, CV_8UC1, 255.0);
            // cout<<"bandMat: "<<bandMat<<endl;

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