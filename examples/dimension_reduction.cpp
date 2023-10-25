#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "../src/hyperfunctions.h"

using namespace cv;
using namespace std;
using namespace std::chrono;



int main (int argc, char *argv[])
{
    string input_file_path = "../../HyperImages/img1.tiff";
    string reduced_file_path = "../../HyperImages/dimension_reduced.tiff";
     
    HyperFunctions HyperFunctions1;
    // load hyperspectral image 
    HyperFunctions1.LoadImageHyper(input_file_path);
    HyperFunctions1.PCA_img();
    imshow("test",HyperFunctions1.pca_img);
    cv::waitKey();

    return 0;

}
