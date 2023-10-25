#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "../src/hyperfunctions.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

// references  https://docs.opencv.org/3.4/d3/db0/samples_2cpp_2pca_8cpp-example.html
// https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html
/*
static  Mat formatImagesForPCA(const vector<Mat> &data)
{
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_32F);
    }
    return dst;
}

static Mat toGrayscale(InputArray _src) {
    Mat src = _src.getMat();
    // only allow one channel
    if(src.channels() != 1) {
        CV_Error(Error::StsBadArg, "Only Matrices with one channel are supported");
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}*/


int main (int argc, char *argv[])
{
    int reduced_image_layers = 5;
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
