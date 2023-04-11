#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "../src/hyperfunctions.cpp"
using namespace cv;
using namespace std;
using namespace std::chrono;

int main (int argc, char *argv[])
{
    int reduced_image_layers = 3;
    string input_file_path = "../../HyperImages/hyperspectral_images/Indian_pines.tiff";
    string reduced_file_path = "../../HyperImages/hyperspectral_images/dimension_reduced.tiff";
    if (argc > 1) {
        reduced_image_layers = stoi(argv[1]);
    }
    if (argc > 2) {
        input_file_path = argv[2]; //Input file name is set to second command line argument
    }
    if (argc > 3) {
        reduced_file_path = argv[3]; //Output file name is set to third command line argument
    }

    HyperFunctions HyperFunctions1;
    // load hyperspectral image 
    HyperFunctions1.LoadImageHyper1(input_file_path);

    vector<Mat> inputImage = HyperFunctions1.mlt1;

    //Combine the data from all layers into a single matrix
    //Necessary step for the SVD/PCA stuff
    Mat combined_data(inputImage[0].rows * inputImage[0].cols, inputImage.size(), CV_32F);
    for (int i = 0; i < inputImage.size(); i++) { //For each layer in the original tiff
        Mat layer_float;
        inputImage[i].convertTo(layer_float, CV_32F);
        Mat layer_reshaped = layer_float.reshape(1, layer_float.rows * layer_float.cols);
        layer_reshaped.copyTo(combined_data.col(i));
    }

    //Perform PCA on the combined data using opencv's built in PCA. 
    Mat principal_components;
    PCA(combined_data, noArray(), 0, reduced_image_layers, principal_components);

    //Project each layer onto the principal component basis to obtain a reduced set of layers
    cv::Mat reduced_image[reduced_image_layers];
    for (int i = 0; i < inputImage.size(); i++) {
        Mat layer_float;
        inputImage[i].convertTo(layer_float, CV_32F);
        Mat layer_reshaped = layer_float.reshape(1, layer_float.rows * layer_float.cols);
        Mat projected_layer = layer_reshaped * principal_components.t();
        Mat reduced_layer = projected_layer.colRange(0, reduced_image_layers); //This line crops the image to the most important layers
        reduced_layer = reduced_layer.reshape(1, layer_float.rows);
        reduced_layer.convertTo(reduced_image[i], inputImage[i].type());
    }

    //Mat reduced_image_array[] = Mat(reduced_image);

    //Save the reduced image as a multilayered TIFF file
    imwritemulti(reduced_file_path, reduced_image_array);
    //cv::imwritemulti(reduced_file_path, reduced_image, { CV_IMWRITE_TIFF_COMPRESSION, 1 });

    return 0;
}
