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
    string input_file_path = "../../../HyperImages/hyperspectral_images/Indian_pines.tiff";
    string reduced_file_path = "../../../HyperImages/hyperspectral_images/dimension_reduced.tiff";
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
    /*
    // Combine all input images into a single matrix
    Mat combined_data;
    vconcat(inputImage, combined_data);

    // Run PCA on the combined data
    PCA pca(combined_data, noArray(), PCA::DATA_AS_ROW);

    // Keep only the first 9 principal components
    Mat principal_components = pca.eigenvectors.rowRange(0, 9);

    // Project the input data onto the reduced basis
    Mat projected_data = combined_data * principal_components.t();

    // Reshape the projected data into a set of 9 output images
    vector<Mat> reduced_images;
    for (int i = 0; i < 9; i++) {
        Mat reduced_layer = projected_data.col(i).clone().reshape(1, inputImage[0].rows);
        reduced_images.push_back(reduced_layer);
    }
*/
    // ... code to load input images ...
/*
    // Convert images to 2D matrices
    vector<Mat> reshapedImage;
    for (int i = 0; i < inputImage.size(); i++) {
        Mat img = inputImage[i];
        Mat img_2d = img.reshape(1, inputImage[0].total());
        reshapedImage.push_back(img_2d);
    }

    Mat stackedImage;
    vconcat(reshapedImage, stackedImage);

    // Perform PCA on input data
    PCA pca(stackedImage, Mat(), PCA::DATA_AS_ROW, reduced_image_layers);

    // Transform input data into reduced dimensionality space
    Mat reducedImage = pca.project(stackedImage);

    Mat reshapedReducedImage = reducedImage.reshape(reduced_image_layers, stackedImage.cols);

    // Reshape reduced data back into images
    vector<Mat> output_image;
    for (int i = 0; i < reduced_image_layers; i++) {
        Mat layer = reshapedReducedImage.row(i).reshape(1, inputImage[0].rows);
        output_image.push_back(layer);
    }
    */

    //Combine the data from all layers into a single matrix
    //Necessary step for the SVD/PCA stuff
    //Each row is one of the original layers
    Mat combined_data(inputImage[0].rows * inputImage[0].cols, inputImage.size(), CV_32F);
    for (int i = 0; i < inputImage.size(); i++) { //For each layer in the original tiff
        Mat layer_float;
        inputImage[i].convertTo(layer_float, CV_32F);
        Mat layer_reshaped = layer_float.reshape(0, layer_float.rows * layer_float.cols);
        layer_reshaped.copyTo(combined_data.col(i));
    }

    //Perform PCA on the combined data using opencv's built in PCA. 
    Mat importantLayersCombined;
    //PCA::PCA(combined_data, noArray(), PCA::DATA_AS_ROW, reduced_image_layers, principal_components);
    PCA pca(combined_data, noArray(), PCA::DATA_AS_ROW, reduced_image_layers);
    importantLayersCombined = pca.project(combined_data);
    
    // Perform inverse PCA transformation to reconstruct the original data
    Mat reconstructed_data = pca.backProject(importantLayersCombined);

    // Reshape the reconstructed data into a vector of individual image layers
    vector<Mat> important_layers_reconstructed;
    for (int i = 0; i < reconstructed_data.rows; i++) {
        Mat layer = reconstructed_data.row(i);
        Mat layerAsImage = layer.reshape(0, inputImage[0].rows);
        important_layers_reconstructed.push_back(layer);
    }

/*
    //Taking each final layer out of the projected pca
    Mat finalImage[reduced_image_layers];
    for (int i = 0; i < reduced_image_layers; i++) {
        Mat l1 = importantLayersCombined.row(i).clone();
        finalImage[i] = l1.reshape(1, ())
    }
    */




    //PCA pca_analysis(combined_data, cv::, PCA::DATA_AS_ROW, reduced_image_layers, principal_components);
    /*
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
*/
    //Mat reduced_image_array[] = Mat(reduced_image);

    //Save the reduced image as a multilayered TIFF file
    //imwritemulti(reduced_file_path, reduced_image_layers);
    //cv::imwritemulti(reduced_file_path, reduced_image, { CV_IMWRITE_TIFF_COMPRESSION, 1 });

    return 0;

}
