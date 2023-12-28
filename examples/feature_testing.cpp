#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include <matio.h>

using namespace cv;
using namespace std;

std::vector<cv::Mat> loadMatlabFile(const std::string& filename)
{
    std::vector<cv::Mat> mats;

    // Open the MATLAB file
    mat_t *matfp = Mat_Open(file_name.c_str(), MAT_ACC_RDONLY);
    if (matfp == NULL) {
        std::cerr << "Error opening MATLAB file " << filename << std::endl;
        return mats;
    }

    // Read each variable in the file
    matvar_t *matvar = NULL;
    while ((matvar = Mat_VarReadNext(matfp)) != NULL) {
        
        // cout<< "here"<<endl;
       
        // int major, minor, release;
        // Mat_GetLibraryVersion(&major, &minor, &release);
        // cout << "MATIO version: " << major<<' '<<minor<<' '<<release << endl;

        // cout << "Variable type: ";
        // switch (matvar->class_type) {
        //     case MAT_C_DOUBLE: cout << "double"; break;
        //     case MAT_C_SINGLE: cout << "single"; break;
        //     case MAT_C_INT32: cout << "int32"; break;
        //     case MAT_C_UINT8: cout << "uint8"; break;
        //     // Add more cases as needed...
        //     default: cout << "unknown";
        // }
        // cout << endl;

        // hyperspectral image is of type double

        // cout << "Variable dimensions: ";
        // for (int i = 0; i < matvar->rank; i++) {
        //     cout << matvar->dims[i];
        //     if (i < matvar->rank - 1) {
        //         cout << " x ";
        //     }
        // }
        // cout << endl;

        // Assuming matvar is your 3D array variable
        // x,y,chanel
        if (matvar->rank == 3) {
            size_t rows = matvar->dims[0];
            size_t cols = matvar->dims[1];
            size_t slices = matvar->dims[2];

            double* data = static_cast<double*>(matvar->data);

            for (size_t i = 0; i < slices; i++) {
                cv::Mat mat(rows, cols, CV_64F, data + i * rows * cols);
                // normalize the data
                cv::normalize(mat, mat, 0, 255, cv::NORM_MINMAX);
                // convert to 8 bit
                mat.convertTo(mat, CV_8U);
                mats.push_back(mat);
                // cv::imshow("mat", mat);
                // cv::waitKey(100);
                // cout<<i<<endl;
            }
        }

        // Free the current MATLAB variable
        Mat_VarFree(matvar);
    }

    // Close the MATLAB file
    Mat_Close(matfp);

    return mats;
}

int main (int argc, char *argv[])
{

  HyperFunctions HyperFunctions1; 
  

  // load hyperspectral image that is a matlab file 
  string file_name3="../../HyperImages/Indian_pines.mat";

  vector<Mat> mats = loadMatlabFile(file_name3);

  // cout<<"mats size: "<<mats.size()<<endl;

  HyperFunctions1.mlt1=mats;
  HyperFunctions1.false_img_b = HyperFunctions1.mlt1.size()/3;
  HyperFunctions1.false_img_g = HyperFunctions1.mlt1.size()*2/3;
  HyperFunctions1.false_img_r = HyperFunctions1.mlt1.size()-1;
  HyperFunctions1.GenerateFalseImg();
  imshow("false img", HyperFunctions1.false_img);
  waitKey();

cout<<"done"<<endl;
  return -1;

  // load hyperspectral image
  string file_name1="../../HyperImages/img1.tiff";
  HyperFunctions1.LoadImageHyper(file_name1);
  // using the same multilayer tiff for testing
  string file_name2="../../HyperImages/img1.tiff";
  HyperFunctions1.LoadImageHyper(file_name1, false);

  // load single layer image
  // string file_name1="../../HyperImages/cornfields/session_002/session_002_490_PANIMAGE.tiff";
  // string file_name2="../../HyperImages/cornfields/session_002/session_002_491_PANIMAGE.tiff";
  // HyperFunctions1.LoadFeatureImage1(file_name1);
  // HyperFunctions1.LoadFeatureImage2(file_name2);

  // use a single image layer
  HyperFunctions1.feature_img1=HyperFunctions1.mlt1[60];
	HyperFunctions1.feature_img2=HyperFunctions1.mlt2[70];

  // use ga space
  HyperFunctions1.dimensionality_reduction = 1;

  //use pca
  //  HyperFunctions1.dimensionality_reduction = 2;

  // use spectral similarity
  // HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
  // HyperFunctions1.spec_sim_alg=0;
  // HyperFunctions1.SpecSimilParent();
  // HyperFunctions1.feature_img1=HyperFunctions1.spec_simil_img;
  // HyperFunctions1.spec_sim_alg=1;
  // HyperFunctions1.SpecSimilParent();
  // HyperFunctions1.feature_img2=HyperFunctions1.spec_simil_img;



  HyperFunctions1.feature_detector=2;
	HyperFunctions1.feature_descriptor=2;
	HyperFunctions1.feature_matcher=0;
  HyperFunctions1.FeatureExtraction();
  



  // Creating and showing integral image:
  // HyperFunctions1.gaSpace(true);
  // HyperFunctions1.ImgIntegration();
  // imshow("Integral Image",HyperFunctions1.integral_img);

  
  
  // stitch images together
  // reuires featureextraction to be run first
  // HyperFunctions1.Stitching();


  cv::waitKey();

    
  return 0;
}
