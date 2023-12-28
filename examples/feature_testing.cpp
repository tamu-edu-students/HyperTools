#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"


using namespace cv;
using namespace std;

std::vector<cv::Mat> loadMatlabFile(const std::string& filename)
{
    std::vector<cv::Mat> mats;

    // Open the MATLAB file
    mat_t *matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (matfp == NULL) {
        std::cerr << "Error opening MATLAB file " << filename << std::endl;
        return mats;
    }

    // Read each variable in the file
    matvar_t *matvar = NULL;
    while ((matvar = Mat_VarReadNext(matfp)) != NULL) {
        // Convert the variable to a cv::Mat and add it to the vector
        // This assumes that the variable is a 2D double matrix
        if (matvar->rank == 2 && matvar->data_type == MAT_T_DOUBLE) {
            cv::Mat mat(matvar->dims[0], matvar->dims[1], CV_64F, matvar->data);
            mats.push_back(mat.clone());  // clone the mat because matvar->data will be freed
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
  string file_name1="../../HyperImages/Indian_pines.mat";

  vector<Mat> mats = loadMatlabFile(file_name1);

//   sudo apt install libhdf5-dev libtool m4 automake
// # Clone the matio repository
// git clone git://git.code.sf.net/p/matio/matio

// # Navigate into the cloned directory
// cd matio

// # Update submodules (for datasets used in unit tests)
// git submodule update --init

// # Generate the configure script
// ./autogen.sh

// # Configure the build
// ./configure

// # Build the library
// make

// # Run tests (optional)
// make check

// # Install the library
// sudo make install

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
