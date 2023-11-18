#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "../src/hypercuvisfunctions.cpp"
#include "cuvis.hpp"
using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{

  HyperFunctionsCuvis HyperFunctions1; 
  
  
  // load hyperspectral image
  string file_name1="../../HyperImages/img1.tiff";
  HyperFunctions1.LoadImageHyper(file_name1);


    // // below is for ultris 5 example
    // HyperFunctions1.cubert_img = "../../HyperImages/export/Test_001.cu3s";
    // HyperFunctions1.dark_img = "../../HyperImages/Calib100/dark.cu3s";
    // HyperFunctions1.white_img = "../../HyperImages/Calib100/white.cu3s";
    // HyperFunctions1.dist_img = "../../HyperImages/Calib100/distance.cu3s";
    
    // // below are needed if the ultris5 is used instead of the ultris 20
    // HyperFunctions1.cubert_settings="../settings/ultris5";  //camera settings file 
    // HyperFunctions1.factor_dir="../settings/ultris5"; // requires init.daq file
    // HyperFunctions1.ReprocessImage( HyperFunctions1.cubert_img);  




  HyperFunctions1.NWHFC_img();


  // using the same multilayer tiff for testing
  // string file_name2="../../HyperImages/img1.tiff";
  // HyperFunctions1.LoadImageHyper(file_name1, false);

  // load single layer image
  // string file_name1="../../HyperImages/cornfields/session_002/session_002_490_PANIMAGE.tiff";
  // string file_name2="../../HyperImages/cornfields/session_002/session_002_491_PANIMAGE.tiff";
  // HyperFunctions1.LoadFeatureImage1(file_name1);
  // HyperFunctions1.LoadFeatureImage2(file_name2);

  // use a single image layer
  // HyperFunctions1.feature_img1=HyperFunctions1.mlt1[60];
	// HyperFunctions1.feature_img2=HyperFunctions1.mlt2[70];

  // use ga space
  // HyperFunctions1.dimensionality_reduction = 1;

  //use pca
  //  HyperFunctions1.dimensionality_reduction = 2;

  // use mnf (needs to be added to dimensional reduction)
  // HyperFunctions1.dimensionality_reduction = 3;


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
  // HyperFunctions1.FeatureExtraction();
  
  // stitch images together
  // HyperFunctions1.Stitching();

  // dimensionality reduction techniques
  // HyperFunctions1.PCA_img();
  // HyperFunctions1.MNF_img();


  cv::waitKey();

    
  return 0;
}
