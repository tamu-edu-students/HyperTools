#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"


using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{

  HyperFunctions HyperFunctions1; 
  
  
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
