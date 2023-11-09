#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"

using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{

  HyperFunctions HyperFunctions1; 
  // string file_name1="../images/session_002_490_PANIMAGE.tiff";
  // // HyperFunctions1.feature_img1 = imread(file_name1);
  // HyperFunctions1.LoadImageHyper(file_name1);
  // // using the same multilayer tiff for testing
  // string file_name2="../images/session_002_491_PANIMAGE.tiff";
  // //HyperFunctions1.feature_img2 = imread(file_name2);
  // HyperFunctions1.LoadImageHyper(file_name1, false);

  string file_name1="../images/img_1.png";
  HyperFunctions1.feature_img1 = imread(file_name1);
  string file_name2="../images/img2.png";
  HyperFunctions1.feature_img2 = imread(file_name2);
  // imshow("show", HyperFunctions1.feature_img1);
  // imshow("show2", HyperFunctions1.feature_img2);
  // string file_name2="../../HyperImages/img2.tiff";
  // HyperFunctions1.LoadImageHyper(file_name2, false);


  // use a single image layer
  // HyperFunctions1.feature_img1=HyperFunctions1.mlt1[60];
	// HyperFunctions1.feature_img2=HyperFunctions1.mlt2[70];

  // use ga space
    // HyperFunctions1.dimensionality_reduction = 1;

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


  cv::resize(HyperFunctions1.feature_img1,HyperFunctions1.feature_img1,Size(800, 800),INTER_LINEAR);
  cv::resize(HyperFunctions1.feature_img2,HyperFunctions1.feature_img2,Size(800, 800),INTER_LINEAR);
  HyperFunctions1.feature_detector=2;
	HyperFunctions1.feature_descriptor=2;
	HyperFunctions1.feature_matcher=0;
  HyperFunctions1.FeatureExtraction();
  HyperFunctions1.Stitching();


  cv::waitKey();

    
  return 0;
}
