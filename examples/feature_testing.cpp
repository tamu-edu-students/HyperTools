#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"

using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{

  HyperFunctions HyperFunctions1; 
  string file_name1="../../HyperImages/img1.tiff";
  HyperFunctions1.LoadImageHyper(file_name1);
  string file_name2="../../HyperImages/img2.tiff";
  HyperFunctions1.LoadImageHyper(file_name2, false);


  // use a single image layer
  HyperFunctions1.feature_img1=HyperFunctions1.mlt1[60];
	HyperFunctions1.feature_img2=HyperFunctions1.mlt2[70];

  //use pca
  // HyperFunctions1.PCA_img();
  // HyperFunctions1.feature_img1=HyperFunctions1.pca_img;
  // HyperFunctions1.PCA_img(false);
  // HyperFunctions1.feature_img2=HyperFunctions1.pca_img;

  // use spectral similarity
  // HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
  // HyperFunctions1.spec_sim_alg=0;
  // HyperFunctions1.SpecSimilParent();
  // // HyperFunctions1.DispSpecSim();
  // HyperFunctions1.feature_img1=HyperFunctions1.spec_simil_img;
  // HyperFunctions1.spec_sim_alg=1;
  // HyperFunctions1.SpecSimilParent();
  // HyperFunctions1.feature_img2=HyperFunctions1.spec_simil_img;




  HyperFunctions1.feature_detector=2;
	HyperFunctions1.feature_descriptor=2;
	HyperFunctions1.feature_matcher=0;

  HyperFunctions1.FeatureExtraction();


  cv::waitKey();

    
  return 0;
}
