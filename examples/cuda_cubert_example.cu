#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hypercuvisfunctions.cpp"
#include <cassert>
#include "cuvis.h"
#include "cuvis.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

int main (int argc, char *argv[])
{

  HyperFunctionsCuvis HyperFunctions1;
  
  HyperFunctions1.cubert_img = "../../HyperImages/cornfields/session_002/session_002_490.cu3";
  HyperFunctions1.dark_img = "../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
  HyperFunctions1.white_img = "../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
  HyperFunctions1.dist_img = "../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";


  HyperFunctions1.ReprocessImage( HyperFunctions1.cubert_img);  


  // show false rgb image
  // HyperFunctions1.false_img_b=2;
  // HyperFunctions1.false_img_g=13;
  // HyperFunctions1.false_img_r=31;
  // HyperFunctions1.GenerateFalseImg();
  // imshow("test",  HyperFunctions1.false_img);
  // cv::waitKey();

  // load spectral database and move image to gpu
  HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
  HyperFunctions1.mat_to_oneD_array_parallel_parent();
  HyperFunctions1.allocate_memory();

  // perform classification
  HyperFunctions1.spec_sim_alg=2;
  HyperFunctions1.semantic_segmentation(); 
  HyperFunctions1.DispClassifiedImage();
  cv::waitKey();


  HyperFunctions1.deallocate_memory();

  return 0;
}


