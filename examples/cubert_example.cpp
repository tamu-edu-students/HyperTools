#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hypercuvisfunctions.cpp"
#include "cuvis.hpp"
#include <cassert>

using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{

    HyperFunctionsCuvis HyperFunctions1;

    HyperFunctions1.cubert_img = "../../HyperImages/cornfields/session_002/session_002_490.cu3";
    HyperFunctions1.dark_img = "../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
    HyperFunctions1.white_img = "../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
    HyperFunctions1.dist_img = "../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";

    // HyperFunctions1.cubert_img = "/workspaces/HyperImages/test_imgs/Auto_001.cu3s";
    // HyperFunctions1.cubert_settings="../settings/ultris5";  //camera settings file 
    // HyperFunctions1.factor_dir="/workspaces/HyperTools/settings/ultris5"; // requires init.daq file

    HyperFunctions1.LoadImageHyper(HyperFunctions1.cubert_img);

    //below only works for  ultris20 images due to different number of layers in default spectral database
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    HyperFunctions1.SemanticSegmenter();
    HyperFunctions1.DispClassifiedImage();
    cv::waitKey();


    
  return 0;
}
