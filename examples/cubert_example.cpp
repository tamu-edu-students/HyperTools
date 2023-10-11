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

    HyperFunctions1.cubert_img = "/workspaces/HyperImages/export/Auto_001.cu3s";
    HyperFunctions1.cubert_settings="../settings/ultris5";  //camera settings file 
    HyperFunctions1.factor_dir="/workspaces/HyperTools/settings/ultris5"; // requires init.daq file

    
    // string file_name, const int exposure_ms, const int num_image
    // HyperFunctions1.TakeImageHyper1("qferg",20, 1);
    
    
    //below only works for  ultris20 images due to different number of layers in default spectral database
    // HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    // HyperFunctions1.SemanticSegmenter();
    // HyperFunctions1.DispClassifiedImage();


    HyperFunctions1.LoadImageHyper(HyperFunctions1.cubert_img);
    HyperFunctions1.false_img_b=10;
    HyperFunctions1.false_img_g=20;
    HyperFunctions1.false_img_r=30;
    HyperFunctions1.GenerateFalseImg();
    HyperFunctions1.DispFalseImage();

    cv::waitKey();


    
  return 0;
}
