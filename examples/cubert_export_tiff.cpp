#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "cuvis.hpp"
#include <cassert>
#include <opencv2/imgcodecs.hpp>
#include "../src/hypercuvisfunctions.cpp"


using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{
    // the exported image is raw and not reflectance. As a result, the exported image is very dark
    HyperFunctionsCuvis HyperFunctions1;

    HyperFunctions1.cubert_img = "/workspaces/HyperImages/test_imgs/Auto_001.cu3s";
    HyperFunctions1.cubert_settings="../settings/ultris5";  //camera settings file 
    HyperFunctions1.factor_dir="/workspaces/HyperTools/settings/ultris5"; // requires init.daq file

    HyperFunctions1.ExportTiff();
    
    std::cout << "finished." << std::endl;
    
    

  return 0;
}

