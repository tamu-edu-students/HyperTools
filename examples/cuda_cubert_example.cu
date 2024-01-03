#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "../src/hypergpufunctions.cu"
#include <cassert>
#include "cuvis.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

int main (int argc, char *argv[])
{

    string cubert_settings="../settings/";  //ultris20.settings file
    string cubert_img="../../HyperImages/cornfields/session_002/session_002_490.cu3";
    string dark_img="../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
    string white_img="../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
    string dist_img="../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";
    string factor_dir="../../HyperImages/cornfields/Calibration/"; // requires init.daq file
    string output_dir="../../HyperImages/export/";


  return 0;
}


