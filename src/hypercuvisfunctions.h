#if !defined(HYPERCUVISFUNCTIONS_H)
#define HYPERCUVISFUNCTIONS_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include "hyperfunctions.h"
#include "cuvis.hpp"
#include <cassert>
#include <cmath>


using namespace std;
using namespace cv;

class HyperFunctionsCuvis : public HyperFunctions {
public:
    string cubert_settings="../settings/ultris20";  //camera settings file 
    string factor_dir="../settings/ultris20"; // requires init.daq file
    string output_dir="../../HyperImages/export/";
    string cubert_img;
    string dark_img;
    string white_img;
    string dist_img;

    void LoadImageHyper1(string file_name);
    void LoadImageHyper2(string file_name);
    void ExportTiff();


};


#endif
