#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "hyperfunctions.h"
#include "hypercuvisfunctions.h"
#include <fstream>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <stdio.h>
#include "ctpl.h"
#include "opencv2/xfeatures2d.hpp"

int main (int argc, char *argv[])
{

    HyperFunctionsCuvis HyperFunctions1;
    string testImg = "/workspaces/HyperTools/images/lena3.png";
    //take image from camera
    HyperFunctions1.TakeImageHyper1();
    // HyperFunctions1.takeimage


    HyperFunctions1.cubert_img = "/workspaces/HyperImages/export/Auto_001.cu3s";
    HyperFunctions1.cubert_settings="../settings/ultris5";  //camera settings file 
    HyperFunctions1.factor_dir="../settings/ultris5"; // requires init.daq file
    HyperFunctions1.LoadImageHyper1(HyperFunctions1.cubert_img);

    //go through and display images

    Mat singleChannel
    for (size_t i = 0; i < size of mlt1; i++)
    {
        singleChannel=mlt1[i];
        // cv::imshow(" Individual channel ", singleChannel);
        // cv::waitKey(50);    }
    }
//load spectral database

//show specsimilarity image


}

