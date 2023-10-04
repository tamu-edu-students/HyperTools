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
    // take image from camera
    // exposure = 20ms, num images = 1
    HyperFunctions1.TakeImageHyper1("blah", 20, 1);

    HyperFunctions1.cubert_img = "/workspaces/HyperImages/export/Auto_001.cu3s";
    HyperFunctions1.cubert_settings="../settings/ultris5";  //camera settings file 
    HyperFunctions1.factor_dir="../settings/ultris5"; // requires init.daq file
    HyperFunctions1.spectral_database = "test"; // replace with ultris5 database
    HyperFunctions1.LoadImageHyper1(HyperFunctions1.cubert_img);

    //go through and display images
    Mat singleChannel; 
    for (size_t i = 0; i < HyperFunctions1.mlt1.size(); i++)
    {
        singleChannel = HyperFunctions1.mlt1[i];
        cv::imshow("Individual channel: ", singleChannel);
        cv::waitKey(50);  
    }

    //load spectral database
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    HyperFunctions1.SemanticSegmenter();
    HyperFunctions1.DispClassifiedImage();
    //show specsimilarity image
}

