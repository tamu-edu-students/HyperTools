#pragma once
#include "hyperfunctions.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <stdio.h>
#include "ctpl.h"
#include "opencv2/xfeatures2d.hpp"
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


void filter_matches(vector<Dmatch> matches)
{
    vector<Dmatch> good_matches;
    for(auto i = 0; i<matches.size();i++)
    {
        if(matches.at(i).distance > .75)
        {
            good_matches.push_back(matches.at(i));
        }
    }
}