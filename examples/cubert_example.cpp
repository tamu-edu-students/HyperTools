#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "../src/hyperfunctions.cpp"
#include "cuvis.hpp"
#include <cassert>
// #include "cuvis.h"
using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{

    string cubert_settings="../../HyperImages/set1/settings/";
	string cubert_img="../../HyperImages/set1/vegetation_000/vegetation_000_000_snapshot.cu3";
	
	vector<Mat> mlt1;
	
    char* const userSettingsDir = const_cast<char*>(cubert_settings.c_str());  
    char* const measurementLoc = const_cast<char*>(cubert_img.c_str());  
    cuvis::General::init(userSettingsDir);
    cuvis::General::set_log_level(loglevel_info);
    cuvis::Measurement mesu(measurementLoc);
 if (mesu.get_meta()->measurement_flags.size() > 0)
    {
        std::cout << "  Flags" << std::endl;
        for (auto const& flags : mesu.get_meta()->measurement_flags)
        {
            std::cout << "  - " << flags.first << " (" << flags.second << ")" << std::endl;
        }
    }
    assert(
        mesu.get_meta()->processing_mode == Cube_Raw &&
        "This example requires raw mode");


    auto const& cube_it = mesu.get_imdata()->find(CUVIS_MESU_CUBE_KEY);
    assert(
        cube_it != mesu.get_imdata()->end() &&
        "Cube not found");
    auto cube = std::get<cuvis::image_t<std::uint16_t>>(cube_it->second);
 cv::Mat img(
    cv::Size(cube._width, cube._height),
    CV_16UC(cube._channels),
    const_cast<void*>(reinterpret_cast<const void*>(cube._data)),
    cv::Mat::AUTO_STEP);

    for (int i=0;i<img.channels();i++)
    {
        cv::Mat singleChannel;
        cv::extractChannel(
        img, singleChannel, i); 
        singleChannel.convertTo(singleChannel, CV_8U, 1 / 16.0);
        mlt1.push_back(singleChannel);
        cv::imshow(" Individual channel ", singleChannel);
        cv::waitKey(50);
    }

  return 0;
}


