#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "../src/hyperfunctions.cpp"
#include "../src/hypergpufunctions.cu"
#include <cassert>
#include "cuvis.h"
using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{

    string cubert_settings="../../HyperImages/set1/settings/";
    string cubert_img="../../HyperImages/set1/vegetation_000/vegetation_000_000_snapshot.cu3";

    vector<Mat> mlt1;

    char* const userSettingsDir = const_cast<char*>(cubert_settings.c_str());  
    char* const measurementLoc = const_cast<char*>(cubert_img.c_str());  

    CUVIS_MESU mesu1;
    CUVIS_CHECK(cuvis_init(userSettingsDir));
    CUVIS_CHECK(cuvis_set_log_level(loglevel_info));
    CUVIS_CHECK(cuvis_measurement_load(measurementLoc,&mesu1));
    CUVIS_MESU_METADATA mesu_data;
    CUVIS_CHECK(cuvis_measurement_get_metadata(mesu1, &mesu_data));
    assert(
      mesu_data.processing_mode == Cube_Raw &&
      "This example requires raw mode");

    CUVIS_IMBUFFER cube;
    CUVIS_CHECK(cuvis_measurement_get_data_image(mesu1, CUVIS_MESU_CUBE_KEY, &cube));
    CUVIS_IMBUFFER iminfo;
    cuvis_measurement_get_data_image(mesu1, CUVIS_MESU_CUBE_INFO_KEY, &iminfo);

    cv::Mat img(
    cv::Size(cube.width, cube.height),
    CV_16UC(cube.channels),
    const_cast<void*>(reinterpret_cast<const void*>(cube.raw)),
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

    cuvis_measurement_free(&mesu1);
  
    HyperFunctionsGPU HyperFunctions1;
    HyperFunctions1.mlt1=mlt1;
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);   
    int* test_array= HyperFunctions1.mat_to_oneD_array_parallel_parent(  );
    HyperFunctions1.ref_spec_index=1;
    HyperFunctions1.allocate_memory(test_array);
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();


  return 0;
}


