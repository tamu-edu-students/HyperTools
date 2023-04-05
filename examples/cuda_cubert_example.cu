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

    string cubert_settings="../../HyperImages/settings/";  //ultris20.settings file
    string cubert_img="../../HyperImages/cornfields/session_002/session_002_490.cu3";
    string dark_img="../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
    string white_img="../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
    string dist_img="../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";
    string factor_dir="../../HyperImages/cornfields/Calibration/"; // requires init.daq file
    string output_dir="../../HyperImages/cornfields/results/";


    char* const userSettingsDir =  const_cast<char*>(cubert_settings.c_str());
    char* const measurementLoc =  const_cast<char*>(cubert_img.c_str());
    char* const darkLoc =  const_cast<char*>(dark_img.c_str());
    char* const whiteLoc =  const_cast<char*>(white_img.c_str());
    char* const distanceLoc =  const_cast<char*>(dist_img.c_str());
    char* const factoryDir =  const_cast<char*>(factor_dir.c_str());
    char* const outDir =  const_cast<char*>(output_dir.c_str());
    
    CUVIS_MESU mesu;
    CUVIS_MESU_METADATA mesu_data;
    CUVIS_MESU dark;
    CUVIS_MESU white;
    CUVIS_MESU distance;
    CUVIS_CALIB calib;
    CUVIS_PROC_CONT procCont;
    CUVIS_INT is_capable;
    CUVIS_SAVE_ARGS save_args;
    CUVIS_CHECK(cuvis_init(userSettingsDir));
    CUVIS_CHECK(cuvis_measurement_load(measurementLoc,
      &mesu));
    CUVIS_CHECK(cuvis_measurement_load(darkLoc, &dark));
    CUVIS_CHECK(cuvis_measurement_load(whiteLoc, &white));
    CUVIS_CHECK(cuvis_measurement_load(distanceLoc,
      &distance));
    CUVIS_CHECK(cuvis_measurement_get_metadata(mesu, &mesu_data));
    CUVIS_CHECK(
      cuvis_calib_create_from_path(factoryDir, &calib));
    CUVIS_CHECK(cuvis_proc_cont_create_from_calib(calib, &procCont));
    CUVIS_CHECK(cuvis_proc_cont_set_reference(procCont, dark, Reference_Dark));
    CUVIS_CHECK(
      cuvis_proc_cont_set_reference(procCont, white, Reference_White));
    CUVIS_CHECK(
      cuvis_proc_cont_set_reference(procCont, distance, Reference_Distance));
    CUVIS_PROC_ARGS args;
    args.processing_mode = Cube_Raw;
    args.allow_recalib = 0;
    CUVIS_CHECK(cuvis_proc_cont_is_capable(procCont, mesu, args, &is_capable));
    save_args.allow_fragmentation = 0;
    save_args.allow_overwrite = 1;
    args.processing_mode = Cube_Reflectance;
    CUVIS_CHECK(cuvis_proc_cont_is_capable(procCont, mesu, args, &is_capable));

    if (1 == is_capable)
    {
        printf("reprocess measurement to Cube_Reflectance mode...\n");
        fflush(stdout);

        CUVIS_CHECK(cuvis_proc_cont_set_args(procCont, args));
        CUVIS_CHECK(cuvis_proc_cont_apply(procCont, mesu));
        CUVIS_CHECK(cuvis_measurement_get_metadata(mesu, &mesu_data));
        char exportDirREF[CUVIS_MAXBUF];
        strcpy(exportDirREF, outDir);
        strcat(exportDirREF, "/REF");
        CUVIS_CHECK(cuvis_measurement_save(mesu, exportDirREF, save_args));
    }

    CUVIS_IMBUFFER cube;
    CUVIS_CHECK(cuvis_measurement_get_data_image(mesu, CUVIS_MESU_CUBE_KEY, &cube));
    CUVIS_IMBUFFER iminfo;
    cuvis_measurement_get_data_image(mesu, CUVIS_MESU_CUBE_INFO_KEY, &iminfo);
    cv::Mat img(
    cv::Size(cube.width, cube.height),
    CV_16UC(cube.channels),
    const_cast<void*>(reinterpret_cast<const void*>(cube.raw)),
    cv::Mat::AUTO_STEP);
    vector<Mat> mlt1;

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
    cuvis_calib_free(&calib);
    cuvis_proc_cont_free(&procCont);
    cuvis_measurement_free(&mesu);
    cuvis_measurement_free(&dark);
    cuvis_measurement_free(&white);
    cuvis_measurement_free(&distance); 

    HyperFunctionsGPU HyperFunctions1;
    HyperFunctions1.mlt1=mlt1;
    auto start = high_resolution_clock::now();
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);  
    HyperFunctions1.mat_to_oneD_array_parallel_parent(); 
    HyperFunctions1.allocate_memory();
    HyperFunctions1.spec_sim_GPU();
    auto end = high_resolution_clock::now();
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
   
    cv::waitKey();

  return 0;
}


