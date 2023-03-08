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

    /*
    // for already processed image
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
    }  */
    
  // below shows how to process a raw image
  string cubert_settings="../../HyperImages/set1/settings/";  //ultris20.settings file
  string cubert_img="../../HyperImages/cornfields/session_002/session_002_490.cu3";
  string dark_img="../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
  string white_img="../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
  string dist_img="../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";
  string factor_dir="../../HyperImages/cornfields/factory/"; // requires init.daq file
  string output_dir="../../HyperImages/cornfields/results/";
  
	
  
  char* const userSettingsDir =  const_cast<char*>(cubert_settings.c_str());
  char* const measurementLoc =  const_cast<char*>(cubert_img.c_str());
  char* const darkLoc =  const_cast<char*>(dark_img.c_str());
  char* const whiteLoc =  const_cast<char*>(white_img.c_str());
  char* const distanceLoc =  const_cast<char*>(dist_img.c_str());
  char* const factoryDir =  const_cast<char*>(factor_dir.c_str());
  char* const outDir =  const_cast<char*>(output_dir.c_str());
  
  cuvis::General::init(userSettingsDir);
  cuvis::General::set_log_level(loglevel_info);
  cuvis::Measurement mesu(measurementLoc);
  cuvis::Measurement dark(darkLoc);
  cuvis::Measurement white(whiteLoc);
  cuvis::Measurement distance(distanceLoc);
  
  
  /*std::cout << "Data 1:" << mesu.get_meta()->name << " "
            << "t=" << mesu.get_meta()->integration_time << " ms "
            << "mode=" << mesu.get_meta()->processing_mode << " " << std::endl;  
  
  std::cout << "Loading Calibration and processing context (factory)" << std::endl;
  */
  cuvis::Calibration calib(factoryDir);
  cuvis::ProcessingContext proc(calib);

  //std::cout << "Set references" << std::endl;

  proc.set_reference(dark, cuvis::reference_type_t::Reference_Dark);
  proc.set_reference(white, cuvis::reference_type_t::Reference_White);
  proc.set_reference(distance, cuvis::reference_type_t::Reference_Distance);

  cuvis::ProcessingArgs procArgs;
  cuvis::SaveArgs saveArgs;
  saveArgs.allow_overwrite = true;

  std::map<std::string, cuvis::processing_mode_t> target_modes = {
      //{"Raw", cuvis::processing_mode_t::Cube_Raw},
      //{"DS", cuvis::processing_mode_t::Cube_DarkSubtract},
      {"Ref", cuvis::processing_mode_t::Cube_Reflectance}};//, 
      //{"RAD", cuvis::processing_mode_t::Cube_SpectralRadiance}};

  for (auto const& mode : target_modes)
  {
    procArgs.processing_mode = mode.second;
    if (proc.is_capable(mesu, procArgs))
    {
     // std::cout << "processing to mode " << mode.first << std::endl;
      proc.set_processingArgs(procArgs);
      proc.apply(mesu);
      saveArgs.export_dir = std::filesystem::path(outDir) / mode.first;
      mesu.save(saveArgs);
    }
    else
    {
        //std::cout << "cannot process to mode " << mode.first << std::endl;
    }
  }
  std::cout << "finished." << std::endl;
   

	vector<Mat> mlt1;
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


