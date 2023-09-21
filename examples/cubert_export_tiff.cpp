#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "cuvis.hpp"
#include <cassert>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;


int main (int argc, char *argv[])
{
    /*// the exported image is raw and not reflectance. As a result, the exported image is very dark
    
    string cubert_settings="../../HyperImages/settings/";  //ultris20.settings file
    string cubert_img="../../HyperImages/segmented-datasets/Wextel-Dataset/session_002_492.cu3";//"../../HyperImages/export/Ref/session_002_490.cu3";//cornfields/session_002/session_002_490.cu3";
    string dark_img="../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
    string white_img="../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
    string dist_img="../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";
    string factor_dir="../../HyperImages/cornfields/Calibration/"; // requires init.daq file
    string output_dir="../../HyperImages/export";//cornfields/results/";

    char* const userSettingsDir =  const_cast<char*>(cubert_settings.c_str());
    char* const measurementLoc =  const_cast<char*>(cubert_img.c_str());
    char* const darkLoc =  const_cast<char*>(dark_img.c_str());
    char* const whiteLoc =  const_cast<char*>(white_img.c_str());
    char* const distanceLoc =  const_cast<char*>(dist_img.c_str());
    char* const factoryDir =  const_cast<char*>(factor_dir.c_str());
    char* const exportDir =  const_cast<char*>(output_dir.c_str());

    cuvis::General::init(userSettingsDir);
    cuvis::General::set_log_level(loglevel_info);

    cuvis::Measurement mesu(measurementLoc);

    cuvis::Measurement dark(darkLoc);
    cuvis::Measurement white(whiteLoc);
    cuvis::Measurement distance(distanceLoc);
    cuvis::Calibration calib(factoryDir);
    cuvis::ProcessingContext proc(calib);

    // set references 
    proc.set_reference(dark, cuvis::reference_type_t::Reference_Dark);
    proc.set_reference(white, cuvis::reference_type_t::Reference_White);
    proc.set_reference(distance, cuvis::reference_type_t::Reference_Distance);

    cuvis::ProcessingArgs procArgs;
    cuvis::SaveArgs saveArgs;
    saveArgs.allow_overwrite = true;
    
  
    std::map<std::string, cuvis::processing_mode_t> target_modes = {{"Ref", cuvis::processing_mode_t::Cube_Reflectance}};


    for (auto const& mode : target_modes)
    {
        procArgs.processing_mode = mode.second;
        if (proc.is_capable(mesu, procArgs))
        {
          proc.set_processingArgs(procArgs);
          proc.apply(mesu);
          //saveArgs.export_dir = std::filesystem::path(exportDir) / mode.first;
          //mesu.save(saveArgs);
        }
    }

   */
   
   // below is if image is already processed
       // function for .cu3 file
    string cubert_img="../../HyperImages/2023_04_13_Relay1/session_000/session_000_007_snapshot.cu3";
        string output_dir="../../HyperImages/export";//cornfields/results/";
    // calibration files
    string cubert_settings="../../HyperImages/settings/";  //ultris20.settings file
    char* const measurementLoc =  const_cast<char*>(cubert_img.c_str());
    char* const userSettingsDir =  const_cast<char*>(cubert_settings.c_str());
      char* const exportDir =  const_cast<char*>(output_dir.c_str());
  
    cuvis::Measurement mesu(measurementLoc);
  cuvis::ProcessingContext proc(mesu);

     


// below is for visualization 
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


// below is for export 
 assert(mesu.get_meta()->processing_mode != cuvis::processing_mode_t::Preview);
    {
        std::cout << "Export to Multi-Channel Tiff" << std::endl;
        cuvis::TiffArgs args;
        char exportDirMulti[CUVIS_MAXBUF];
        strcpy(exportDirMulti, exportDir);
        strcat(exportDirMulti, "/multi");
        args.export_dir = exportDirMulti;
        args.format = cuvis::tiff_format_t::tiff_format_MultiPage;
        cuvis::TiffExporter exporter(args);
        exporter.apply(mesu);
    }

    std::cout << "finished." << std::endl;
    
    

  return 0;
}

