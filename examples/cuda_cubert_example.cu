#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hypercuvisfunctions.cpp"
#include <cassert>
#include "cuvis.h"
#include "cuvis.hpp"
#include <vector>
#include <algorithm>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main (int argc, char *argv[])
{

    HyperFunctionsCuvis HyperFunctions1;

    bool take_images = true;
    int exposure_ms = 200;
    string file_name = "test";
    int num_images = 3;
    // below is for ultris 5 example
    // // below are needed if the ultris5 is used instead of the ultris 20
    HyperFunctions1.cubert_settings="../settings/ultris5";  //camera settings file 
    HyperFunctions1.factory_dir="../settings/ultris5"; // requires init.daq file

  if (take_images)
  {
        // Take image using cuvis and put it in the export directory

        char* const userSettingsDir = const_cast<char*>(HyperFunctions1.cubert_settings.c_str());
        char* const factoryDir = const_cast<char*>(HyperFunctions1.factory_dir.c_str());
        char* const recDir = const_cast<char*>(HyperFunctions1.output_dir.c_str());

        // Loading user settings
        cuvis::General::init(userSettingsDir);
        // cuvis::General::set_log_level(loglevel_info);

        // std::cout << "Loading Calibration and processing context..." << std::endl;
        // Loading calibration and processing context
        cuvis::Calibration calib(factoryDir);
        cuvis::ProcessingContext proc(calib);
        cuvis::AcquisitionContext acq(calib);

        cuvis::SaveArgs saveArgs;
        saveArgs.allow_overwrite = true;
        saveArgs.export_dir = recDir;
        saveArgs.allow_session_file = true;

        cuvis::CubeExporter exporter(saveArgs);

        while (cuvis::hardware_state_t::hardware_state_offline == acq.get_state())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "Camera is online" << std::endl;
        acq.set_operation_mode(cuvis::operation_mode_t::OperationMode_Software).get();
        acq.set_integration_time(exposure_ms).get();
        
        auto session = cuvis::SessionInfo();
        session.name = file_name; // this is the name of the session and the base, the images will be named as base_001.cu3s
        acq.set_session_info(session);
    
        std::cout << "Start recording now" << std::endl;
        for (int k = 0; k < num_images; k++)
        {
            std::cout << "Record image #" << k << "... (async) ";
            auto async_mesu = acq.capture();
            auto mesu_res = async_mesu.get(std::chrono::milliseconds(500));
            if (mesu_res.first == cuvis::async_result_t::done &&
                mesu_res.second.has_value())
            {
                auto& mesu = mesu_res.second.value();

                // sets single image name and saves as a .cu3
                // mesu.set_name("Test");
                // mesu.save(saveArgs);

                proc.apply(mesu);
                // exporter.apply(mesu);


                auto const& cube_it = mesu.get_imdata()->find(CUVIS_MESU_CUBE_KEY);
                assert(cube_it != mesu.get_imdata()->end() && "Cube not found");

                auto cube = std::get<cuvis::image_t<std::uint16_t>>(cube_it->second);

                cv::Mat img(
                cv::Size(cube._width, cube._height),
                CV_16UC(cube._channels),
                const_cast<void*>(reinterpret_cast<const void*>(cube._data)),
                cv::Mat::AUTO_STEP);
                HyperFunctions1.mlt1.clear();
                for (int i=0;i<img.channels();i++)
                {
                    cv::Mat singleChannel;
                    cv::extractChannel(
                    img, singleChannel, i); 
                    singleChannel.convertTo(singleChannel, CV_8U, 1 / 16.0);
                    HyperFunctions1.mlt1.push_back(singleChannel);
                    // cv::imshow(" Individual channel ", singleChannel);
                    // cv::waitKey(50);
                }
                HyperFunctions1.false_img_b=2;
                HyperFunctions1.false_img_g=13;
                HyperFunctions1.false_img_r=31;
                HyperFunctions1.GenerateFalseImg();
                imshow("test",  HyperFunctions1.false_img);
                cv::waitKey();

                std::cout << "done" << std::endl;
            }
            else
            {
            std::cout << "failed" << std::endl;
            }
        }

        //uncomment for recording in queue mode
        /* 
        for (int k = 0; k < num_images; k++)
        {
            std::cout << "Record image #" << k << "... (queue)";
            acq.capture_queue();
            auto mesu = acq.get_next_measurement(std::chrono::milliseconds(500));
            if (mesu)
            {
                proc.apply(mesu.value());
                exporter.apply(mesu.value());
                std::cout << "done" << std::endl;
            }
            else
            {
                std::cout << "failed" << std::endl;
            }
            std::cout << "done" << std::endl;
        }
        */
        std::cout << "finished." << std::endl;
    
  }



  else // if not taking images, then process images in a directory
  {
  

        // calibration files
        HyperFunctions1.dark_img = "../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
        HyperFunctions1.white_img = "../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
        HyperFunctions1.dist_img = "../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";

        // image directory 
        string image_directory = "../../HyperImages/cornfields/session_002/";

        string file_extension = ".cu3";


        // below is start of the code

        std::vector<std::filesystem::directory_entry> entries;

        for (const auto &entry :  std::filesystem::directory_iterator(image_directory)) 
        {
            // cout<<entry.path().extension()<<endl;
            if (entry.path().extension() == file_extension)
            {
                entries.push_back(entry);
            }
        }

        std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
            return std::filesystem::last_write_time(a) < std::filesystem::last_write_time(b);
        });


        
        char* const userSettingsDir =  const_cast<char*>(HyperFunctions1.cubert_settings.c_str());
        char* const darkLoc =  const_cast<char*>(HyperFunctions1.dark_img.c_str());
        char* const whiteLoc =  const_cast<char*>(HyperFunctions1.white_img.c_str());
        char* const distanceLoc =  const_cast<char*>(HyperFunctions1.dist_img.c_str());
        char* const factoryDir =  const_cast<char*>(HyperFunctions1.factory_dir.c_str());
        char* const outDir =  const_cast<char*>(HyperFunctions1.output_dir.c_str());

        cuvis::General::init(userSettingsDir);
        // uncomment below for verbose output from cuvis processing pipeline
        // cuvis::General::set_log_level(loglevel_info);
        cuvis::Measurement dark(darkLoc);
        cuvis::Measurement white(whiteLoc);
        cuvis::Measurement distance(distanceLoc);
        cuvis::Calibration calib(factoryDir);
        cuvis::ProcessingContext proc(calib);
        proc.set_reference(dark, cuvis::reference_type_t::Reference_Dark);
        proc.set_reference(white, cuvis::reference_type_t::Reference_White);
        proc.set_reference(distance, 
        cuvis::reference_type_t::Reference_Distance);
        HyperFunctions1.procArgs.processing_mode = cuvis::processing_mode_t::Cube_Reflectance;
        proc.set_processingArgs(HyperFunctions1.procArgs);

        // load spectral database and move image to gpu
        HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
        


        for (const auto &entry : entries) {
            // std::cout << entry.path() << '\n';

            // // method 1
            // HyperFunctions1.cubert_img = entry.path() ;
            // HyperFunctions1.ReprocessImage( HyperFunctions1.cubert_img);


            // method 2 
            HyperFunctions1.mlt1.clear();

            char* const measurementLoc =  const_cast<char*>(entry.path().c_str());
            cuvis::Measurement mesu(measurementLoc);
                
            
            
                        
            if (proc.is_capable(mesu, HyperFunctions1.procArgs))
            {
                proc.apply(mesu);
            }
            
            
            
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
                HyperFunctions1.mlt1.push_back(singleChannel);
                
                // cv::imshow(" Individual channel ", singleChannel);
                // cv::waitKey(50);

            } 




            // show false rgb image
            HyperFunctions1.false_img_b=2;
            HyperFunctions1.false_img_g=13;
            HyperFunctions1.false_img_r=31;
            HyperFunctions1.GenerateFalseImg();
            HyperFunctions1.DispFalseImage();


            HyperFunctions1.mat_to_oneD_array_parallel_parent();
            HyperFunctions1.allocate_memory();
            // perform classification
            HyperFunctions1.spec_sim_alg=2;
            HyperFunctions1.semantic_segmentation(); 
            HyperFunctions1.DispClassifiedImage();
            HyperFunctions1.deallocate_memory();

            cv::waitKey(1);

            // return -1;
        }
  }


  


  return 0;
}


