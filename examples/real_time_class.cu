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

    int exposure_ms = 200;
    string file_name = "test";
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    // below is for ultris 5 example
   
    HyperFunctions1.cubert_settings="../settings/ultris5";  //camera settings file 
    HyperFunctions1.factory_dir="../settings/ultris5"; // requires init.daq file


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

        // cuvis::SaveArgs saveArgs;
        // saveArgs.allow_overwrite = true;
        // saveArgs.export_dir = recDir;
        // saveArgs.allow_session_file = true;

        // cuvis::CubeExporter exporter(saveArgs);

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
        while(true)
        {
            
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
                }
                HyperFunctions1.false_img_b=2;
                HyperFunctions1.false_img_g=13;
                HyperFunctions1.false_img_r=31;
                HyperFunctions1.GenerateFalseImg();
                imshow("test",  HyperFunctions1.false_img);

                HyperFunctions1.mat_to_oneD_array_parallel_parent();
                HyperFunctions1.allocate_memory();
                // // perform classification
                HyperFunctions1.spec_sim_alg=2;
                // HyperFunctions1.semantic_segmentation(); 
                // HyperFunctions1.DispClassifiedImage();
                HyperFunctions1.spec_sim_GPU();
                HyperFunctions1.DispSpecSim();
                HyperFunctions1.deallocate_memory();

                cv::waitKey(1);

                std::cout << "image processed" << std::endl;
            }
            else
            {
            std::cout << "failed" << std::endl;
            }
        }
  


  return 0;
}


