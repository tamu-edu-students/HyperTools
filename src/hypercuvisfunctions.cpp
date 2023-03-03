#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include "hyperfunctions.cpp"
#include "hyperfunctions.h"
#include "hypercuvisfunctions.h"
#include "ctpl.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

// Loads first hyperspectral image for analysis
void HyperFunctionsCuvis::LoadImageHyper1(string file_name)
{
    mlt1.clear();
	string file_ext;

    size_t dotPos = file_name.find_last_of('.');
    if (dotPos != std::string::npos) {
        file_ext = file_name.substr(dotPos + 1);
    }
   
    // cout<<"extension is: "<<file_ext<<endl; 

    if (file_ext=="cu3")
    {
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
        cuvis::Calibration calib(factoryDir);
        cuvis::ProcessingContext proc(calib);
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
            saveArgs.export_dir = std::filesystem::path(outDir) / mode.first;
            // mesu.save(saveArgs);
            }
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
            mlt1.push_back(singleChannel);
            // cv::imshow(" Individual channel ", singleChannel);
            // cv::waitKey(50);
        } 


    }
    else if (file_ext=="cu3s")
    {
        char* const userSettingsDir =  const_cast<char*>( cubert_settings.c_str());
        char* const measurementLoc =  const_cast<char*>( cubert_img.c_str());

        // std::cout << "loading user settings..." << std::endl;
        cuvis::General::init(userSettingsDir);
        cuvis::General::set_log_level(loglevel_info);

        // std::cout << "loading session... " << std::endl;
        cuvis::SessionFile sess(measurementLoc);

        // std::cout << "loading measurement... " << std::endl;
        auto optmesu = sess.get_mesu(0);
        assert(optmesu.has_value());
        cuvis::Measurement mesu = optmesu.value();

        // std::cout << "Data 1" << mesu.get_meta()->name << " "
        //         << "t=" << mesu.get_meta()->integration_time << " ms "
        //         << "mode=" << mesu.get_meta()->processing_mode << " " << std::endl;
        if (mesu.get_meta()->measurement_flags.size() > 0)
        {
        std::cout << "  Flags" << std::endl;
        for (auto const& flags : mesu.get_meta()->measurement_flags)
        {
            std::cout << "  - " << flags.first << " (" << flags.second << ")"
                    << std::endl;
        }
        }

        assert(
            mesu.get_meta()->processing_mode == Cube_Raw &&
            "This example requires raw mode");

        auto const& cube_it = mesu.get_imdata()->find(CUVIS_MESU_CUBE_KEY);
        assert(cube_it != mesu.get_imdata()->end() && "Cube not found");

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
            // cv::imshow(" Individual channel ", singleChannel);
            // cv::waitKey(50);
        }         

    }
    else if (file_ext=="tiff")
    {
        imreadmulti(file_name, mlt1);
    }
    
    
    
}

// Loads second hyperspectral image for analysis
// mainly used for feature matching 
void HyperFunctionsCuvis::LoadImageHyper2(string file_name)
{
	mlt2.clear();
	
    string file_ext;

    size_t dotPos = file_name.find_last_of('.');
    if (dotPos != std::string::npos) {
        file_ext = file_name.substr(dotPos + 1);
    }
   
    // cout<<"extension is: "<<file_ext<<endl; 

    if (file_ext=="cu3")
    {
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
        cuvis::Calibration calib(factoryDir);
        cuvis::ProcessingContext proc(calib);
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
            saveArgs.export_dir = std::filesystem::path(outDir) / mode.first;
            // mesu.save(saveArgs);
            }
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
            mlt2.push_back(singleChannel);
            // cv::imshow(" Individual channel ", singleChannel);
            // cv::waitKey(50);
        } 
        

    }
    else if (file_ext=="cu3s")
    {
        char* const userSettingsDir =  const_cast<char*>( cubert_settings.c_str());
        char* const measurementLoc =  const_cast<char*>( cubert_img.c_str());

        // std::cout << "loading user settings..." << std::endl;
        cuvis::General::init(userSettingsDir);
        cuvis::General::set_log_level(loglevel_info);

        // std::cout << "loading session... " << std::endl;
        cuvis::SessionFile sess(measurementLoc);

        // std::cout << "loading measurement... " << std::endl;
        auto optmesu = sess.get_mesu(0);
        assert(optmesu.has_value());
        cuvis::Measurement mesu = optmesu.value();

        // std::cout << "Data 1" << mesu.get_meta()->name << " "
        //         << "t=" << mesu.get_meta()->integration_time << " ms "
        //         << "mode=" << mesu.get_meta()->processing_mode << " " << std::endl;
        if (mesu.get_meta()->measurement_flags.size() > 0)
        {
        std::cout << "  Flags" << std::endl;
        for (auto const& flags : mesu.get_meta()->measurement_flags)
        {
            std::cout << "  - " << flags.first << " (" << flags.second << ")"
                    << std::endl;
        }
        }

        assert(
            mesu.get_meta()->processing_mode == Cube_Raw &&
            "This example requires raw mode");

        auto const& cube_it = mesu.get_imdata()->find(CUVIS_MESU_CUBE_KEY);
        assert(cube_it != mesu.get_imdata()->end() && "Cube not found");

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
            mlt2.push_back(singleChannel);
            // cv::imshow(" Individual channel ", singleChannel);
            // cv::waitKey(50);
        } 


    }
    else if (file_ext=="tiff")
    {
        imreadmulti(file_name, mlt2);
    }
   
}


//export cubert image to multipage tiff image
//assumes cu3s as input
void HyperFunctionsCuvis::ExportTiff()
{

    string cubert_settings="../../HyperImages/settings/";  //ultris20.settings file
    char* const sessionLoc  =  const_cast<char*>(cubert_img.c_str());
    char* const userSettingsDir =  const_cast<char*>(cubert_settings.c_str());
    char* const exportDir =  const_cast<char*>(output_dir.c_str());

    cuvis::General::init(userSettingsDir);
    cuvis::General::set_log_level(loglevel_info);
    cuvis::SessionFile sess(sessionLoc);

    std::cout << "loading measurement... " << std::endl;
    auto optmesu = sess.get_mesu(0);
    assert(optmesu.has_value());
    cuvis::Measurement mesu = optmesu.value();

    cuvis::ProcessingArgs procArgs;   
    procArgs.processing_mode = cuvis::processing_mode_t::Cube_Reflectance;

    char* const factoryDir =  const_cast<char*>(factor_dir.c_str());
    cuvis::Calibration calib(factoryDir);
    cuvis::ProcessingContext proc(calib);
    proc.set_processingArgs(procArgs);
    
    // not sure of how to change the processing mode
    // right now it is just set to raw
    // maybe it is because calibration files are not in cu3s file

    // proc.apply(mesu);
    
    cout<<mesu.get_meta()->processing_mode<<" process mode "<<endl;
    
    
    assert(mesu.get_meta()->processing_mode != cuvis::processing_mode_t::Preview);
    {
        std::cout << "Export to Multi-Page Tiff" << std::endl;
        cuvis::TiffArgs args;
        char exportDirMulti[CUVIS_MAXBUF];
        strcpy(exportDirMulti, exportDir);
        strcat(exportDirMulti, "/multi");
        args.export_dir = exportDirMulti;
        args.format = cuvis::tiff_format_t::tiff_format_MultiPage;
        cuvis::TiffExporter exporter(args);
        exporter.apply(mesu);
     }


    

}