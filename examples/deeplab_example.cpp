#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include <cassert>
#include "cuvis.hpp"
#include "cuvis.h"
#include <filesystem>


using namespace  std::filesystem;
using namespace cv;
using namespace std;
using namespace std::chrono;

int reprocess_data_child(int id, int j, string path, string rgb_path_train, string label_path_train, string cubert_settings, string dark_img, string white_img, string dist_img, string factor_dir, string output_dir, vector<string> cu3_files,string rgb_path_val, string label_path_val);

int load_data_child(int id, int j, string path, string rgb_path_train, string label_path_train, string cubert_settings, vector<string> cu3_files,string rgb_path_val, string label_path_val);   

int main (int argc, char *argv[])
{
   //directory path for input 
   string path("../../HyperImages/segmented-datasets/Wextel-Dataset");
   // diectory path for output rgb images
  string rgb_path_train("../../HyperImages/export/deeplab/train/Images");
   string label_path_train("../../HyperImages/export/deeplab/train/Labels");
     string rgb_path_val("../../HyperImages/export/deeplab/val/Images");
   string label_path_val("../../HyperImages/export/deeplab/val/Labels");
   
   // calibration files
   string cubert_settings="../../HyperImages/settings/";  //ultris20.settings file
    
    // below for reprocessing
    string dark_img="../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
    string white_img="../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
    string dist_img="../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";
    string factor_dir="../../HyperImages/cornfields/Calibration/"; // requires init.daq file
    string output_dir="../../HyperImages/cornfields/results/";
   
   // set low enough to not overload ram
   // 7 threads for reprocessing code
   	int num_threads=std::thread::hardware_concurrency(); // number of parallel threads
    ctpl::thread_pool p2(num_threads);
   
   //extension 
   string ext(".cu3");
   int k=0;
   vector<string> cu3_files;
    
    // put all cu3 files from directory into vector for processing
    for (auto &p : std::filesystem::recursive_directory_iterator(path))
    {
        if (p.path().extension() == ext)
        {
            
            k++;
            string temp_string=p.path().stem().string();
            char temp_char=temp_string[0];
            //cout<<temp_char <<endl;
            if(temp_char=='s')
            {
            cu3_files.push_back(p.path().stem().string());
            cout << "file "<< k <<"  "<<p.path().stem().string() << '\n';
            }
        }
            
    }
    
    
    // writes an empty view file because sdk requires it
    Mat view_img(410,410,0);
    
    for(int j=0; j<cu3_files.size() ; j++)
    {
            imwrite(path+"/"+cu3_files[j]+"_view.tiff",view_img);
            
    }
    
    
    
    
    // process images to reflectance and save 
    // multithreaded 
auto start = high_resolution_clock::now();


    for(int j=0; j<cu3_files.size() ; j++)
    {
    //p2.push(reprocess_data_child,j,  path,  rgb_path_train,  label_path_train,  cubert_settings,  dark_img,  white_img,  dist_img,  factor_dir,  output_dir,  cu3_files,rgb_path_val  , label_path_val);
    p2.push(load_data_child,j,  path,  rgb_path_train,  label_path_train,  cubert_settings,   cu3_files,rgb_path_val  , label_path_val);
    
   }
     
   
       // wait until threadpool is finished here
    while(p2.n_idle()<num_threads)
    {
        //cout<<" running threads "<< p.size()  <<" idle threads "<<  p.n_idle()  <<endl;
        //do nothing 
    }
auto end = high_resolution_clock::now();
cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;



  return 0;
}



int load_data_child(int id, int j, string path, string rgb_path_train, string label_path_train, string cubert_settings, vector<string> cu3_files,string rgb_path_val, string label_path_val)
{

    string cubert_img=path+"/"+cu3_files[j]+".cu3";
    
    char* const measurementLoc =  const_cast<char*>(cubert_img.c_str());
    char* const userSettingsDir =  const_cast<char*>(cubert_settings.c_str());
    
    

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
    vector<Mat> mlt1;
    
    for (int i=0; i<img.channels();i++)
    {
    cv::Mat singleChannel;
    cv::extractChannel(
        img, singleChannel, i); // extract channel 25 as an example
    singleChannel.convertTo(singleChannel, CV_8U, 1 / 16.0);
    mlt1.push_back(singleChannel);
    //cv::imshow("Individual channel", singleChannel);
    //cv::waitKey(50);
    }

  // write rgb image out 
  Mat false_img;
  vector<Mat> channels(3);
  channels[0]=mlt1[25]; //b
  channels[1]=mlt1[40]; //g
  channels[2]=mlt1[78]; //r
  merge(channels,false_img); // create new single channel image

   
   
   HyperFunctions HyperFunctions1;
    HyperFunctions1.mlt1=mlt1;
    HyperFunctions1.num_threads=10;
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    HyperFunctions1.SemanticSegmenter();
   
   // HyperFunctions1.classified_img assumes values are the labels
   Mat label_img;
   label_img = HyperFunctions1.classified_img;
   cvtColor(label_img, label_img, COLOR_BGR2GRAY);
    
    if (j%4==0)
    {
        //cout<<"multiple of 4 "<<j<<endl;
        imwrite(label_path_val+"/"+cu3_files[j]+".png",label_img);
        cout<<"saved labeled image for "<<cu3_files[j]<<endl;
        imwrite(rgb_path_val+"/"+cu3_files[j]+".png",false_img);
        cout<<"saved rgb image for "<<cu3_files[j]<<endl;
        
    }
    else
    {
        imwrite(label_path_train+"/"+cu3_files[j]+".png",label_img);
        cout<<"saved labeled image for "<<cu3_files[j]<<endl;
        imwrite(rgb_path_train+"/"+cu3_files[j]+".png",false_img);
        cout<<"saved rgb image for "<<cu3_files[j]<<endl;
    
    }
    

  
return -1;
}



int reprocess_data_child(int id, int j, string path, string rgb_path_train, string label_path_train, string cubert_settings, string dark_img, string white_img, string dist_img, string factor_dir, string output_dir, vector<string> cu3_files,string rgb_path_val, string label_path_val)
{
    string cubert_img=path+"/"+cu3_files[j]+".cu3";
    

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
        //printf("reprocess measurement to Cube_Reflectance mode...\n");
        //fflush(stdout);

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
        //cv::imshow(" Individual channel ", singleChannel);
        //cv::waitKey(50);
    }
    cuvis_calib_free(&calib);
    cuvis_proc_cont_free(&procCont);
    cuvis_measurement_free(&mesu);
    cuvis_measurement_free(&dark);
    cuvis_measurement_free(&white);
    cuvis_measurement_free(&distance); 

    // write rgb image out 
  Mat false_img;
  vector<Mat> channels(3);
  channels[0]=mlt1[25]; //b
  channels[1]=mlt1[40]; //g
  channels[2]=mlt1[78]; //r
  merge(channels,false_img); // create new single channel image

   
   
   HyperFunctions HyperFunctions1;
    HyperFunctions1.mlt1=mlt1;
    HyperFunctions1.num_threads=10;
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    HyperFunctions1.SemanticSegmenter();
   
   // HyperFunctions1.classified_img assumes values are the labels
   Mat label_img;
   label_img = HyperFunctions1.classified_img;
   cvtColor(label_img, label_img, COLOR_BGR2GRAY);
    
    if (j%4==0)
    {
        //cout<<"multiple of 4 "<<j<<endl;
        imwrite(label_path_val+"/"+cu3_files[j]+".png",label_img);
        cout<<"saved labeled image for "<<cu3_files[j]<<endl;
        imwrite(rgb_path_val+"/"+cu3_files[j]+".png",false_img);
        cout<<"saved rgb image for "<<cu3_files[j]<<endl;
        
    }
    else
    {
        imwrite(label_path_train+"/"+cu3_files[j]+".png",label_img);
        cout<<"saved labeled image for "<<cu3_files[j]<<endl;
        imwrite(rgb_path_train+"/"+cu3_files[j]+".png",false_img);
        cout<<"saved rgb image for "<<cu3_files[j]<<endl;
    
    }
    
   
    
    
  return 0;     
}        
       
       
       
       
