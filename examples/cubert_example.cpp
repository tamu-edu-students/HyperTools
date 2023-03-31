#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "../src/hyperfunctions.cpp"
#include "cuvis.hpp"
#include <cassert>

#include <gtk/gtk.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "../src/gtkfunctions.cpp"
#include "../src/hyperfunctions.cpp"
using namespace cv;
using namespace std;

  struct img_struct {
  GObject *image;
  HyperFunctions *HyperFunctions1;
  } ;
  
  struct entry_struct {
  GObject *entry;
  HyperFunctions *HyperFunctions1;
  } ;

  struct spin_struct {
  GObject *button1;
  GObject *button2;
  GObject *button3;
  HyperFunctions *HyperFunctions1;
  } ;
  


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
  /*
  string cubert_settings="../../HyperImages/set1/settings/";  //ultris20.settings file
  string cubert_img="../../HyperImages/cornfields/session_002/session_002_490.cu3";
  string dark_img="../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
  string white_img="../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
  string dist_img="../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";
  string factor_dir="../../HyperImages/cornfields/factory/"; // requires init.daq file
  string output_dir="../../HyperImages/cornfields/results/";
  */
  
  
  string cubert_settings="../../HyperImages/set1/settings/";  //ultris20.settings file
  string cubert_img="../../HyperImages/hyperspectral-images/cu3/session_000_342.cu3";
  string dark_img="../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
  string white_img="../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
  string dist_img="../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";
  string factor_dir="../../HyperImages/hyperspectral-images/calibration/"; // requires init.daq file
  string output_dir="../../HyperImages/hyperspectral-images/results/";
  
	
  
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
  
  
  ///*std::cout << "Data 1:" << mesu.get_meta()->name << " "
  //          << "t=" << mesu.get_meta()->integration_time << " ms "
  //          << "mode=" << mesu.get_meta()->processing_mode << " " << std::endl;  
  //
 // std::cout << "Loading Calibration and processing context (factory)" << std::endl;
 // */
 
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






GtkBuilder *builder;
  GObject *window;
  GObject *button;
  GError *error = NULL;
  
  HyperFunctions HyperFunctions1;
   HyperFunctions1.mlt1=mlt1;






  gtk_init (&argc, &argv);

  /* Construct a GtkBuilder instance and load our UI description */
  builder = gtk_builder_new ();
  if (gtk_builder_add_from_file (builder, "../UI/image_tool.ui", &error) == 0)
    {
      g_printerr ("Error loading file: %s\n", error->message);
      g_clear_error (&error);
      return 1;
    }

  /* Connect signal handlers to the constructed widgets. */
  window = gtk_builder_get_object (builder, "window");
  g_signal_connect (window, "destroy", G_CALLBACK (gtk_main_quit), NULL);

  button = gtk_builder_get_object (builder, "choose_file");
  g_signal_connect (button, "file-set", G_CALLBACK (choose_image_file), &HyperFunctions1); //Should be able to see what file they chose. Then call LoadImageHyper1

  button = gtk_builder_get_object(builder, "choose_database");
  g_signal_connect (button, "file-set", G_CALLBACK (choose_database), &HyperFunctions1); //See what database file you choose

  button = gtk_builder_get_object (builder, "spectrum_box");
  //Nothing happens when you click on the spectrum image

  img_struct *gtk_hyper_image, temp_var1;
  gtk_hyper_image=&temp_var1;
  GObject *image;
  image= gtk_builder_get_object (builder, "image1");
  
  (*gtk_hyper_image).image=image;
  (*gtk_hyper_image).HyperFunctions1=&HyperFunctions1;


  entry_struct *gtk_hyper_entry, temp_var3;
  gtk_hyper_entry=&temp_var3;
  
  button = gtk_builder_get_object (builder, "database_name");
  (*gtk_hyper_entry).entry=button;
  (*gtk_hyper_entry).HyperFunctions1=&HyperFunctions1;
  button = gtk_builder_get_object (builder, "create_database");
  g_signal_connect (button, "clicked", G_CALLBACK (create_database), gtk_hyper_entry);

  entry_struct *gtk_hyper_entry2, temp_var5;
  gtk_hyper_entry2=&temp_var5;

  button = gtk_builder_get_object (builder, "spectrum_name");
  (*gtk_hyper_entry2).entry=button;
  (*gtk_hyper_entry2).HyperFunctions1=&HyperFunctions1;
  button = gtk_builder_get_object (builder, "save_spectrum");
  g_signal_connect (button, "clicked", G_CALLBACK (save_spectrum), gtk_hyper_entry2);

  
  image= gtk_builder_get_object (builder, "spec_curve");
  img_struct *gtk_hyper_image2, temp_var2;
  gtk_hyper_image2=&temp_var2;
  (*gtk_hyper_image2).image=image;
  (*gtk_hyper_image2).HyperFunctions1=&HyperFunctions1;
  
  
  button = gtk_builder_get_object (builder, "image_box");
  g_signal_connect (G_OBJECT (button),"button_press_event",G_CALLBACK (button_press_callback),&HyperFunctions1);
  g_signal_connect (G_OBJECT (button),"button_press_event",G_CALLBACK (show_spectrum),gtk_hyper_image2);

  // button set to invisible since the function does not work properly yet
  //button = gtk_builder_get_object (builder, "disp_ndvi_img");
  //g_signal_connect (button, "clicked", G_CALLBACK (show_ndvi_image), gtk_hyper_image);

  button = gtk_builder_get_object (builder, "disp_false_img");
  g_signal_connect (button, "clicked", G_CALLBACK (show_false_img), gtk_hyper_image);

  button = gtk_builder_get_object (builder, "disp_spec_sim_img");
  g_signal_connect (button, "clicked", G_CALLBACK (calc_spec_sim), &HyperFunctions1);
  g_signal_connect (button, "clicked", G_CALLBACK (show_spec_sim_img), gtk_hyper_image);

  button = gtk_builder_get_object (builder, "disp_semantic_img");
  g_signal_connect (button, "clicked", G_CALLBACK (calc_semantic), &HyperFunctions1);
  g_signal_connect (button, "clicked", G_CALLBACK (show_semantic_img), gtk_hyper_image);  
  
  button = gtk_builder_get_object (builder, "tiled_img");
  g_signal_connect (button, "clicked", G_CALLBACK (TileImage), gtk_hyper_image);
  
  
  spin_struct *gtk_spin_buttons, temp_var4;
  gtk_spin_buttons=&temp_var4;
  (*gtk_spin_buttons).HyperFunctions1 = &HyperFunctions1;
  

  //gtk_spin_button_set_value(rgb_spin[0],10); This is how you change the value of a spin button
  button = gtk_builder_get_object (builder, "spin_red");
  (*gtk_spin_buttons).button1 = button;
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_r), gtk_hyper_image);
  
 
  button = gtk_builder_get_object (builder, "spin_green");
  (*gtk_spin_buttons).button2 = button;
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_g), gtk_hyper_image);
  
  
  button = gtk_builder_get_object (builder, "spin_blue");
  (*gtk_spin_buttons).button3 = button;
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_b), gtk_hyper_image);  
  
  button = gtk_builder_get_object (builder, "reset_false_img");
  g_signal_connect (button, "clicked", G_CALLBACK (set_false_img_reset), gtk_hyper_image);
  g_signal_connect (button, "clicked", G_CALLBACK (set_spin_buttons_reset), gtk_spin_buttons);

  button = gtk_builder_get_object (builder, "false_img_standard");
  g_signal_connect (button, "clicked", G_CALLBACK (set_false_img_standard_rgb), gtk_hyper_image);
  g_signal_connect (button, "clicked", G_CALLBACK (set_spin_buttons_standard_rgb), gtk_spin_buttons);
  //g_signal_connect (button, "clicked", G_CALLBACK (set_false_img_standard_rgb), &HyperFunctions1);
  //reset r,g,b spin buttons to standard values here
  // currently not implemented

  button = gtk_builder_get_object (builder, "semantic_SAM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SAM), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (calc_semantic), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (show_semantic_img), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "semantic_SCM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SCM), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (calc_semantic), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (show_semantic_img), &HyperFunctions1);
   
  button = gtk_builder_get_object (builder, "semantic_SID");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SID), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (calc_semantic), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (show_semantic_img), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_SAM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SAM), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (calc_spec_sim), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (show_spec_sim_img), &HyperFunctions1);
    
  button = gtk_builder_get_object (builder, "similarity_SCM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SCM), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (calc_spec_sim), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (show_spec_sim_img), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "similarity_SID");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SID), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (calc_spec_sim), &HyperFunctions1);
  //g_signal_connect (button, "toggled", G_CALLBACK (show_spec_sim_img), &HyperFunctions1);

  // regenerate object list here here
  // need to set an initial item
  button = gtk_builder_get_object (builder, "similarity_items");
  g_signal_connect (button, "set-focus-child", G_CALLBACK (get_class_list), &HyperFunctions1);
  // selected item is changed
  g_signal_connect (button, "changed", G_CALLBACK (get_list_item), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "clear_database");
  g_signal_connect (button, "clicked", G_CALLBACK (clear_database), &HyperFunctions1);
  
  
  button = gtk_builder_get_object (builder, "quit");
  g_signal_connect (button, "clicked", G_CALLBACK (gtk_main_quit), NULL);

  gtk_main ();

 












  return 0;
}


