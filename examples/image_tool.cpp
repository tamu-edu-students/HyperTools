#include <gtk/gtk.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/gtkfunctions.cpp"
#include "../src/hyperfunctions.cpp"
#include "cuvis.hpp"
#include "cuvis.h"

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



   string cubert_img="/media/anthony/Antonio/HyperCode/HyperImages/segmented-datasets/Wextel-Dataset/session_002_637.cu3";
     string cubert_settings="../../HyperImages/settings/";  //ultris20.settings file
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
      HyperFunctions HyperFunctions1;
  HyperFunctions1.mlt1=mlt1;
  

  GtkBuilder *builder;
  GObject *window;
  GObject *button;
  GError *error = NULL;
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
  g_signal_connect (button, "file-set", G_CALLBACK (choose_image_file), &HyperFunctions1); 

  button = gtk_builder_get_object(builder, "choose_database");
  g_signal_connect (button, "file-set", G_CALLBACK (choose_database), &HyperFunctions1); 

  button = gtk_builder_get_object (builder, "spectrum_box");

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

  button = gtk_builder_get_object (builder, "semantic_SAM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SAM), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "semantic_SCM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SCM), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "semantic_SID");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SID), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "semantic_ED");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_EuD), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_SAM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SAM), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_SCM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SCM), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_SID");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SID), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_ED");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_EuD), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_items");
  g_signal_connect (button, "set-focus-child", G_CALLBACK (get_class_list), &HyperFunctions1);

  g_signal_connect (button, "changed", G_CALLBACK (get_list_item), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "clear_database");
  g_signal_connect (button, "clicked", G_CALLBACK (clear_database), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "quit");
  g_signal_connect (button, "clicked", G_CALLBACK (gtk_main_quit), NULL);

  gtk_main ();

  return 0;
}


