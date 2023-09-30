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


int main (int argc, char *argv[])
{
  string file_name3="../../HyperImages/img1.tiff";
  string file_name4="../../HyperImages/img2.tiff";
  
  HyperFunctions HyperFunctions1;
  HyperFunctions1.LoadImageHyper1(file_name3);
  HyperFunctions1.feature_img1=HyperFunctions1.mlt1[0];
  HyperFunctions1.LoadImageHyper2(file_name4);
  HyperFunctions1.feature_img2=HyperFunctions1.mlt2[0];
  
  GtkBuilder *builder;
  GObject *window;
  GObject *button;
  GError *error = NULL;
  gtk_init (&argc, &argv);

  /* Construct a GtkBuilder instance and load our UI description */
  builder = gtk_builder_new ();
  if (gtk_builder_add_from_file (builder, "../UI/feature_tool.ui", &error) == 0)
    {
      g_printerr ("Error loading file: %s\n", error->message);
      g_clear_error (&error);
      return 1;
    }

  /* Connect signal handlers to the constructed widgets. */

  img_struct *gtk_hyper_image, temp_var1;
  gtk_hyper_image=&temp_var1;
  GObject *image;
  image= gtk_builder_get_object (builder, "disp_img");
  
  (*gtk_hyper_image).image=image;
  (*gtk_hyper_image).HyperFunctions1=&HyperFunctions1;

  window = gtk_builder_get_object (builder, "window");
  g_signal_connect (window, "destroy", G_CALLBACK (gtk_main_quit), NULL);
  
  button = gtk_builder_get_object (builder, "spin_image_layer");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_img_layer), &HyperFunctions1);
  g_signal_connect (button, "value-changed", G_CALLBACK (feature_images), gtk_hyper_image); 

  button = gtk_builder_get_object (builder, "detect_SIFT");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_detector_SIFT), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "detect_SURF");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_detector_SURF), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "detect_FAST");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_detector_FAST), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "detect_ORB");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_detector_ORB), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "descript_SIFT");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_descriptor_SIFT), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "descript_SURF");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_descriptor_SURF), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "descript_ORB");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_descriptor_ORB), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "match_bf");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_matcher_BF), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "match_flann");
  g_signal_connect (button, "toggled", G_CALLBACK (set_feature_matcher_FLANN), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "show_results");
  g_signal_connect (button, "clicked", G_CALLBACK (feature_results), gtk_hyper_image);

  button = gtk_builder_get_object (builder, "show_base_imgs");
  g_signal_connect (button, "clicked", G_CALLBACK (feature_images), gtk_hyper_image);
  
  button = gtk_builder_get_object (builder, "disp_transformation");
  g_signal_connect (button, "clicked", G_CALLBACK (print_transformation), &HyperFunctions1);  

  button = gtk_builder_get_object (builder, "quit");
  g_signal_connect (button, "clicked", G_CALLBACK (gtk_main_quit), NULL);

  gtk_main ();

  return 0;
}


