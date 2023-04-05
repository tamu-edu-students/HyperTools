#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "../src/hypergpufunctions.cu"
#include <gtk/gtk.h>
#include "../src/gtkfunctions.cpp"
#include "../src/gtkfunctions_gpu.cpp"

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

int main (int argc, char *argv[]) {

  string file_name2="../../HyperImages/img1.tiff";

  HyperFunctionsGPU HyperFunctions1;
  HyperFunctions1.LoadImageHyper1(file_name2);
  HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
  HyperFunctions1.mat_to_oneD_array_parallel_parent();
  HyperFunctions1.allocate_memory();


  GtkBuilder *builder;
  GObject *window;
  GObject *button;
  GError *error = NULL;  
  gtk_init (&argc, &argv);
  builder = gtk_builder_new ();
  if (gtk_builder_add_from_file (builder, "../UI/image_tool.ui", &error) == 0)
    {
      g_printerr ("Error loading file: %s\n", error->message);
      g_clear_error (&error);
      return 1;
    }

  window = gtk_builder_get_object (builder, "window");
  g_signal_connect (window, "destroy", G_CALLBACK (gtk_main_quit), NULL);
  
  button = gtk_builder_get_object (builder, "choose_file");
  g_signal_connect (button, "file-set", G_CALLBACK (choose_image_file_gpu), &HyperFunctions1); 

  
  button = gtk_builder_get_object (builder, "spectrum_box");

  img_struct *gtk_hyper_image, temp_var1;
  gtk_hyper_image=&temp_var1;
  GObject *image;
  image= gtk_builder_get_object (builder, "image1");
  
  (*gtk_hyper_image).image=image;
  (*gtk_hyper_image).HyperFunctions1=&HyperFunctions1;
  
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
  g_signal_connect (button, "clicked", G_CALLBACK (calc_spec_sim_gpu), &HyperFunctions1);
  g_signal_connect (button, "clicked", G_CALLBACK (show_spec_sim_img), gtk_hyper_image);

  button = gtk_builder_get_object (builder, "disp_semantic_img");
  g_signal_connect (button, "clicked", G_CALLBACK (calc_semantic_gpu), &HyperFunctions1);
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

  // not currently implemented in CUDA so nothing will happen
  button = gtk_builder_get_object (builder, "semantic_ED");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_EuD), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_SAM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SAM), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_SCM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SCM), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_SID");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SID), &HyperFunctions1);

  // not currently implemented in CUDA so nothing will happen
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


