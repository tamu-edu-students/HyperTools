#include <gtk/gtk.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "gtkfunctions.cpp"
#include "hyperfunctions.cpp"
using namespace cv;
using namespace std;

  struct img_struct {
  GObject *image;
  HyperFunctions *HyperFunctions1;
  } ;
  
  
int main (int argc, char *argv[])
{
  GtkBuilder *builder;
  GObject *window;
  GObject *button;
  GError *error = NULL;
  
  HyperFunctions HyperFunctions1;

  string file_name2="../../HyperImages/img1.tiff";

  HyperFunctions1.LoadImageHyper1(file_name2);
  
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


  /* Next 4 components are in progress and some are using placeholder callback functions*/

  //button = gtk_builder_get_object (builder, "show_spectrum");
  //g_signal_connect (button, "FIX", G_CALLBACK (show_spectrum), &HyperFunctions1); 
  
  
    

  button = gtk_builder_get_object (builder, "image_box"); 
  //g_signal_connect (button, "clicked", G_CALLBACK (get_point_pos), &HyperFunctions1); //Have a new variable in hyperfunctions.h for point (global variable).
  //Updating could be done through a hyperfunctions method. 
  //int result=gtk_spin_button_get_value (widget);
  //gtk functions.h in the cuvis integration inbutton press callback
  //g_signal_connect (button, "clicked", G_CALLBACK (update_show_spectrum), &HyperFunctions1);
  //Always recalculate but only display image if appropriate/ if toggled on.
  //May need to pass in structs to give toggle switch position, pixel buffer / image file
  //Could have c++ class with struct to store toggle and other UI data.

  button = gtk_builder_get_object (builder, "spectrum_box");
  //Nothing happens when you click on the spectrum image

 
  img_struct *gtk_hyper_image, temp_var1;
  gtk_hyper_image=&temp_var1;
  
  GObject *image;
  
  image= gtk_builder_get_object (builder, "image1");
  
  (*gtk_hyper_image).image=image;
  (*gtk_hyper_image).HyperFunctions1=&HyperFunctions1;

  
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
  
  
  //gtk_spin_button_set_value(rgb_spin[0],10); This is how you change the value of a spin button
  button = gtk_builder_get_object (builder, "spin_red");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_r), &HyperFunctions1);
  
 
  button = gtk_builder_get_object (builder, "spin_green");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_g), &HyperFunctions1);
  
  
  button = gtk_builder_get_object (builder, "spin_blue");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_b), &HyperFunctions1);  
  

  button = gtk_builder_get_object (builder, "false_img_standard");
  g_signal_connect (button, "clicked", G_CALLBACK (set_false_img_standard_rgb), gtk_hyper_image);
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
  
  
  
  button = gtk_builder_get_object (builder, "quit");
  g_signal_connect (button, "clicked", G_CALLBACK (gtk_main_quit), NULL);

  gtk_main ();

  return 0;
}


