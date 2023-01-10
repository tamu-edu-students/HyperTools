#include <gtk/gtk.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "gtkfunctions.cpp"
#include "hyperfunctions.cpp"
using namespace cv;
using namespace std;


int
main (int   argc,
      char *argv[])
{
  GtkBuilder *builder;
  GObject *window;
  GObject *button;
  GError *error = NULL;
  
  HyperFunctions HyperFunctions1;
  string file_name2="../../HyperImages/imagePython2_8int.tiff";
  HyperFunctions1.LoadImageHyper1(file_name2);
  file_name2="../lena.png";
  HyperFunctions1.LoadImageClassified(file_name2);
  HyperFunctions1.spec_simil_img=HyperFunctions1.classified_img;

  
  gtk_init (&argc, &argv);

  /* Construct a GtkBuilder instance and load our UI description */
  builder = gtk_builder_new ();
  if (gtk_builder_add_from_file (builder, "../image_tool.ui", &error) == 0)
    {
      g_printerr ("Error loading file: %s\n", error->message);
      g_clear_error (&error);
      return 1;
    }

  /* Connect signal handlers to the constructed widgets. */
  window = gtk_builder_get_object (builder, "window");
  g_signal_connect (window, "destroy", G_CALLBACK (gtk_main_quit), NULL);

  button = gtk_builder_get_object (builder, "tiled_img");
  g_signal_connect (button, "clicked", G_CALLBACK (TileImage), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "disp_false_img");
  g_signal_connect (button, "clicked", G_CALLBACK (show_false_img), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "disp_spec_sim_img");
  g_signal_connect (button, "clicked", G_CALLBACK (show_spec_sim_img), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "disp_semantic_img");
  g_signal_connect (button, "clicked", G_CALLBACK (show_semantic_img), &HyperFunctions1);  
  
  button = gtk_builder_get_object (builder, "spin_red");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_r), &HyperFunctions1);
 
  button = gtk_builder_get_object (builder, "spin_green");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_g), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "spin_blue");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_false_img_b), &HyperFunctions1);  

  button = gtk_builder_get_object (builder, "false_img_standard");
  g_signal_connect (button, "clicked", G_CALLBACK (set_false_img_standard_rgb), &HyperFunctions1);
  //reset r,g,b spin buttons to standard values here
  // currently not implemented

  button = gtk_builder_get_object (builder, "semantic_SAM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SAM), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (calc_semantic), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (show_semantic_img), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "semantic_SCM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SCM), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (calc_semantic), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (show_semantic_img), &HyperFunctions1);
   
  button = gtk_builder_get_object (builder, "semantic_SID");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SID), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (calc_semantic), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (show_semantic_img), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "similarity_SAM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SAM), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (calc_spec_sim), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (show_spec_sim_img), &HyperFunctions1);
    
  button = gtk_builder_get_object (builder, "similarity_SCM");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SCM), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (calc_spec_sim), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (show_spec_sim_img), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "similarity_SID");
  g_signal_connect (button, "toggled", G_CALLBACK (set_spec_sim_alg_SID), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (calc_spec_sim), &HyperFunctions1);
  g_signal_connect (button, "toggled", G_CALLBACK (show_spec_sim_img), &HyperFunctions1);

  // regenerate object list here here
  // need to set an initial item
  button = gtk_builder_get_object (builder, "similarity_items");
  g_signal_connect (button, "set-focus-child", G_CALLBACK (print_hello), NULL);
  // selected item is changed
  g_signal_connect (button, "changed", G_CALLBACK (print_hello), NULL);
  
  button = gtk_builder_get_object (builder, "quit");
  g_signal_connect (button, "clicked", G_CALLBACK (gtk_main_quit), NULL);

  gtk_main ();

  return 0;
}


