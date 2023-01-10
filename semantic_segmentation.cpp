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
  GtkAdjustment *adjustment;
  GtkSpinButton *spinbutton;
  

    
    
  HyperFunctions HyperFunctions1;
  gpointer HyperFunctions2 = static_cast<gpointer>(&HyperFunctions1); 
  string file_name2="../lena.png";
  string file_name3="../json/lena3.json";
  HyperFunctions1.LoadImageClassified(file_name2);
  HyperFunctions1.read_img_json(file_name3);
  
    gtk_init (&argc, &argv);

  /* Construct a GtkBuilder instance and load our UI description */
  builder = gtk_builder_new ();
  if (gtk_builder_add_from_file (builder, "../segmentation.ui", &error) == 0)
    {
      g_printerr ("Error loading file: %s\n", error->message);
      g_clear_error (&error);
      return 1;
    }

  /* Connect signal handlers to the constructed widgets. */
  window = gtk_builder_get_object (builder, "window");
  g_signal_connect (window, "destroy", G_CALLBACK (gtk_main_quit), NULL);

  button = gtk_builder_get_object (builder, "edge_detect");
  g_signal_connect (button, "clicked", G_CALLBACK (EdgeDetection), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "show_contours");
  g_signal_connect (button, "clicked", G_CALLBACK (show_contours), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "show_differences");
  g_signal_connect (button, "clicked", G_CALLBACK (show_difference), &HyperFunctions1);

  button = gtk_builder_get_object (builder, "orig_img");
  g_signal_connect (button, "clicked", G_CALLBACK (show_semantic_img), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "scale_approx_poly");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_approx_poly), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "scale_min_area");
  g_signal_connect (button, "value-changed", G_CALLBACK (set_min_area), &HyperFunctions1);
  
  button = gtk_builder_get_object (builder, "quit");
  g_signal_connect (button, "clicked", G_CALLBACK (gtk_main_quit), NULL);

  gtk_main ();

  return 0;
}


