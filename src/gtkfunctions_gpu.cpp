#if !defined(GTKFUNCTIONS_GPU_H)
#define GTKFUNCTIONS_H
#include <gtk/gtk.h>
#include <iostream>
#include "gtkfunctions.h"
#include "hyperfunctions.cpp"
#include "hypergpufunctions.cu"
#include <string>

using namespace std;
using namespace cv;

static void calc_spec_sim_gpu (GtkWidget *widget, gpointer   data);
static void calc_semantic_gpu (GtkWidget *widget, gpointer   data);
static void choose_image_file_gpu(GtkFileChooser *widget,  gpointer data);

static void calc_semantic_gpu (GtkWidget *widget, gpointer   data)
{
    void * data_new=data;
    HyperFunctionsGPU *HyperFunctions1=static_cast<HyperFunctionsGPU*>(data_new);
    HyperFunctions1->semantic_segmentation();  
}

static void calc_spec_sim_gpu (GtkWidget *widget, gpointer   data)
{

    void * data_new=data;
    HyperFunctionsGPU *HyperFunctions1=static_cast<HyperFunctionsGPU*>(data_new);
    HyperFunctions1->spec_sim_GPU();
     
}

static void choose_image_file_gpu(GtkFileChooser *widget,  gpointer data) {

    gchar* file_chosen;
    file_chosen = gtk_file_chooser_get_filename(widget);
    void * data_new=data;
    HyperFunctionsGPU *HyperFunctions1=static_cast<HyperFunctionsGPU*>(data_new);
    
    HyperFunctions1->deallocate_memory();
    HyperFunctions1->LoadImageHyper(file_chosen);
    HyperFunctions1->read_ref_spec_json(HyperFunctions1->spectral_database);
    HyperFunctions1->mat_to_oneD_array_parallel_parent();
    HyperFunctions1->allocate_memory();
    
}

#endif 
