#if !defined(GTKFUNCTIONS_GPU_H)
#define GTKFUNCTIONS_H
#include <gtk/gtk.h>
#include <iostream>
#include "gtkfunctions.h"
#include "hyperfunctions.cpp"
#include "hypergpufunctions.cu"
#include <string>
#include <opencv2/plot.hpp>


using namespace std;
using namespace cv;


static void calc_spec_sim_gpu (GtkWidget *widget, gpointer   data);
static void calc_semantic_gpu (GtkWidget *widget, gpointer   data);



static void calc_semantic_gpu (GtkWidget *widget, gpointer   data)
{
    void * data_new=data;
    HyperFunctionsGPU *HyperFunctions1=static_cast<HyperFunctionsGPU*>(data_new);
    HyperFunctions1->read_ref_spec_json( HyperFunctions1->spectral_database);
    //int* test_array= HyperFunctions1->mat_to_oneD_array_parallel_parent(  );
    HyperFunctions1->semantic_segmentation(HyperFunctions1->img_array_base);
    //HyperFunctions1->DispClassifiedImage();
    //cv::waitKey();
    //delete test_array;
    
}

static void calc_spec_sim_gpu (GtkWidget *widget, gpointer   data)
{

    
    void * data_new=data;
    HyperFunctionsGPU *HyperFunctions1=static_cast<HyperFunctionsGPU*>(data_new);
    HyperFunctions1->read_ref_spec_json(HyperFunctions1->spectral_database);
    //int* test_array= HyperFunctions1->mat_to_oneD_array_parallel_parent(  );
    HyperFunctions1->allocate_memory(HyperFunctions1->img_array_base);

    HyperFunctions1->spec_sim_GPU();
    HyperFunctions1->deallocate_memory();
    //delete test_array;
    //HyperFunctions1->DispSpecSim();
    //cv::waitKey();    
    
}




#endif 
