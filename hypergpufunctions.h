#if !defined(HYPERGPUFUNCTIONS_H)
#define HYPERGPUFUNCTIONS_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
//#include <jsoncpp/json/json.h>
//#include <jsoncpp/json/writer.h>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include "ctpl.h"
#include <stdio.h>
#include "hyperfunctions.h"


using namespace std;
using namespace cv;

void mat_to_oneD_array_parallel_child(int id,vector<Mat>* mlt2, int* host_img_array, int val_it, int k );

__global__ void img_test_multi_thread_SAM(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__global__ void img_test_multi_thread_SID(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__global__ void img_test_multi_thread_SCM(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__global__ void img_test_classifier(int *out, int *img_array, int num_pixels, int num_spectrums, int* color_info ) ;


class HyperFunctionsGPU : public HyperFunctions {
public:

    int N_points;
    int N_size;    
    int num_lay;
    int *out;
    int *d_img_array,  *d_out, *d_ref_spectrum; 
    int block_size = 512;
    int grid_size ;
    
    
    void spec_sim_GPU();
    void allocate_memory(int* img_array);
    void deallocate_memory( );
    int* mat_to_oneD_array_parallel_parent();
    void oneD_array_to_mat(int* img_array);
    void semantic_segmentation(int* test_array);
    void oneD_array_to_mat(int* img_array, int cols, int rows, int channels, Mat* mlt1);
    int* mat_to_oneD_array_parallel_parent(vector<Mat>* matvector1, int* img_array);


};


#endif
