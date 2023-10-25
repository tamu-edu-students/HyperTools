#if !defined(HYPERGPUFUNCTIONS_H)
#define HYPERGPUFUNCTIONS_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "hyperfunctions.h"

using namespace std;
using namespace cv;

__global__ void img_test_classifier(int *out, int *img_array, int num_pixels, int num_spectrums, int* color_info ) ;
__global__ void img_test_multi_thread_SAM(int *out, int *img_array, int n, int num_layers, int* ref_spectrum, int sum) ;
__global__ void img_test_multi_thread_SCM(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__global__ void img_test_multi_thread_SID(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__global__ void img_test_multi_thread_cos(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__global__ void img_test_multi_thread_JM(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__global__ void mat_to_oneD_array_child(uchar* mat_array, int* img_array, int n, int start, int inc) ;
__global__ void img_test_multi_thread_cityblock(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__global__ void img_test_multi_thread_EuD(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;


class HyperFunctionsGPU : public HyperFunctions {
public:


    int *d_img_array, *d_out, *d_ref_spectrum; 
    int* d_classified_img_array;
    int *img_array_base; 
    int *out;   
    int *ref_spectrum;

    int grid_size;
    int N_points;
    int N_size;    
    int num_lay;

    int block_size = 512;

    void allocate_memory();
    void deallocate_memory();
    void mat_to_oneD_array_parallel_parent();
    void mat_to_oneD_array_parallel_parent(vector<Mat>* matvector1);
    void oneD_array_to_mat(int* img_array); 
    void oneD_array_to_mat(int* img_array, int cols, int rows, int channels, Mat* mlt1);
    void semantic_segmentation();
    void spec_sim_GPU();

};


#endif
