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

__global__ void mat_to_oneD_array_child(uchar* mat_array, int* img_array, int n, int start, int inc) ;

/*start of hybrid functions*/
__device__ void child_SAM(float *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__device__ void child_SCM(float *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__device__ void child_SID(float *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__device__ void child_cos(float *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__device__ void child_JM(float *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__device__ void child_cityblock(float *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__device__ void child_EuD(float *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;
__device__ void child_SID_SAM(float *out, int *img_array, int n, int num_layers, int* ref_spectrum) ;

__global__ void parent_control(float *out, int *img_array, int n, int num_layers, int* ref_spectrum, int sim_alg);



class HyperFunctionsGPU : public HyperFunctions {
public:


    int *d_img_array, *d_ref_spectrum; 
    float *d_out; 
    int* d_classified_img_array;
    int *img_array_base; 
    float *out;   
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
    void oneD_array_to_mat(float* img_array); 
    void oneD_array_to_mat(int* img_array, int cols, int rows, int channels, Mat* mlt1);
    void semantic_segmentation();
    void spec_sim_GPU();

};


#endif
