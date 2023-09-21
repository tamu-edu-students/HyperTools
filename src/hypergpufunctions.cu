#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include "hyperfunctions.cpp"
#include "hyperfunctions.h"
#include "hypergpufunctions.h"
#include "ctpl.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

/**
 * Measures spectral similarity between our image and a reference spectrum
 * with the Spectral Angle Mapper algorithm using concurrent GPU threads.
 * 
 * Formula : 
 * r^2 = sum of squared reference spectra values, 
 * i * r = sum of image spectra values * corresponding reference spectra values
 * i^2 = sum of squared image spectra values. 
 * 
 * returns : inverse cos of i * r / (r^2 * i^2). 
 * 
*/
__global__ void img_test_multi_thread_SAM(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) 
{
    
    // parallelize tasks
    // pixels are stored with all pixel values next to each other for the layers    
    // n is number of pixels 
    // blockID : block index within the grid
    // blockDim : how many threads per block
    // threadIdx : thread index within the block 

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum1=0, sum2=0;
    float sum3 = 0;
    for (int a=0; a<num_layers-1; a++) {
        sum3+=ref_spectrum[a] *ref_spectrum[a]; //sum of squared reference spectra values
    }
    if (tid < n){
        int offset=tid*num_layers; //calculating which index in the image array the values for threadID pixel start at
        for (int a=0; a<num_layers-1; a++) //iterating through spectra layers for that pixel
        {
            sum1+=img_array[offset+a]*ref_spectrum[a]; //image spectra values * corresponding referencec spectrum values
            sum2+=img_array[offset+a]*img_array[offset+a]; //Squared image spectra values
        }
        
        if (sum1<=0 || sum2<=0 || sum3<=0 )
        {
            out[tid] =255; // set to white due to an error
        }
        else
        {
            float temp1= sum1/(sqrt(sum2)*sqrt(sum3));
            double alpha_rad=acos(temp1);
            out[tid] =(int)((double)alpha_rad*(double)255/(double)3.14159) ;
        }
    }
    
}

/**
 * Measures spectral similarity between our image and a reference spectrum
 * with the Spectral Information Divergence algorithm using concurrent GPU threads.
 * 
 * Formula : 
 * q = probability array for our image, divide image array by sum of image array values.
 * p = probability array for reference spectrum, divide reference spectrum array by sum of reference spectrum array values
 * 
 * returns : sum of p[i] * log(p[i]/q[i]) + sum of q[i] * log(q[i]/p[i]) for 0 <= i < num_layers-1 
 * 
*/

__global__ void img_test_multi_thread_SID(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) 
{
    // parallelize tasks
    
    // pixels are stored with all pixel values next to eachother for the layers    
    // n is number of pixels 
    // blockID : block index within the grid
    // blockDim : how many threads per block
    // threadIdx : thread index within the block 

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum1=0, sum2=0, ref_sum=0, pix_sum=0;

    if (tid < n){
        int offset=tid*num_layers;
        for (int a=0; a<num_layers-1; a++)
        {
            if (ref_spectrum[a]<1){ref_spectrum[a]+=1;}
            if (img_array[offset+a]<1){img_array[offset+a]+=1;}              
            ref_sum+= ref_spectrum[a] ;
            pix_sum+= img_array[offset+a];
        }
        
        // error handling to avoid division by zero
        if (ref_sum<1){ref_sum+=1;}
        if (pix_sum<1){pix_sum+=1;}
        
        float ref_new[300], pix_new[300];
        
        for (int a=0; a<num_layers-1; a++)
        {
            ref_new[a]=ref_spectrum[a] / ref_sum ; //probability distribution for reference spectrum
            pix_new[a]=img_array[offset+a]/pix_sum; //probabiltiy distribution for our image
            // error handling to avoid division by zero
        }
        
        for (int a=0; a<num_layers-1; a++)
        {
            sum1+= ref_new[a]*log(ref_new[a]/pix_new[a]);
            sum2+= pix_new[a]*log(pix_new[a]/ref_new[a]);
        }        

        // need to normalize the results better here
        out[tid] =(sum1+sum2) *60;
        if (out[tid]>255){out[tid]=255;}

    }
    
}

/**
 * 
 * Spectral Corellation Mapper function for spectral similarity analysis
 * 
 * 
 * 
*/
__global__ void img_test_multi_thread_SCM(int *out, int *img_array, int n, int num_layers, int* ref_spectrum) 
{ 
    // parallelize tasks
    // pixels are stored with all pixel values next to eachother for the layers    
    // n is number of pixels 

    // blockID : block index within the grid
    // blockDim : how many threads per block
    // threadIdx : thread index within the block 

    int tid = blockIdx.x * blockDim.x + threadIdx.x; //unique thread ID
    float sum1=0, sum2=0, sum3=0, mean1=0, mean2=0;
    if (tid < n){
        int offset=tid*num_layers;

        for (int a=0; a<num_layers-1; a++)
        {
            mean1+=((float)1/(float)(num_layers-1)* (float)img_array[offset+a])  ;
            mean2+=((float)1/(float)(num_layers-1)* (float)ref_spectrum[a]) ;
        }

        for (int a=0; a<num_layers-1; a++)
        {
            sum1+=(img_array[offset+a]-mean1)*(ref_spectrum[a]-mean2) ;
            sum2+=(img_array[offset+a]-mean1)*(img_array[offset+a]-mean1);
            sum3+=(ref_spectrum[a]-mean2)*(ref_spectrum[a]-mean2);
        }        
        if (sum2<=0 || sum3<=0 )
        {
            out[tid] =255; // set to white due to an error
        }
        else
        {
            float temp1= sum1/(sqrt(sum2)*sqrt(sum3));
            double alpha_rad=acos(temp1);
            out[tid] =(int)((double)alpha_rad*(double)255/(double)3.14159) ;
        }
    }
}

/**
 * Calls the multithreaded spectral similarity algorithms, based on the variable spec_sim_alg, set in
    hyperfunctions.cpp.
 * Retrieves output in "out", then calls oneD_array_to_mat(out) to convert out into the OPENCV matrix "spec_simil_img".
*/

void HyperFunctionsGPU::spec_sim_GPU() {

    if (spec_sim_alg == 0) { //running the multithreaded algorithms
        img_test_multi_thread_SAM<<<grid_size,block_size>>>(d_out, d_img_array, N_size, num_lay, d_ref_spectrum);
    } else if (spec_sim_alg == 1) {
        img_test_multi_thread_SCM<<<grid_size,block_size>>>(d_out, d_img_array, N_size, num_lay, d_ref_spectrum);
    } else if (spec_sim_alg == 2) {
        img_test_multi_thread_SID<<<grid_size,block_size>>>(d_out, d_img_array, N_size, num_lay, d_ref_spectrum);
    }

    cudaDeviceSynchronize();
    cudaMemcpyAsync(out, d_out, sizeof(int) * N_size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();

    this->oneD_array_to_mat(out);   
}

void HyperFunctionsGPU::deallocate_memory() 
{
    cudaFree(d_ref_spectrum); cudaFree(d_out); cudaFreeHost(out);
}

/* allocating CUDA memory. 
* In cuda, grids contain blocks of threads, which are used for parallel computations. 
* One pixel in img_array needs one thread for a computation, so threads = number of pixels. 
* Set the number of threads per block to be 512 - this is the maximum threads per block for older GPUs.
* Grid size is the number of of blocks, given by the number of threads / block size + 1. 
*/
void HyperFunctionsGPU::allocate_memory() {
    N_points=mlt1[1].rows*mlt1[1].cols*mlt1.size(); 
    N_size=mlt1[1].rows*mlt1[1].cols;    
    num_lay=  mlt1.size();
    block_size = 512;
    grid_size = ((N_points + block_size) / block_size); 
    
    int tmp_len1=reference_spectrums[ref_spec_index].size(); 

    cudaHostAlloc ((void**)&out, sizeof(int) * N_size, cudaHostAllocDefault);
    cudaMalloc((void**)&d_out, sizeof(int) * N_size);
    cudaMalloc((void**)&d_ref_spectrum, sizeof(int) * tmp_len1);

    //allocating memory on the GPU device

    ref_spectrum=new int[tmp_len1];
    for (int i=0;i<reference_spectrums[ref_spec_index].size();i++)
    {
        ref_spectrum[i] = reference_spectrums[ref_spec_index][i]; 
        //converting the 2-D array of reference spectrums into one-D for CUDA processing
    }
    cudaMemcpy(d_ref_spectrum, ref_spectrum, sizeof(int) * tmp_len1, cudaMemcpyHostToDevice);
    delete[] ref_spectrum;
    //copying our existing memory that holds image data and reference spectrum data to the 
    //allocated memory on the GPU
}

/**
 * Converts one-D array to 1 channel 8 bit OPENCV matrix. 
 * Used in manipulating spectral similarity data. 
*/
void HyperFunctionsGPU::oneD_array_to_mat(int* img_array)
{
    spec_simil_img = cv::Mat(mlt1[1].rows, mlt1[1].cols, CV_32SC1, img_array);
    spec_simil_img.convertTo(spec_simil_img, CV_8UC1); //converting to 8 bit unsigned, 1 channel. 
}

/**
 * Converting a one-D array to a OPENCV matrix. 
 * This matrix will has three channels per point which store RGB color values. 
*/

void HyperFunctionsGPU::oneD_array_to_mat(int* img_array, int cols, int rows, int channels, Mat* mlt1)
{

    *mlt1 = cv::Mat(mlt1->rows, mlt1->cols, CV_32SC3, img_array);
    mlt1->convertTo(*mlt1, CV_8UC3);
}


__global__ void mat_to_oneD_array_child(uchar* mat_array, int* img_array, int n, int start, int inc) 
{ 

    // blockID : block index within the grid
    // blockDim : how many threads per block
    // threadIdx : thread index within the block 
    int tid = blockIdx.x * blockDim.x + threadIdx.x; //unique thread ID
    if (tid < n){
       img_array[tid * inc + start] = mat_array[tid];
    }
}

void HyperFunctionsGPU::mat_to_oneD_array_parallel_parent()
{
    int array_size=mlt1[1].rows*mlt1[1].cols*mlt1.size();    
    uchar* d_mat_array; 
    int sz = mlt1[1].rows * mlt1[1].cols;
    cudaHostAlloc ((void**)&d_mat_array, sizeof(uchar) * sz, cudaHostAllocDefault);
    cudaMalloc((void**)&d_img_array, sizeof(int) * array_size);
    int grid_size1 = (sz + block_size) / block_size;
    for (int i = 0; i < mlt1.size(); i++) {
        uchar* mat_array = (uchar*)mlt1[i].data;
        cudaMemcpyAsync(d_mat_array, mat_array, sizeof(uchar) * sz, cudaMemcpyHostToDevice);
        mat_to_oneD_array_child<<<grid_size1, block_size>>>(d_mat_array, d_img_array, sz, i, mlt1.size());
    }
    cudaFreeHost(d_mat_array);  
}

void HyperFunctionsGPU::mat_to_oneD_array_parallel_parent(vector<Mat>* matvector1)
{
    vector<Mat> matvector = *matvector1;
    uchar* d_mat_array;
    int array_size=matvector[1].rows*matvector[1].cols*matvector.size();  
    int sz = matvector[1].rows * matvector[1].cols;
    cudaHostAlloc ((void**)&d_mat_array, sizeof(uchar) * sz, cudaHostAllocDefault);
    cudaMalloc((void**)&d_classified_img_array, sizeof(int) * array_size);
    block_size = 512;
    int grid_size1 = (sz + block_size) / block_size;
    for (int i = 0; i < matvector.size(); i++) {
        uchar* mat_array = (uchar*)matvector[i].data;
        cudaMemcpy(d_mat_array, mat_array, sizeof(uchar) * sz, cudaMemcpyHostToDevice);
        mat_to_oneD_array_child<<<grid_size1, block_size>>>(d_mat_array, d_classified_img_array, sz, i, matvector.size());
    }
    cudaFreeHost(d_mat_array);  
}

/**
 * Fills in the "out" array, which holds RGB values based on which reference spectra a pixel in the image is most similar to. 
*/

__global__ void img_test_classifier(int *out, int *img_array, int num_pixels, int num_spectrums, int* color_info, int classification_threshold ) 
{
    // blockID : block index within the grid
    // blockDim : how many threads per block
    // threadIdx : thread index within the block 

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_pixels){
        //out[tid] =img_array[tid];
        int offset=tid*num_spectrums; //where the spectral similarity image starts for a pixel. 
        int low_val;
        for (int a=0; a<num_spectrums;a++) //iterating through all the spectral similarity scores for a pixel
        {
            if (a==0) 
            //Setting the initial lowest value. Goal is to look through all the spectral
            //similarity values for a pixel and find the lowest score. The reference
            //spectra that yields the lowest score will be the most similar.  
            {
                low_val=img_array[offset];
                if (low_val<=classification_threshold)
                {
                out[tid*3]=color_info[a+0];
                out[tid*3+1]=color_info[a+1];  
                out[tid*3+2]=color_info[a+2];
                }
                else
                {
                out[tid*3]=0;
                out[tid*3+1]=0;  
                out[tid*3+2]=0;
                }
            }
            else if (img_array[offset+a]<low_val && img_array[offset+a]<= classification_threshold) 
            //looking for a new minimum. If found, set the color channels of the RGB image to the color corresponding to the reference spectra. 
            {
                out[tid*3]=color_info[a*3+0];
                out[tid*3+1]=color_info[a*3+1];
                out[tid*3+2]=color_info[a*3+2];                
            }            
           
        }
    }   
}

/**
 * Takes hyperspectral data and gives each pixel a color based on which reference spectra
 * it is most similar to. 
 * 
*/

void HyperFunctionsGPU::semantic_segmentation() {
    ref_spec_index = 0;
    int tmp_len1 = reference_spectrums[0].size();
    vector<Mat> similarity_images;
    int* ref_spectrum = new int[tmp_len1];

    for (int i=0;i<reference_spectrums[ref_spec_index].size();i++)
    {
        ref_spectrum[i] = reference_spectrums[ref_spec_index][i]; 
    }
    cudaMemcpy(d_ref_spectrum, ref_spectrum, sizeof(int) * tmp_len1, cudaMemcpyHostToDevice);

    this->spec_sim_GPU(); 
    //performs spectral similarity, comparing each pixel in our image array to the first reference spectra. 
    similarity_images.push_back(spec_simil_img);  
    for (int i = 1; i < reference_spectrums.size(); i++) { //loop to iterate through all the reference spectras. 
        for (int j=0; j < reference_spectrums[i].size(); j++) {
            ref_spectrum[j] = reference_spectrums[i][j]; //updating the reference spectrum we are comparing our pixels to. 
        }
        //updating the memory allocated on the GPU that stores the current reference spectra
        cudaMemcpy(d_ref_spectrum, ref_spectrum, sizeof(int) * tmp_len1, cudaMemcpyHostToDevice); 
        this->spec_sim_GPU();
        similarity_images.push_back(spec_simil_img);
    }

    int array_size2=similarity_images[1].rows*similarity_images[1].cols*similarity_images.size();  
    //converting the vector of matrices that store the similarity values for each reference spectrum into a 1-D array

    mat_to_oneD_array_parallel_parent(&similarity_images);

    int *d_color_info, *d_out2, *out2;
    int N_size_sim = similarity_images[1].rows*similarity_images[1].cols*3; 
    int N_points_sim = similarity_images[1].rows*similarity_images[1].cols*similarity_images.size(); 
    int grid_size_sim = ((N_points + block_size) / block_size);
    int tmp_len2 = color_combos.size()*3;


    /**
     * out2 : 1-d array that represents matrix with 3 channels, to store R G and B values. 
     * d_clasified_img_array : 1-d array containing spectra similarity values 
     * d_color_info : holds reference colors that will be used to colorize our final image
     * 
    */
   
    cudaHostAlloc ((void**)&out2, sizeof(int) *N_size_sim, cudaHostAllocDefault); 
    cudaMalloc((void**)&d_out2, sizeof(int) * N_size_sim);
    int temp_val=reference_colors.size() * 3;
    cudaMalloc((void**)&d_color_info, sizeof(int) * temp_val);

    int* reference_colors_c = new int[reference_colors.size() * 3];

    //converting reference_colors into a 1-d array
    for (int i = 0; i < reference_colors.size(); i++) {
        reference_colors_c[i*3] = reference_colors[i][0];
        reference_colors_c[i*3+1] = reference_colors[i][1];
        reference_colors_c[i*3+2] = reference_colors[i][2];
    }

    cudaMemcpy(d_color_info, reference_colors_c, sizeof(int) * temp_val, cudaMemcpyHostToDevice);
    //multi-threaded function to find the most similar spectra for a pixel and color it based on the color assigned to that spectra
    img_test_classifier<<<grid_size_sim,block_size>>>(d_out2, d_classified_img_array, N_size_sim/3, similarity_images.size(), d_color_info,classification_threshold);

    /**
     * copying the color image into out2, and converting that into a OPENCV matrix. 
    */

    cudaMemcpy(out2, d_out2, sizeof(int) * N_size_sim, cudaMemcpyDeviceToHost);
   
    Mat test_img2(similarity_images[1].rows, similarity_images[1].cols, CV_8UC3, Scalar(0,0,0)); 
    oneD_array_to_mat(out2, similarity_images[1].cols,similarity_images[1].rows,3, &test_img2);
    classified_img = test_img2;

    cudaFree(d_color_info);
    cudaFree(d_out2);
    cudaFree(d_classified_img_array);
    cudaFreeHost(out2);
    delete[] reference_colors_c;
    delete[] ref_spectrum;

}


