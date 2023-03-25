#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "../src/hypergpufunctions.cu"

using namespace std::chrono;
using namespace cv;
using namespace std;

int main (int argc, char *argv[]) {
    auto start = high_resolution_clock::now();
    int *prt;
    cudaMalloc(&prt, 0);
    cudaFree(prt);
    
    HyperFunctionsGPU HyperFunctions1;

    string file_name2="../../HyperImages/img1.tiff";
    HyperFunctions1.LoadImageHyper1(file_name2);
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    
    HyperFunctions1.mat_to_oneD_array_parallel_parent();
    
    HyperFunctions1.allocate_memory();
    HyperFunctions1.spec_sim_GPU();

    HyperFunctions1.DispSpecSim();
    auto end = high_resolution_clock::now();
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds";

    cv::waitKey();
    
    HyperFunctions1.spec_sim_alg=1;

    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();

    cv::waitKey();

    HyperFunctions1.spec_sim_alg=2;    
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();
    
    //HyperFunctions1.deallocate_memory();
    //HyperFunctions1.mat_to_oneD_array_parallel_parent();

    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();

    cv::waitKey();
    HyperFunctions1.spec_sim_alg=1;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
    cv::waitKey();
    
    HyperFunctions1.spec_sim_alg=2;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
    cv::waitKey();

    HyperFunctions1.deallocate_memory();

  return 0;
}


