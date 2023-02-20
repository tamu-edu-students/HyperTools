#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "hyperfunctions.cpp"
#include "hypergpufunctions.cu"

using namespace cv;
using namespace std;


int main (int argc, char *argv[]) {

    HyperFunctionsGPU HyperFunctions1;
    string file_name2="../../HyperImages/img1.tiff";
    HyperFunctions1.LoadImageHyper1(file_name2);
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    
    int* test_array= HyperFunctions1.mat_to_oneD_array_parallel_parent(  );


    HyperFunctions1.allocate_memory(test_array);

    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();

    /*HyperFunctions1.spec_sim_alg=1;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();

    HyperFunctions1.spec_sim_alg=2;    
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();*/

    HyperFunctions1.deallocate_memory();

/*
  HyperFunctions1.semantic_segmentation(test_array);
    HyperFunctions1.DispClassifiedImage();
    cv::waitKey();
    
    HyperFunctions1.spec_sim_alg=1;
    HyperFunctions1.semantic_segmentation(test_array);
    HyperFunctions1.DispClassifiedImage();
    cv::waitKey();
    
    HyperFunctions1.spec_sim_alg=2;
    HyperFunctions1.semantic_segmentation(test_array);
    HyperFunctions1.DispClassifiedImage();
    cv::waitKey();*/

  return 0;
}


