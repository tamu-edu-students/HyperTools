
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "../src/hypergpufunctions.cu"

using namespace std::chrono;
using namespace cv;
using namespace std;

int main (int argc, char *argv[]) {
       
    string file_name2="../../HyperImages/img1.tiff"; 
    
    HyperFunctionsGPU HyperFunctions1;
    HyperFunctions1.LoadImageHyper(file_name2);
    
    auto start = high_resolution_clock::now();
    
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    HyperFunctions1.mat_to_oneD_array_parallel_parent();
    HyperFunctions1.allocate_memory();
    HyperFunctions1.spec_sim_GPU();
    
    auto end = high_resolution_clock::now();
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    HyperFunctions1.DispSpecSim();
    cv::waitKey();
    
    HyperFunctions1.spec_sim_alg=1;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();

    HyperFunctions1.spec_sim_alg=2;    
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();
    
    // SAM - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=0;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
    end = high_resolution_clock::now();
    cout << "Proccess SAM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
    
    // SCM - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=1;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
    end = high_resolution_clock::now();
    cout << "Proccess SCM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
    
    // SID - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=2;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
    end = high_resolution_clock::now();
    cout << "Proccess SID Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // COS - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=3;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
    end = high_resolution_clock::now();
    cout << "Proccess COS Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // JM -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=4;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
    end = high_resolution_clock::now();
    cout << "Proccess JM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
    
    // City Block -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=5;
    HyperFunctions1.semantic_segmentation();
    HyperFunctions1.DispClassifiedImage();
    end = high_resolution_clock::now();
    cout << "Proccess City Block Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    HyperFunctions1.deallocate_memory();

  return 0;
}
