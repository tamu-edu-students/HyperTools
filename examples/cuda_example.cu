
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
    
    // initialize time, spectral database, and allocate memory on host and device
    auto start = high_resolution_clock::now();
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    HyperFunctions1.mat_to_oneD_array_parallel_parent();
    HyperFunctions1.allocate_memory();
    auto end = high_resolution_clock::now();
    cout << "Initialization Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
       
    // perform semantic segmentation
    for (int spec_sim_val=0; spec_sim_val<15; spec_sim_val++)
    {
        HyperFunctions1.spec_sim_alg=spec_sim_val;
        start = high_resolution_clock::now();
        HyperFunctions1.semantic_segmentation(); 
        HyperFunctions1.DispClassifiedImage();
        end = high_resolution_clock::now();
        cout << spec_sim_val<< " Proccess Classification:" << endl;
        cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
        cv::waitKey();
    }

    // SAM - Classification
    // slower the first time not sure why so running twice
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=0;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess SAM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    
    // SAM - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=0;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess SAM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // SCM - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=1;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess SCM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
    
    // SID - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=2;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess SID Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // Euclidean -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=3;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess Euclidean Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
    
    // chi square - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=4;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess chi square Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
  
    // COS - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=5;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess COS Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // City Block -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=6;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess City Block Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
    
    // JM -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=7;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess JM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
    
    // ns3
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=8;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess ns3 Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // JM-SAM -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=9;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess JM-SAM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // SCA -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=10;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess SCA Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();
    
    // SID-SAM -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=11;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess SID-SAM Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // SID-SCA -  Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=12;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess SID-SCA Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // Hellinger - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=13;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess Hellinger Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();

    // Canberra - Classification
    start = high_resolution_clock::now();
    HyperFunctions1.spec_sim_alg=14;
    HyperFunctions1.spec_sim_GPU();
    HyperFunctions1.DispSpecSim();
    end = high_resolution_clock::now();
    cout << "Proccess Canberra Classification:" << endl;
    cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    cv::waitKey();


    HyperFunctions1.deallocate_memory();

  return 0;
}
