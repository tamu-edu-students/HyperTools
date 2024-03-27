#include <torch/torch.h>
#include <torch/script.h> 
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

int main() {
  // torch::Tensor tensor = torch::rand({2, 3});
  // std::cout << tensor << std::endl;




  // Load the TorchScript model
  torch::jit::script::Module module;
  
    try {
        module = torch::jit::load("/workspaces/HyperTools/dino_vitb16_model.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }


   Mat test_img =  imread("../images/lena3.png");
   // resize to 224, 224
    cv::resize(test_img,test_img,Size(224,224),INTER_LINEAR); 


   imshow("test", test_img);

   waitKey();


    std::cout << "Model loaded\n";
}