#include <torch/torch.h>
#include <iostream>

using namespace std;

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;


  // torch hub load 
  string REPO_NAME = "facebookresearch/dinov2"
  string MODEL_NAME = "dinov2_vitb14_reg"
}