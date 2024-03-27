#include <torch/torch.h>
#include <torch/script.h> 
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


std::vector<float> extract_features(torch::Tensor image_tensor, torch::jit::script::Module& model, torch::Device device, bool half_precision) {
    // Add a new dimension at the beginning of the tensor and set the tensor to the device
    if (half_precision) {
        image_tensor = image_tensor.unsqueeze(0).to(torch::kHalf).to(device);
    } else {
        image_tensor = image_tensor.unsqueeze(0).to(device);
    }

    // Pass the tensor through the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image_tensor);
    torch::Tensor output = model.forward(inputs).toTensor();

    // Convert the output to a CPU tensor and then to a standard array
    output = output.squeeze().to(torch::kCPU);
    cout<<"output size "<<output.sizes()<<endl;

    std::vector<float> output_array(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());

    return output_array;
}

std::pair<torch::Tensor, std::pair<int, int>> prepare_image(cv::Mat& image, float smaller_edge_size, int patch_size) {
    // Apply the transformation to the image
    // This is a placeholder. You'll need to replace this with the actual transformation.
    cv::resize(image, image, cv::Size(), smaller_edge_size / image.cols, smaller_edge_size / image.rows);

    // Convert the OpenCV image to a Torch tensor
    torch::Tensor image_tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kByte);
    image_tensor = image_tensor.permute({2, 0, 1}); // C x H x W

    // Crop the image to dimensions that are a multiple of the patch size
    int height = image_tensor.size(1);
    int width = image_tensor.size(2);
    int cropped_height = height - height % patch_size;
    int cropped_width = width - width % patch_size;
    image_tensor = image_tensor.slice(1, 0, cropped_height).slice(2, 0, cropped_width);

    // Calculate the grid size
    int grid_height = cropped_height / patch_size;
    int grid_width = cropped_width / patch_size;

    return {image_tensor.clone(), {grid_height, grid_width}};
}

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


  //  imshow("test", test_img);

  //  waitKey();


    // // Create a vector of inputs.
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}));

    // // Execute the model and turn its output into a tensor.
    // at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << "here " <<output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    // cout<<"test "<<endl;
    // cout<<output<<endl;

    cv::Mat image = test_img;
    float smaller_edge_size = 224.0; // Replace with the actual size
    int patch_size = 16; // Replace with the actual patch size
    auto [image_tensor, grid_size] = prepare_image(image, smaller_edge_size, patch_size);

    // Extract features from the image
    torch::Device device(torch::kCPU); // Use torch::kCUDA if you have a GPU
    bool half_precision = false; // Set to true if you want to use half precision
    std::vector<float> features = extract_features(image_tensor, module, device, half_precision);

    // // Print the features
    // for (float feature : features) {
    //     std::cout << feature << ' ';
    // }
    // std::cout << '\n';



    std::cout << "Model loaded\n";
}