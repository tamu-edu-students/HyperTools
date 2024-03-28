

#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
// #include "../src/hyperfunctions.cpp"

#include <torch/torch.h>
#include <torch/script.h> 
#include <iostream>
#include <memory>
// #include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

/*
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
}*/

int main() {
  // torch::Tensor tensor = torch::rand({2, 3});
  // std::cout << tensor << std::endl;




  // Load the TorchScript model
  torch::jit::script::Module module;
  
    try {
        module = torch::jit::load("/workspaces/HyperTools/dino_vitb16_model_test.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }


   cv::Mat test_img =  cv::imread("../images/lena3.png");
   // resize to 224, 224
    cv::resize(test_img,test_img,Size(224,224),INTER_LINEAR); 


//    cv::imshow("test", test_img);

//    cv::waitKey();


// // Convert the image to float and normalize to [0, 1]
test_img.convertTo(test_img, CV_32F, 1.0 / 255);

// // Create a tensor from the image
auto test_tensor = torch::from_blob(test_img.data, {1, test_img.rows, test_img.cols, 3});

// // Change the dimensions from BxHxWxC to BxCxHxW
test_tensor = test_tensor.permute({0, 3, 1, 2});

// // Create a vector of inputs
std::vector<torch::jit::IValue> inputs;
inputs.push_back(test_tensor);



    // Create a vector of inputs.
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}));

    // // Execute the model and turn its output into a tensor.
    // at::Tensor output = module.forward(inputs).toTensor();
    // // std::cout << "here " <<output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    // cout<<"test "<<endl;
    // cout<<output<<endl;


// Execute the model and get the output.
auto output = module.forward(inputs);

// Check if the output is a tuple
if (output.isTuple()) {
    auto output_tuple = output.toTuple();

    // Print the size of the tuple
    std::cout << "Tuple size: " << output_tuple->elements().size() << std::endl;

    // Access the elements of the tuple
    for (size_t i = 0; i < output_tuple->elements().size(); ++i) {
        auto element = output_tuple->elements()[i];

        // Check if the element is a tensor and print it
        if (element.isTensor()) {
            std::cout << "Element " << i << ": " << element.toTensor() << std::endl;
        }
    }
} else {
    std::cerr << "Output is not a tuple\n";
}



    // cv::Mat image = test_img;
    // float smaller_edge_size = 224.0; // Replace with the actual size
    // int patch_size = 16; // Replace with the actual patch size
    // auto [image_tensor, grid_size] = prepare_image(image, smaller_edge_size, patch_size);

    // // Extract features from the image
    // torch::Device device(torch::kCPU); // Use torch::kCUDA if you have a GPU
    // bool half_precision = false; // Set to true if you want to use half precision
    // std::vector<float> features = extract_features(image_tensor, module, device, half_precision);

    // // Print the features
    // for (float feature : features) {
    //     std::cout << feature << ' ';
    // }
    // std::cout << '\n';



    std::cout << "Model loaded\n";
}