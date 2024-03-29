

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

c10::intrusive_ptr<torch::ivalue::Tuple> run_model_on_image(const cv::Mat& input_img, torch::jit::script::Module& module) {
    cv::Mat img;
    input_img.copyTo(img);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_LINEAR);

    img.convertTo(img, CV_32F, 1.0 / 255);

    auto tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3});
    if (tensor.dim() == 4) {
        tensor = tensor.permute({0, 3, 1, 2});
    } else {
        std::cerr << "Tensor does not have 4 dimensions\n";
        return nullptr;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);

    auto output = module.forward(inputs);

    if (!output.isTuple()) {
        std::cerr << "Output is not a tuple\n";
        return nullptr;
    }

    auto output_tuple = output.toTuple();
    std::cout << "Tuple size: " << output_tuple->elements().size() << std::endl;

    for (size_t i = 0; i < output_tuple->elements().size(); ++i) {
        auto element = output_tuple->elements()[i];
        if (element.isTensor()) {
            // std::cout << "Element " << i << ": " << element.toTensor() << std::endl;
        }
    }

    // Return the entire output tuple
    return output_tuple;
}


cv::Mat tensorToMat(torch::Tensor tensor, bool is_gray=false) {
    tensor = tensor.detach().cpu();
    tensor = tensor.squeeze();
    if (tensor.dim() == 3) {
        tensor = tensor.permute({1, 2, 0});
    } else if (tensor.dim() != 2) {
        std::cerr << "Tensor does not have 2 or 3 dimensions\n";
        return cv::Mat();
    }
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    cv::Mat mat(cv::Size{tensor.size(1), tensor.size(0)}, is_gray ? CV_8UC1 : CV_8UC3);
    std::memcpy((void*)mat.data, tensor.data_ptr(), sizeof(torch::kU8)*tensor.numel());
    return mat.clone();
}

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


    c10::intrusive_ptr<torch::ivalue::Tuple> output1 = run_model_on_image(test_img, module);


    c10::intrusive_ptr<torch::ivalue::Tuple> output2 = run_model_on_image(test_img, module);



    // Convert the output tensors to cv::Mat
    cv::Mat features1 = tensorToMat(output1->elements()[0].toTensor());
    cv::Mat features2 = tensorToMat(output2->elements()[0].toTensor());

    // Convert the cv::Mat objects to CV_32F
    features1.convertTo(features1, CV_32F);
    features2.convertTo(features2, CV_32F);



    // Print the shapes of the feature matrices
    std::cout << "features1 shape: " << features1.rows << " x " << features1.cols << std::endl;
    std::cout << "features2 shape: " << features2.rows << " x " << features2.cols << std::endl;


/*
// Ensure that features1 and features2 have the same number of rows
if (features1.rows < features2.rows) {
    features2 = features2.rowRange(0, features1.rows);
} else if (features1.rows > features2.rows) {
    features1 = features1.rowRange(0, features2.rows);
}

// Create the descriptor matcher
cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

// Use the descriptor matcher to find the matches
std::vector<cv::DMatch> matches;
matcher->match(features1, features2, matches);

// Print the number of matches
std::cout << "Number of matches: " << matches.size() << std::endl;


*/

   /*
   // resize to 224, 224 to match the input size of the model
    cv::resize(test_img,test_img,cv::Size(224,224),cv::INTER_LINEAR); 

    //visualize image 
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

    // Execute the model and get the output.
    // should be of float size 1,256,768  (256 is number of patches, 768 is descriptor length)
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


*/

    // Create a vector of inputs.
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}));

    // // Execute the model and turn its output into a tensor.
    // at::Tensor output = module.forward(inputs).toTensor();
    // // std::cout << "here " <<output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    // cout<<"test "<<endl;
    // cout<<output<<endl;



    



    


    std::cout << "Model loaded\n";
}