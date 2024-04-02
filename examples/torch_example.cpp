

#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "../src/hyperfunctions.cpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <torch/torch.h>
#include <torch/script.h> 
#include <iostream>
#include <memory>
// #include <opencv2/opencv.hpp>


using namespace std;


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
    // std::cout << "Tuple size: " << output_tuple->elements().size() << std::endl;

    // for (size_t i = 0; i < output_tuple->elements().size(); ++i) {
    //     auto element = output_tuple->elements()[i];
    //     if (element.isTensor()) {
    //         // std::cout << "Element " << i << ": " << element.toTensor() << std::endl;
    //        torch::ScalarType dtype = element.toTensor().scalar_type();
    //         std::cout << "Element " << i << " data type: " << dtype << std::endl;
    //     }
    // }

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
    tensor = tensor.to(torch::kF32);
    // cv::Mat mat(cv::Size{tensor.size(1), tensor.size(0)}, CV_32FC1);
    cv::Mat mat(cv::Size{static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(0))}, CV_32F);
    std::memcpy((void*)mat.data, tensor.data_ptr(), sizeof(float)*tensor.numel());
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



    // cv::Mat img1 = cv::imread("../images/lena3.png");
    // cv::Mat img2 = cv::imread("../images/pca_result.png");
    HyperFunctions HyperFunctions1; 
    string file_name1="../../HyperImages/img1.tiff";
    HyperFunctions1.LoadImageHyper(file_name1);
    HyperFunctions1.false_img_b=2;
    HyperFunctions1.false_img_g=13;
    HyperFunctions1.false_img_r=31;
    HyperFunctions1.GenerateFalseImg();

    cv::Mat img1 = HyperFunctions1.false_img.clone();

    HyperFunctions1.false_img_b=78;
    HyperFunctions1.false_img_g=45;
    HyperFunctions1.false_img_r=20;
    HyperFunctions1.GenerateFalseImg();

    cv::Mat img2 = HyperFunctions1.false_img.clone();


cv::resize(img1, img1, cv::Size(224, 224), cv::INTER_LINEAR);
cv::resize(img2, img2, cv::Size(224, 224), cv::INTER_LINEAR);


    c10::intrusive_ptr<torch::ivalue::Tuple> output1 = run_model_on_image(img1, module);

    c10::intrusive_ptr<torch::ivalue::Tuple> output2 = run_model_on_image(img2, module);

    // Convert the output tensors to cv::Mat
    cv::Mat features1 = tensorToMat(output1->elements()[0].toTensor());
    cv::Mat features2 = tensorToMat(output2->elements()[0].toTensor());


    // // Convert the cv::Mat objects to grayscale
    // cv::cvtColor(features1, features1, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(features2, features2, cv::COLOR_BGR2GRAY);

    // // Convert the cv::Mat objects to CV_32F
    // features1.convertTo(features1, CV_32F);
    // features2.convertTo(features2, CV_32F);



    // Print the shapes of the feature matrices
    std::cout << "features1 shape: " << features1.rows << " x " << features1.cols << std::endl;
    std::cout << "features2 shape: " << features2.rows << " x " << features2.cols << std::endl;

    std::vector<cv::KeyPoint> keypoints1, keypoints2;



    // Assuming grid_size and resize_scale are defined
    int grid_size = 14;

    int resize_scale = 224;

    // Convert features to keypoints
    for (int i = 0; i < (resize_scale/14); i++) {
        for(int j = 0; j < (resize_scale/14); j++) {
            int row, col;

            row = i * grid_size + grid_size / 2;
            col = j * grid_size + grid_size / 2;
            
            keypoints1.push_back(cv::KeyPoint(cv::Point2f(col, row), 1));
        }
    }

    keypoints2 = keypoints1;

/*
   // print features size and type
    cout<<"features1 size and type "<<features1.size()<<" "<<features1.type()<<endl;
    cout<<"features2 size and type "<<features2.size()<<" "<<features2.type()<<endl;


    cv::Mat img1 = cv::imread("../images/lena3.png");
    cv::Mat img2 = cv::imread("../images/lena3.png");

    cv::resize(img1, img1, cv::Size(224, 224), cv::INTER_LINEAR);
    cv::resize(img2, img2, cv::Size(224, 224), cv::INTER_LINEAR);


 //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
 int minHessian = 400;
 cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
 std::vector<cv::KeyPoint> keypoints1, keypoints2;
 cv::Mat descriptors1, descriptors2;
 detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
 detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );

 */
 //-- Step 2: Matching descriptor vectors with a FLANN based matcher
 // Since SURF is a floating-point descriptor NORM_L2 is used
 cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
 std::vector< std::vector<cv::DMatch> > knn_matches;
//  matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
 matcher->knnMatch( features1, features2, knn_matches, 2 );

 //-- Filter matches using the Lowe's ratio test

//  cout<<"descriptor size and type "<<descriptors1.size()<<" "<<descriptors1.type()<<endl;
// cout<<"descriptor size and type "<<descriptors2.size()<<" "<<descriptors2.type()<<endl;
 const float ratio_thresh = 0.9f;
 std::vector<cv::DMatch> good_matches;
 for (size_t i = 0; i < knn_matches.size(); i++)
 {
 if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
 {
 good_matches.push_back(knn_matches[i][0]);
 }
 }

cout<<"good matches size "<<good_matches.size()<<endl;
 
 //-- Draw matches
 cv::Mat img_matches;
 // get keypoints from grid
 cv::drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//  -- Show detected matches
 cv::imshow("Good Matches", img_matches );
 cv::waitKey();

 
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