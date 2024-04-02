#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/providers.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <vector>

#include "../src/hyperfunctions.cpp"
// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
// #include <onnxruntime/core/providers/cpu/cpu_execution_provider.h>


using namespace cv;
using namespace std;

int main() {
    
    // Initialize the ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelLoadingExample");

    // Initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* model_path = "/workspaces/HyperTools/autoencoder_U20_model.onnx";

    
    // intialize hyperfunctions
    HyperFunctions HyperFunctions1; 
    // string file_name1="../../HyperImages/low_res_mlt.tiff";
    string file_name1="../../HyperImages/img1.tiff";
    HyperFunctions1.LoadImageHyper(file_name1);
    HyperFunctions1.false_img_b=65;
    HyperFunctions1.false_img_g=104;
    HyperFunctions1.false_img_r=163;
    HyperFunctions1.GenerateFalseImg();
    imshow("RGB Image",  HyperFunctions1.false_img);
    cv::waitKey(100);

    // 2048 = 2^11 = 32* 64
    // resize mlt 1 to 32*64
    vector<Mat> mlt1_32_64;
    for (int i = 0; i < HyperFunctions1.mlt1.size(); i++) {
        Mat mlt1_32_64_img;
        resize(HyperFunctions1.mlt1[i], mlt1_32_64_img, Size(64, 32), INTER_LINEAR);
        mlt1_32_64_img.convertTo(mlt1_32_64_img, CV_32FC1);
        mlt1_32_64_img = mlt1_32_64_img / 255.0f;
        mlt1_32_64.push_back(mlt1_32_64_img);
        // cout<<mlt1_32_64_img<<endl;
    }


    

    const void* img_data = &mlt1_32_64;

    // create custom image 
    cv::Mat mlt1_32_64_img;
    cv::merge(mlt1_32_64, mlt1_32_64_img);

    // cout mlt1_32_64 size 
    cout<<mlt1_32_64_img.size()<<endl;
    // cout number of channels
    cout<<mlt1_32_64_img.channels()<<endl;

    Mat false_img;
    vector<Mat> channels(3);
    channels[0] = mlt1_32_64[HyperFunctions1.false_img_b]; // b
    channels[1] = mlt1_32_64[HyperFunctions1.false_img_g]; // g
    channels[2] = mlt1_32_64[HyperFunctions1.false_img_r]; // r
    merge(channels, false_img);      // create new single channel image
    cv::Mat false_img_resized2;
    resize(false_img, false_img_resized2, Size(500,500),INTER_LINEAR);
    imshow("RGB Image Resize",  false_img_resized2);
    cv::waitKey(100);

    

    try {
        // Create an ONNX runtime session
        Ort::Session autoencoder_session(env, model_path, session_options);

        std::cout << "Successfully loaded model from " << model_path << std::endl;

        // Get the model metadata
        Ort::ModelMetadata model_metadata = autoencoder_session.GetModelMetadata();

        // Create a memory info object
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // // Get the input and output names
        // char* input_name = autoencoder_session.GetInputName(0, allocator);
        // std::vector<const char*> input_names = {input_name};

        // char* output_name = autoencoder_session.GetOutputName(0, allocator);
        // std::vector<const char*> output_names = {output_name};

        const char* input_name = "onnx::Gemm_0";
        const char* output_name = "38";

        std::vector<const char*> input_names = {input_name};
        std::vector<const char*> output_names = {output_name};

        // Create an input tensor
        // std::vector<float> input_tensor_values(2048 * 164 * 4); // batch size times number of channels in image times size of float in bytes 
        
        std::vector<float> input_tensor_values;
        for (auto& channel : mlt1_32_64) {
            input_tensor_values.insert(input_tensor_values.end(), channel.begin<float>(), channel.end<float>());
        }
        
        std::vector<int64_t> input_tensor_shape = {2048, 164};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

        // Run the model
        std::vector<Ort::Value> output_tensors = autoencoder_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

        // Get the output tensor
        Ort::Value& output_tensor = output_tensors.front();

        // Print the output values
        std::vector<int64_t> output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

        std::cout << "Output shape: "<<output_shape[0] << " " << output_shape[1] << std::endl;

        // for (int i = 0; i < output_shape[1]; i++) {
        //     std::cout << "Output value " << i << ": " << output_tensor.At<float>({0, i}) << std::endl;
        // }
        
        // Get the output tensor values
        float* floatarr = output_tensor.GetTensorMutableData<float>();

        // Convert the output tensor to a std::vector<cv::Mat>
        std::vector<cv::Mat> output_mats;
        for (int i = 0; i < output_shape[1]; ++i) {
            cv::Mat mat(32,64, CV_32F, floatarr + i * 32*64);
            output_mats.push_back(mat);
        }

        cout<<output_mats.size()<<endl;
        cout<<output_mats[0].size()<<endl;

        channels[0] = output_mats[HyperFunctions1.false_img_b]; // b
        channels[1] = output_mats[HyperFunctions1.false_img_g]; // g
        channels[2] = output_mats[HyperFunctions1.false_img_r]; // r
        // channels[0] = output_mats[0]; // b
        // channels[1] = output_mats[1]; // g
        // channels[2] = output_mats[2]; // r
        merge(channels, false_img);      // create new single channel image
        cv::Mat false_img_resized;
        resize(false_img, false_img_resized, Size(500,500),INTER_LINEAR);
        imshow("RGB Image Output",  false_img_resized);




    } catch (const Ort::Exception& exception) {
        std::cerr << "Failed to load model from " << model_path << ": " << exception.what() << std::endl;
        return -1;
    }


    
    cv::waitKey(0);


    return 0;
}