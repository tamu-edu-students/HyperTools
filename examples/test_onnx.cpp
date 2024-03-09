#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/providers.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <vector>

// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
// #include <onnxruntime/core/providers/cpu/cpu_execution_provider.h>


// using namespace cv;
using namespace std;

int main() {
    // Initialize the ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelLoadingExample");

    // Initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* model_path = "/workspaces/HyperTools/autoencoder_U20_model.onnx";

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
        std::vector<float> input_tensor_values(2048 * 164 * 4); // batch size times number of channels in image times size of float in bytes 
        std::vector<int64_t> input_tensor_shape = {2048, 164};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

        // Run the model
        std::vector<Ort::Value> output_tensors = autoencoder_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

        // Get the output tensor
        Ort::Value& output_tensor = output_tensors.front();

        // Print the output values
        std::vector<int64_t> output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

        std::cout << "Output shape: "<<output_shape[0] << " " << output_shape[1] << std::endl;

        for (int i = 0; i < output_shape[1]; i++) {
            std::cout << "Output value " << i << ": " << output_tensor.At<float>({0, i}) << std::endl;
        }
        



    } catch (const Ort::Exception& exception) {
        std::cerr << "Failed to load model from " << model_path << ": " << exception.what() << std::endl;
        return -1;
    }


    



    return 0;
}