#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/providers.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

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

    const char* model_path = "/path/to/your/model.onnx";

    try {
        // Create an ONNX runtime session
        Ort::Session session(env, model_path, session_options);

        std::cout << "Successfully loaded model from " << model_path << std::endl;
    } catch (const Ort::Exception& exception) {
        std::cerr << "Failed to load model from " << model_path << ": " << exception.what() << std::endl;
        return -1;
    }

    return 0;
}