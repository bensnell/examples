// #include <torch/torch.h>
#include <torch/script.h> // One-stop header.

// #include <cmath>
// #include <cstdio>
#include <iostream>
#include <memory>
#include <chrono>
#include <math.h>

using namespace torch;

// Examples:
// https://pytorch.org/tutorials/advanced/cpp_export.html
// https://github.com/bensnell/examples/blob/cpp/cpp/dcgan/dcgan.cpp

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "loaded successfully\n";
   
      // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCUDA);
//   if (torch::cuda::is_available()) {
//     std::cout << "CUDA is available! Training on GPU." << std::endl;
//     device = torch::Device(torch::kCUDA);
//   }
    
    for (int i = 0; i < 6; i++) {
    
        // Parameters
        int batchSize = int(round(pow(10, i)));
        int seqLength = 128;
        int nFeatures = 1;

        // Now attempt to pass information through the module
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::randn({batchSize, seqLength, nFeatures}, device));

        // Execute the model and turn its output into a tensor.
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        at::Tensor output = module.forward(inputs).toTensor();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        
//         std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';

        uint64_t elapsedMS = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        
        std::cout << "Batch of size " << batchSize << " took " << float(elapsedMS)/1000.0 << " ms." << std::endl;
    }
}
