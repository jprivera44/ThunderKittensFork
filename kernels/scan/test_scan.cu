#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Simple test to verify our scan kernel works
template<typename T>
void print_tensor(const T* data, int batch, int dim, int seq_len, const char* name) {
    std::cout << name << " (first few elements):\n";
    for(int b = 0; b < std::min(batch, 2); b++) {
        for(int d = 0; d < std::min(dim, 4); d++) {
            for(int s = 0; s < std::min(seq_len, 8); s++) {
                int idx = b * dim * seq_len + d * seq_len + s;
                std::cout << data[idx] << " ";
            }
            std::cout << "...\n";
        }
        if(b == 0 && batch > 1) std::cout << "...\n";
    }
}

int main() {
    // Test dimensions
    const int batch = 2;
    const int dim = 64;
    const int seq_len = 128;
    const int dstate = 16;
    
    // Allocate host memory
    std::vector<float> h_u(batch * dim * seq_len);
    std::vector<float> h_delta(batch * dim * seq_len);
    std::vector<float> h_A(dim * dstate);
    std::vector<float> h_B(batch * dstate * seq_len);
    std::vector<float> h_C(batch * dstate * seq_len);
    std::vector<float> h_out(batch * dim * seq_len);
    
    // Initialize with simple values for testing
    for(int i = 0; i < h_u.size(); i++) h_u[i] = 0.1f;
    for(int i = 0; i < h_delta.size(); i++) h_delta[i] = 0.01f;
    for(int i = 0; i < h_A.size(); i++) h_A[i] = -0.1f; // negative for stability
    for(int i = 0; i < h_B.size(); i++) h_B[i] = 0.1f;
    for(int i = 0; i < h_C.size(); i++) h_C[i] = 0.1f;
    
    // For now, just test compilation
    std::cout << "Scan kernel test setup complete!\n";
    std::cout << "Dimensions: batch=" << batch << ", dim=" << dim 
              << ", seq_len=" << seq_len << ", dstate=" << dstate << "\n";
    
    print_tensor(h_u.data(), batch, dim, seq_len, "Input u");
    
    return 0;
}