#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>


#define SIZE (100 * 1024 * 1024)

void check_result(const char* a, const char* b, size_t size) {
    if (memcmp(a, b, size) == 0) {
        std::cout << "OK\n";
    }
    else {
        std::cout << "ERROR\n";
    }
}

double bandwidth(size_t bytes, double time_sec) {
    return (bytes / (1024.0 * 1024 * 1024)) / time_sec;
}

int main() {
    size_t size = SIZE;

    // host
    char* h_a = (char*)malloc(size);
    char* h_b = (char*)malloc(size);

    // pinned 
    char* h_pinned_a;
    char* h_pinned_b;
    cudaMallocHost((void**)&h_pinned_a, size);
    cudaMallocHost((void**)&h_pinned_b, size);

    // device
    char* d_a;
    char* d_b;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);

    // инициализация
    for (size_t i = 0; i < size; i++) {
        h_a[i] = i % 256;
        h_pinned_a[i] = i % 256;
    }

    std::cout << “Host -> Host \n";
    auto t1 = std::chrono::high_resolution_clock::now();
    memcpy(h_b, h_a, size);
    auto t2 = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Bandwidth: " << bandwidth(size, time) << " GB/s\n";
    check_result(h_a, h_b, size);

    std::cout << "\nHost -> Device (normal)\n";
    t1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Bandwidth: " << bandwidth(size, time) << " GB/s\n";

    std::cout << "\nDevice -> Host (normal)\n";
    t1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Bandwidth: " << bandwidth(size, time) << " GB/s\n";
    check_result(h_a, h_b, size);

    std::cout << "\nHost -> Device (pinned)\n";
    t1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_pinned_a, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Bandwidth: " << bandwidth(size, time) << " GB/s\n";

    std::cout << "\nDevice -> Host (pinned)\n";
    t1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_pinned_b, d_a, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Bandwidth: " << bandwidth(size, time) << " GB/s\n";
    check_result(h_pinned_a, h_pinned_b, size);

    std::cout << "\nDevice -> Device\n";
    t1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Bandwidth: " << bandwidth(size, time) << " GB/s\n";

    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    check_result(h_a, h_b, size);

    free(h_a);
    free(h_b);
    cudaFreeHost(h_pinned_a);
    cudaFreeHost(h_pinned_b);
    cudaFree(d_a);
    cudaFree(d_b);
}
