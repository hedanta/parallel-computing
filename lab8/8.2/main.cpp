#include <iostream>
#include <cuda_runtime.h>


int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << "CUDA device count: " << device_count << "\n\n";

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, i);

        std::cout << "Device " << i << ":\n";
        std::cout << "Name: " << dp.name << "\n";
        std::cout << "Global memory: " << dp.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "Constant memory: " << dp.totalConstMem / 1024 << " KB\n";
        std::cout << "Shared memory per block: " << dp.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "Registers per block: " << dp.regsPerBlock << "\n";
        std::cout << "Warp size: " << dp.warpSize << "\n";
        std::cout << "Max threads per block: " << dp.maxThreadsPerBlock << "\n";
        std::cout << "Compute capability: " << dp.major << "." << dp.minor << "\n";
        std::cout << "Multiprocessors: " << dp.multiProcessorCount << "\n";
        std::cout << "Core clock: " << dp.clockRate / 1000 << " MHz\n";
        std::cout << "Memory clock: " << dp.memoryClockRate / 1000 << " MHz\n";
        std::cout << "L2 cache size: " << dp.l2CacheSize / 1024 << " KB\n";
        std::cout << "Memory bus width: " << dp.memoryBusWidth << " bits\n";
        std::cout << "Max threads dim: ("
                  << dp.maxThreadsDim[0] << ", "
                  << dp.maxThreadsDim[1] << ", "
                  << dp.maxThreadsDim[2] << ")\n";

        std::cout << "Max grid size: ("
                  << dp.maxGridSize[0] << ", "
                  << dp.maxGridSize[1] << ", "
                  << dp.maxGridSize[2] << ")\n";
    }
}