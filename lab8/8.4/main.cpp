#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>


double matmul_gpu_naive    (const float* h_A, const float* h_B, float* h_C, int N);
double matmul_gpu_cache_row(const float* h_A, const float* h_B, float* h_C, int N);
double matmul_gpu_cache_col(const float* h_A, const float* h_B, float* h_C, int N);
double matmul_gpu_tiled    (const float* h_A, const float* h_B, float* h_C, int N);

static void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < N; k++) {
                s += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

static bool verify(const float* ref, const float* res, int N, float eps = 1e-2f) {
    for (int i = 0; i < N * N; i++) {
        if (fabsf(ref[i] - res[i]) > eps * fabsf(ref[i]) + eps) {
            return false;
        }
    }
    return true;
}

int main() {
    const int sizes[] = {128, 256, 512, 1024, 2048};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    std::cout << std::fixed;
    std::cout.precision(4);

    std::ofstream csv("results.csv");
    csv << "N,method,t_ms,gflops,speedup\n";
    csv << std::fixed;
    csv.precision(6);

    for (int si = 0; si < n_sizes; si++) {
        int N = sizes[si];
        size_t bytes = (size_t)N * N * sizeof(float);

        float* h_A  = new float[N * N];
        float* h_B  = new float[N * N];

        //cpu
        float* h_C  = new float[N * N];
        //gpu
        float* h_Cg = new float[N * N];

        for (int i = 0; i < N * N; i++) {
            h_A[i] = (float)(rand() % 10) / 10.f;
            h_B[i] = (float)(rand() % 10) / 10.f;
        }

        double cpu_ms = 0;
        
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_cpu(h_A, h_B, h_C, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        

        double flops = 2.0 * (double)N * N * N;

        auto bench = [&](const char* name,
                         double(*fn)(const float*, const float*, float*, int))
        {
            double ms      = fn(h_A, h_B, h_Cg, N);
            bool   ok      = verify(h_C, h_Cg, N);
            double gflops  = flops / (ms * 1e-3) / 1e9;
            double speedup = cpu_ms / ms;

            std::cout << "N=" << N << "  " << name
                      << " t=" << ms << " ms"
                      << " perf=" << gflops << " GFLOP/s"
                      << " speedup=" << speedup << "x"
                      << "  " << (ok ? "ok" : "MISMATCH") << "\n";

            csv << N << "," << name << ","
                << ms << "," << gflops << "," << speedup << "\n";
        };

        std::cout << "\nN=" << N << "\n";
        std::cout << "CPUt=" << cpu_ms << " ms\n";

        bench("naive ", matmul_gpu_naive);
        bench("cache_row ", matmul_gpu_cache_row);
        bench("cache_col ", matmul_gpu_cache_col);
        bench("tiled ", matmul_gpu_tiled);

        delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_Cg;
    }
}
