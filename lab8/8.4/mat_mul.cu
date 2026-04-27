#include <cuda_runtime.h>
#include <iostream>

#define TILE 16

#define CUDA_CHECK(call) \
    { cudaError_t e = (call); \
      if (e != cudaSuccess) { std::cerr << "CUDA: " << cudaGetErrorString(e) << "\n"; exit(1); } }


__global__ void k_naive(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) {
        return;
    }
    float s = 0.f;
    for (int k = 0; k < N; k++) {
        s += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = s;
}

__global__ void k_cache_row(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) {
        return;
    }

    extern __shared__ float s_A[];

    float s = 0.f;
    int block_size = blockDim.x;
    for (int t = 0; t < (N + block_size - 1) / block_size; t++) {
        int k = t * block_size + threadIdx.x;
        s_A[threadIdx.x] = (k < N) ? A[i * N + k] : 0.f;
        __syncthreads();

        int k_base = t * block_size;
        int k_end  = (k_base + block_size < N) ? k_base + block_size : N;
        for (int kk = k_base; kk < k_end; kk++) {
            s += s_A[kk - k_base] * B[kk * N + j];
        }
        __syncthreads();
    }
    C[i * N + j] = s;
}

__global__ void k_cache_col(const float* A, const float* B, float* C, int N) {
    int j = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    extern __shared__ float s_B[];

    float s = 0.f;
    int block_size = blockDim.x;
    for (int t = 0; t < (N + block_size - 1) / block_size; t++) {
        int k = t * block_size + threadIdx.x;
        s_B[threadIdx.x] = (k < N) ? B[k * N + j] : 0.f;
        __syncthreads();

        int k_base = t * block_size;
        int k_end  = (k_base + block_size < N) ? k_base + block_size : N;
        for (int kk = k_base; kk < k_end; kk++) {
            s += A[i * N + kk] * s_B[kk - k_base];
        }
        __syncthreads();
    }
    C[i * N + j] = s;
}

__global__ void k_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float s = 0.f;
    
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int ac = t * TILE + threadIdx.x;
        int br = t * TILE + threadIdx.y;

        s_A[threadIdx.y][threadIdx.x] = (row < N && ac < N) ? A[row * N + ac] : 0.f;
        s_B[threadIdx.y][threadIdx.x] = (br  < N && col < N) ? B[br  * N + col] : 0.f;
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            s += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        C[row * N + col] = s;
    }
}


typedef void (*KernelFn)(const float*, const float*, float*, int);

static double run_timed(KernelFn fn, float* d_A, float* d_B, float* d_C, int N, dim3 grid, dim3 block, size_t smem = 0)
{
    fn<<<grid, block, smem>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    fn<<<grid, block, smem>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return (double)ms;
}

static void alloc_copy(const float* h_A, const float* h_B, float** d_A, float** d_B, float** d_C, size_t bytes)
{
    CUDA_CHECK(cudaMalloc(d_A, bytes));
    CUDA_CHECK(cudaMalloc(d_B, bytes));
    CUDA_CHECK(cudaMalloc(d_C, bytes));
    CUDA_CHECK(cudaMemcpy(*d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_B, h_B, bytes, cudaMemcpyHostToDevice));
}

static void copy_free(float* h_C, float* d_A, float* d_B, float* d_C, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

double matmul_gpu_naive(const float* h_A, const float* h_B, float* h_C, int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    alloc_copy(h_A, h_B, &d_A, &d_B, &d_C, bytes);
    
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    double ms = run_timed(k_naive, d_A, d_B, d_C, N, grid, block);
    
    copy_free(h_C, d_A, d_B, d_C, bytes);
    return ms;
}

double matmul_gpu_cache_row(const float* h_A, const float* h_B, float* h_C, int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    alloc_copy(h_A, h_B, &d_A, &d_B, &d_C, bytes);

    int block_size = (N <= 1024) ? N : 1024; 
    dim3 block(block_size), grid((N + block_size - 1) / block_size, N);
    double ms = run_timed(k_cache_row, d_A, d_B, d_C, N, grid, block, block_size * sizeof(float));
    
    copy_free(h_C, d_A, d_B, d_C, bytes);
    return ms;
}

double matmul_gpu_cache_col(const float* h_A, const float* h_B, float* h_C, int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    alloc_copy(h_A, h_B, &d_A, &d_B, &d_C, bytes);
    
    int block_size = (N <= 1024) ? N : 1024;
    dim3 block(block_size), grid((N + block_size - 1) / block_size, N);
    double ms = run_timed(k_cache_col, d_A, d_B, d_C, N, grid, block, block_size * sizeof(float));
    
    copy_free(h_C, d_A, d_B, d_C, bytes);
    return ms;
}

double matmul_gpu_tiled(const float* h_A, const float* h_B, float* h_C, int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    alloc_copy(h_A, h_B, &d_A, &d_B, &d_C, bytes);

    dim3 block(TILE, TILE), grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);
    double ms = run_timed(k_tiled, d_A, d_B, d_C, N, grid, block);

    copy_free(h_C, d_A, d_B, d_C, bytes);
    return ms;
}
