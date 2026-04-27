#include <cuda.h>
#include <cuda_runtime.h>


__global__ void VecSumKernel(float *a, float *b, float *c) {
    // Определение индекса потока
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Обработка соответствующей порции данных
    c[i] = a[i] + b[i];
} 


// a, b – указатели на исходные массивы
// c – указатель на результирующий массив
// n – размер массивов (число элементов)
void vec_sum_cuda(float *a, float *b, float *c, int n)
{
    int SizeInBytes = n * sizeof(float);

    // Указатели на массивы в видеопамяти
    float *a_gpu = NULL;
    float *b_gpu = NULL;
    float *c_gpu = NULL;

    // Выделение памяти под массивы на GPU
    cudaMalloc( (void **)&a_gpu, SizeInBytes );
    cudaMalloc( (void **)&b_gpu, SizeInBytes );
    cudaMalloc( (void **)&c_gpu, SizeInBytes );

    // Копирование исходных данных из CPU на GPU
    cudaMemcpy(a_gpu, a, SizeInBytes, cudaMemcpyHostToDevice); // a_gpu = a
    cudaMemcpy(b_gpu, b, SizeInBytes, cudaMemcpyHostToDevice); // b_gpu = b

    // Задание конфигурации запуска ядра
    dim3 threads = dim3(512, 1); // 512 потоков в блоке
    dim3 blocks = dim3(n/threads.x, 1); // n/512 блоков в сетке

    // Запуск ядра (покомпонентное умножение векторов c = a * b)
    VecSumKernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);

    // Копирование результата из GPU в CPU
    cudaMemcpy(c, c_gpu, SizeInBytes, cudaMemcpyDeviceToHost); // c = c_gpu

    // Освобождение памяти GPU
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}