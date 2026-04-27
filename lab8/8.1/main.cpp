#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>


void vec_sum_cuda(float *a, float *b, float *c, int n);

const int N = 1024;

float a[N], b[N], c[N];

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, N - 1);

    for (int i=0; i<N; i++) {
        a[i] = dis(gen);
        b[i] = dis(gen);
        c[i] = 0;
    }

    vec_sum_cuda(a, b, c, 1024);

    std::cout << "\nVecSum: ";
    for (int i=0; i<20; i++) 
        std::cout << c[i] << " ";

    std::cout << "\nreference: ";
    for (int i=0; i<20; i++) 
        std::cout << a[i]+b[i] << " ";
}