#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>

#define N 1024

// CUDA kernel: 每个线程计算一个 C[row][col]
__global__ void matMul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 当前线程负责的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 当前线程负责的列

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// 初始化数据
void initMatrix(float *mat, int n) {
    for (int i = 0; i < n * n; ++i)
        mat[i] = rand() % 5; 
}

// 验证结果是否一致
bool checkResult(float *a, float *b, int n) {
    for (int i = 0; i < n * n; ++i)
        if (fabs(a[i] - b[i]) > 1e-3) return false;
    return true;
}

// CPU版本
void matMul_cpu(const float *A, const float *B, float *C, int n) {
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += A[row * n + k] * B[k * n + col];
            C[row * n + col] = sum;
        }
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);

    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // CPU 计时
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matMul_cpu(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU 分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    // GPU 计时
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    matMul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time: %.3f ms\n", gpu_time);
    printf("Result matched: %s\n", checkResult(h_C_cpu, h_C_gpu, N) ? "Yes" : "No");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}
