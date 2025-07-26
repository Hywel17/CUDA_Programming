#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// GPU 核函数
__global__ void vecAddGPU(float *A, float *B, float *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// CPU 实现
void vecAddCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

int main() {
    int n = 1 << 24; // 16 million
    size_t size = n * sizeof(float);

    // 分配主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ---------------- CPU 执行时间 ----------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vecAddCPU(h_A, h_B, h_C_cpu, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double time_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    printf("[CPU] Elapsed time: %.3f ms\n", time_cpu);

    // ---------------- GPU 执行时间 ----------------
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);

    vecAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float time_gpu = 0;
    cudaEventElapsedTime(&time_gpu, start_gpu, end_gpu);
    printf("[GPU] Elapsed time: %.3f ms, Grid: %d, Block: %d, Threads total: %d\n",
           time_gpu, blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // ---------------- 结果验证 ----------------
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
            correct = false;
            printf("Mismatch at %d: CPU=%f, GPU=%f\n", i, h_C_cpu[i], h_C_gpu[i]);
            break;
        }
    }

    // 释放资源
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);

    return 0;
}
