#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

#define N 1024
#define TILE_WIDTH 16

// CUDA kernel: 使用共享内存的 GEMM 实现
__global__ void matMul_shared(const float *A, const float *B, float *C, int n) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;

    // 遍历每一个Tile（相当于分块矩阵进行乘法运算）
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // 读取 tile 的一部分 A/B
        if (row < n && t * TILE_WIDTH + tx < n)
            tile_A[ty][tx] = A[row * n + t * TILE_WIDTH + tx];
        else
            tile_A[ty][tx] = 0.0f;

        if (col < n && t * TILE_WIDTH + ty < n)
            tile_B[ty][tx] = B[(t * TILE_WIDTH + ty) * n + col];
        else
            tile_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += tile_A[ty][k] * tile_B[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// CPU 版本的矩阵乘法
void matMul_cpu(const float *A, const float *B, float *C, int n) {
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += A[row * n + k] * B[k * n + col];
            C[row * n + col] = sum;
        }
}

// 初始化矩阵
void initMatrix(float *mat, int n) {
    for (int i = 0; i < n * n; ++i)
        mat[i] = rand() % 5;
}

// 检查结果是否一致
bool checkResult(float *ref, float *test, int n) {
    for (int i = 0; i < n * n; ++i)
        if (fabs(ref[i] - test[i]) > 1e-3)
            return false;
    return true;
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);

    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // ===== CPU 测试 =====
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matMul_cpu(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // ===== CUDA 测试 =====
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    matMul_shared<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // ===== 打印结果 =====
    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time: %.3f ms\n", gpu_time);
    printf("Result matched: %s\n", checkResult(h_C_cpu, h_C_gpu, N) ? "Yes" : "No");

    // 释放内存
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}
