#include <cstdio>
#include <cstdlib>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16
#define MAX_VAL 5.0f       // 可调最大输入值
#define SCALE_TARGET 0.25f  // 缩放目标区间上限

// CUDA Core 实现（使用 float 原始数据）
__global__ void cuda_core_gemm(const float *A, const float *B, float *C) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

// Tensor Core 实现（使用 WMMA API，FP16 × FP16 → FP32）
__global__ void tensor_core_gemm(const half *A, const half *B, float *C) {
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, K);
    wmma::load_matrix_sync(b_frag, B, K);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    wmma::store_matrix_sync(C, acc_frag, N, wmma::mem_row_major);
}

// CUDA 错误检查封装
void check(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float A_orig[M * K], B_orig[K * N];      // 原始 float 输入（CUDA Core 使用）
    half A_tensor[M * K], B_tensor[K * N];   // 缩放后 half 输入（Tensor Core 使用）
    float C_cuda[M * N] = {0}, C_tensor[M * N] = {0};

    srand((unsigned int)time(NULL));

    float scale_in = SCALE_TARGET / MAX_VAL;
    float scale_out = 1.0f / (scale_in * scale_in);  // 输出还原因子

    // 初始化原始输入并生成缩放后的 half 数据
    for (int i = 0; i < M * K; ++i) {
        A_orig[i] = (rand() / (float)RAND_MAX) * MAX_VAL;
        A_tensor[i] = __float2half(A_orig[i] * scale_in);
    }
    for (int i = 0; i < K * N; ++i) {
        B_orig[i] = (rand() / (float)RAND_MAX) * MAX_VAL;
        B_tensor[i] = __float2half(B_orig[i] * scale_in);
    }

    // 分配显存
    float *d_A_float, *d_B_float, *d_C_cuda, *d_C_tensor;
    half *d_A_half, *d_B_half;
    check(cudaMalloc(&d_A_float, sizeof(A_orig)), "cudaMalloc A_float");
    check(cudaMalloc(&d_B_float, sizeof(B_orig)), "cudaMalloc B_float");
    check(cudaMalloc(&d_C_cuda, sizeof(C_cuda)), "cudaMalloc C_cuda");

    check(cudaMalloc(&d_A_half, sizeof(A_tensor)), "cudaMalloc A_half");
    check(cudaMalloc(&d_B_half, sizeof(B_tensor)), "cudaMalloc B_half");
    check(cudaMalloc(&d_C_tensor, sizeof(C_tensor)), "cudaMalloc C_tensor");

    // 拷贝数据到 device
    check(cudaMemcpy(d_A_float, A_orig, sizeof(A_orig), cudaMemcpyHostToDevice), "Memcpy A_float");
    check(cudaMemcpy(d_B_float, B_orig, sizeof(B_orig), cudaMemcpyHostToDevice), "Memcpy B_float");
    check(cudaMemcpy(d_A_half, A_tensor, sizeof(A_tensor), cudaMemcpyHostToDevice), "Memcpy A_half");
    check(cudaMemcpy(d_B_half, B_tensor, sizeof(B_tensor), cudaMemcpyHostToDevice), "Memcpy B_half");

    // CUDA Core 执行
    dim3 threads(N, M);
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    cuda_core_gemm<<<1, threads>>>(d_A_float, d_B_float, d_C_cuda);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float time_cuda = 0;
    cudaEventElapsedTime(&time_cuda, start1, stop1);

    // Tensor Core 执行
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    tensor_core_gemm<<<1, 32>>>(d_A_half, d_B_half, d_C_tensor);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float time_tensor = 0;
    cudaEventElapsedTime(&time_tensor, start2, stop2);

    // 拷贝回主机
    check(cudaMemcpy(C_cuda, d_C_cuda, sizeof(C_cuda), cudaMemcpyDeviceToHost), "Memcpy C_cuda");
    check(cudaMemcpy(C_tensor, d_C_tensor, sizeof(C_tensor), cudaMemcpyDeviceToHost), "Memcpy C_tensor");

    // Tensor Core 输出还原
    for (int i = 0; i < M * N; ++i)
        C_tensor[i] *= scale_out;

    // 验证精度
    bool pass = true;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(C_cuda[i] - C_tensor[i]);
        if (diff > 1e-2) {
            printf("Mismatch at %d: CUDA=%.2f, Tensor=%.2f, Diff=%.5f\n", i, C_cuda[i], C_tensor[i], diff);
            pass = false;
        }
    }

    printf("\nResult Match: %s\n", pass ? "YES" : "NO");
    printf("CUDA Core Time: %.4f ms\n", time_cuda);
    printf("Tensor Core Time: %.4f ms\n", time_tensor);

    // 释放内存
    cudaFree(d_A_float); cudaFree(d_B_float);
    cudaFree(d_A_half);  cudaFree(d_B_half);
    cudaFree(d_C_cuda);  cudaFree(d_C_tensor);
    return 0;
}
