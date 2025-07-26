#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 16
#define N 16
#define K 16

#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(call)); \
        exit(EXIT_FAILURE); \
    }

// CUDA Core 实现（int8 × int8 → int32）
__global__ void cuda_core_gemm(const int8_t *A, const int8_t *B, int32_t *C) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    int32_t sum = 0;
    for (int i = 0; i < K; ++i) {
        sum += static_cast<int32_t>(A[row * K + i]) * static_cast<int32_t>(B[i * N + col]);
    }
    C[row * N + col] = sum;
}

int main() {
    int8_t A[M * K], B[K * N];     // 原始 int8 输入
    int32_t C_cuda[M * N] = {0};   // CUDA Core 输出
    int32_t C_tensor[M * N] = {0}; // Tensor Core 输出

    srand((unsigned int)time(NULL));

    // 初始化 int8 输入数据 [-128, 127]
    for (int i = 0; i < M * K; ++i)
        A[i] = rand() % 256 - 128;
    for (int i = 0; i < K * N; ++i)
        B[i] = rand() % 256 - 128;

    // 分配 device 内存
    int8_t *d_A, *d_B;
    int32_t *d_C_cuda, *d_C_tensor;
    CHECK(cudaMalloc(&d_A, sizeof(A)));
    CHECK(cudaMalloc(&d_B, sizeof(B)));
    CHECK(cudaMalloc(&d_C_cuda, sizeof(C_cuda)));
    CHECK(cudaMalloc(&d_C_tensor, sizeof(C_tensor)));

    CHECK(cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, sizeof(B), cudaMemcpyHostToDevice));

    // CUDA Core 执行
    dim3 threads(N, M);
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1); cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    cuda_core_gemm<<<1, threads>>>(d_A, d_B, d_C_cuda);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float time_cuda = 0;
    cudaEventElapsedTime(&time_cuda, start1, stop1);

    // Tensor Core 执行（int8 × int8 → int32）
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int32_t alpha = 1, beta = 0;
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2); cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_8I, N,
                 d_A, CUDA_R_8I, K,
                 &beta,
                 d_C_tensor, CUDA_R_32I, N,
                 CUDA_R_32I,
                 CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float time_tensor = 0;
    cudaEventElapsedTime(&time_tensor, start2, stop2);

    // 拷回主机
    CHECK(cudaMemcpy(C_cuda, d_C_cuda, sizeof(C_cuda), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(C_tensor, d_C_tensor, sizeof(C_tensor), cudaMemcpyDeviceToHost));

    // 检查结果
    bool match = true;
    for (int i = 0; i < M * N; ++i) {
        int32_t diff = abs(C_cuda[i] - C_tensor[i]);
        if (diff > 0) {
            printf("Mismatch at %d: CUDA=%d, Tensor=%d, Diff=%d\n", i, C_cuda[i], C_tensor[i], diff);
            match = false;
        }
    }

    printf("\nResult Match: %s\n", match ? "YES" : "NO");
    printf("CUDA Core Time: %.4f ms\n", time_cuda);
    printf("Tensor Core Time: %.4f ms\n", time_tensor);

    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_cuda); cudaFree(d_C_tensor);
    cublasDestroy(handle);
    return 0;
}
