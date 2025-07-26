#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define M 128
#define N 128
#define K 128
#define TILE 16

#define MAX_VAL 2.0f       // 输入上限
#define SCALE_TARGET 0.25f // 缩放目标区间上限

// CUDA Core 版本（float输入，验证用）
__global__ void cuda_core_gemm(const float *A, const float *B, float *C) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

// Tensor Core 实现（按 16x16 tile 执行）
__global__ void tensor_core_gemm(const half *A, const half *B, float *C) {
    int warpRow = blockIdx.y;
    int warpCol = blockIdx.x;

    if (warpRow * TILE >= M || warpCol * TILE >= N) return;

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    const half *tile_a = A + (warpRow * TILE * K);
    const half *tile_b = B + (warpCol * TILE);

    for (int i = 0; i < K; i += TILE) {
        wmma::load_matrix_sync(a_frag, tile_a + i, K);
        wmma::load_matrix_sync(b_frag, tile_b + i * N, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float *tile_c = C + warpRow * TILE * N + warpCol * TILE;
    wmma::store_matrix_sync(tile_c, acc_frag, N, wmma::mem_row_major);
}

void check(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float A_orig[M * K], B_orig[K * N];
    half A_half[M * K], B_half[K * N];
    float C_cuda[M * N] = {0}, C_tensor[M * N] = {0};

    float scale_in = SCALE_TARGET / MAX_VAL;
    float scale_out = 1.0f / (scale_in * scale_in);

    // 初始化原始输入并生成缩放后的 half 数据
    srand(time(NULL));
    for (int i = 0; i < M * K; ++i) {
        A_orig[i] = (rand() / (float)RAND_MAX) * MAX_VAL;
        A_half[i] = __float2half(A_orig[i] * scale_in);
    }
    for (int i = 0; i < K * N; ++i) {
        B_orig[i] = (rand() / (float)RAND_MAX) * MAX_VAL;
        B_half[i] = __float2half(B_orig[i] * scale_in);
    }

    float *d_Af, *d_Bf, *d_C_cuda, *d_C_tensor;
    half *d_Ah, *d_Bh;
    check(cudaMalloc(&d_Af, sizeof(A_orig)), "cudaMalloc A_float");
    check(cudaMalloc(&d_Bf, sizeof(B_orig)), "cudaMalloc B_float");
    check(cudaMalloc(&d_C_cuda, sizeof(C_cuda)), "cudaMalloc C_cuda");

    check(cudaMalloc(&d_Ah, sizeof(A_half)), "cudaMalloc A_half");
    check(cudaMalloc(&d_Bh, sizeof(B_half)), "cudaMalloc B_half");
    check(cudaMalloc(&d_C_tensor, sizeof(C_tensor)), "cudaMalloc C_tensor");

    check(cudaMemcpy(d_Af, A_orig, sizeof(A_orig), cudaMemcpyHostToDevice), "Memcpy A_float");
    check(cudaMemcpy(d_Bf, B_orig, sizeof(B_orig), cudaMemcpyHostToDevice), "Memcpy B_float");
    check(cudaMemcpy(d_Ah, A_half, sizeof(A_half), cudaMemcpyHostToDevice), "Memcpy A_half");
    check(cudaMemcpy(d_Bh, B_half, sizeof(B_half), cudaMemcpyHostToDevice), "Memcpy B_half");

    // CUDA Core 执行并计时
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    dim3 threads1(16, 16);
    dim3 blocks1((N + 15) / 16, (M + 15) / 16);
    cuda_core_gemm<<<blocks1, threads1>>>(d_Af, d_Bf, d_C_cuda);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float time_cuda = 0;
    cudaEventElapsedTime(&time_cuda, start1, stop1);

    // Tensor Core 执行并计时
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    dim3 threads2(32);  // 一个 warp
    dim3 blocks2(N / 16, M / 16);
    tensor_core_gemm<<<blocks2, threads2>>>(d_Ah, d_Bh, d_C_tensor);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float time_tensor = 0;
    cudaEventElapsedTime(&time_tensor, start2, stop2);

    check(cudaMemcpy(C_cuda, d_C_cuda, sizeof(C_cuda), cudaMemcpyDeviceToHost), "Memcpy C_cuda");
    check(cudaMemcpy(C_tensor, d_C_tensor, sizeof(C_tensor), cudaMemcpyDeviceToHost), "Memcpy C_tensor");

    // 反缩放恢复原精度
    for (int i = 0; i < M * N; ++i)
        C_tensor[i] *= scale_out;

    // 结果一致性检查
    bool match = true;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(C_cuda[i] - C_tensor[i]);
        if (diff > 1e-2) {
            printf("Mismatch at %d: CUDA=%.2f, Tensor=%.2f, Diff=%.5f\n",
                   i, C_cuda[i], C_tensor[i], diff);
            match = false;
        }
    }

    printf("\nResult Match: %s\n", match ? "YES" : "NO");
    printf("CUDA Core Time:   %.4f ms\n", time_cuda);
    printf("Tensor Core Time: %.4f ms\n", time_tensor);

    cudaFree(d_Af); cudaFree(d_Bf);
    cudaFree(d_Ah); cudaFree(d_Bh);
    cudaFree(d_C_cuda); cudaFree(d_C_tensor);
    return 0;
}
