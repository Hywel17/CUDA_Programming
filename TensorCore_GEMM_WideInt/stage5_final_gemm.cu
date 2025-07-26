#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ====== 用户自定义参数 ======
// 矩阵规模（矩阵A: M×K  矩阵B: K×N  矩阵C: M×N）
#define M 128
#define N 128
#define K 128

const int BIT_WIDTH = 1024;  // 相乘的bit宽度（例如：128/256/512）
const int REPEAT = 10000;    // 每种实现重复执行次数（用于平均计时）
// ===========================

#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(call)); \
        exit(EXIT_FAILURE); \
    }

// CUDA Core 实现：int8 × int8 → int32，支持分段组合成高位乘法
__global__ void cuda_core_gemm_seg(const int8_t *A, const int8_t *B, int32_t *C_segs, int seg) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int32_t sum = 0;
    for (int i = 0; i < K; ++i) {
        int8_t a = A[row * K + i + seg * M * K];
        int8_t b = B[i * N + col + seg * K * N];
        sum += static_cast<int32_t>(a) * static_cast<int32_t>(b);
    }
    C_segs[row * N + col + seg * M * N] = sum;
}

int main() {
    const int SEGMENTS = BIT_WIDTH / 64;  // 每个segment相当于64-bit contribution（来自int8×int8→int32）
    const int total_size = SEGMENTS * M * K;

    // 主机端输入矩阵（拆成多个bit segment）
    int8_t *A = new int8_t[total_size];
    int8_t *B = new int8_t[total_size];
    int32_t *C_cuda = new int32_t[M * N * SEGMENTS];
    int32_t *C_tensor = new int32_t[M * N * SEGMENTS];

    // 初始化数据：随机int8范围 [-128, 127]
    srand((unsigned int)time(NULL));
    for (int i = 0; i < total_size; ++i) {
        A[i] = rand() % 256 - 128;
        B[i] = rand() % 256 - 128;
    }

    // 分配GPU内存
    int8_t *d_A, *d_B;
    int32_t *d_C_cuda, *d_C_tensor;
    CHECK(cudaMalloc(&d_A, sizeof(int8_t) * total_size));
    CHECK(cudaMalloc(&d_B, sizeof(int8_t) * total_size));
    CHECK(cudaMalloc(&d_C_cuda, sizeof(int32_t) * M * N * SEGMENTS));
    CHECK(cudaMalloc(&d_C_tensor, sizeof(int32_t) * M * N * SEGMENTS));

    CHECK(cudaMemcpy(d_A, A, sizeof(int8_t) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, sizeof(int8_t) * total_size, cudaMemcpyHostToDevice));

    // CUDA Core GEMM 执行（逐段模拟高位乘法）
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float time_cuda_total = 0;
    for (int r = 0; r < REPEAT; ++r) {
        cudaEvent_t start1, stop1;
        cudaEventCreate(&start1); cudaEventCreate(&stop1);
        cudaEventRecord(start1);

        for (int seg = 0; seg < SEGMENTS; ++seg) {
            cuda_core_gemm_seg<<<grid, block>>>(d_A, d_B, d_C_cuda, seg);
        }

        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        float time_cuda;
        cudaEventElapsedTime(&time_cuda, start1, stop1);
        time_cuda_total += time_cuda;
    }

    // Tensor Core GEMM 执行（逐段模拟高位乘法）
    // 每一段使用 cublasGemmEx 调用 Tensor Core 加速 int8 × int8 → int32 的矩阵乘法
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const int32_t alpha = 1, beta = 0;
    float time_tensor_total = 0;
    for (int r = 0; r < REPEAT; ++r) {
        cudaEvent_t start2, stop2;
        cudaEventCreate(&start2); cudaEventCreate(&stop2);
        cudaEventRecord(start2);

        for (int seg = 0; seg < SEGMENTS; ++seg) {
            cublasGemmEx(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,  // 矩阵不转置
                        N, M, K,                  // 矩阵规模
                        &alpha,                   // 缩放因子 alpha
                        d_B + seg * K * N,        // B矩阵（segment偏移）
                        CUDA_R_8I,                // 数据类型为 int8
                        N,                        // 每列之间的步长（B为列主序）
                        d_A + seg * M * K,        // A矩阵（segment偏移）
                        CUDA_R_8I,                // 数据类型为 int8
                        K,                        // 每列之间的步长（A为行主序）
                        &beta,                    // 缩放因子 beta
                        d_C_tensor + seg * M * N, // 结果矩阵 C（segment偏移）
                        CUDA_R_32I,               // 输出类型为 int32
                        N,                        // 每列之间的步长
                        CUDA_R_32I,               // 运算使用 int32 精度
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP); // 启用 Tensor Core（混合精度矩阵乘法）
        }
        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        float time_tensor;
        cudaEventElapsedTime(&time_tensor, start2, stop2);
        time_tensor_total += time_tensor;
    }

    // 拷贝回主机，验证结果是否一致
    CHECK(cudaMemcpy(C_cuda, d_C_cuda, sizeof(int32_t) * M * N * SEGMENTS, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(C_tensor, d_C_tensor, sizeof(int32_t) * M * N * SEGMENTS, cudaMemcpyDeviceToHost));

    bool match = true;
    for (int seg = 0; seg < SEGMENTS; ++seg) {
        for (int i = 0; i < M * N; ++i) {
            int idx = seg * M * N + i;
            if (C_cuda[idx] != C_tensor[idx]) {
                printf("Mismatch at [%d][%d] seg=%d: CUDA=%d, Tensor=%d\n",
                       i / N, i % N, seg, C_cuda[idx], C_tensor[idx]);
                match = false;
            }
        }
    }

    printf("\n模拟乘法位宽：%d-bit\n", BIT_WIDTH);
    printf("重复次数：%d\n", REPEAT);
    printf("结果是否一致：%s\n", match ? "YES" : "NO");
    printf("CUDA Core 平均时间：%.4f ms\n", time_cuda_total / REPEAT);
    printf("Tensor Core 平均时间：%.4f ms\n", time_tensor_total / REPEAT);

    // 清理资源
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_cuda); cudaFree(d_C_tensor);
    delete[] A; delete[] B; delete[] C_cuda; delete[] C_tensor;
    cublasDestroy(handle);

    return 0;
}
