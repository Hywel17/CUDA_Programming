#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 1024
#define KERNEL_SIZE 3
#define TILE_WIDTH 16

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE] = {
     0, -1,  0,
    -1,  5, -1,
     0, -1,  0
};

// CUDA 二维卷积实现
__global__ void conv2D_gpu_shared(const float *input, float *output, int width) {
    // 分配共享内存 tile 区域（包含 halo）
    __shared__ float tile[TILE_WIDTH + 2][TILE_WIDTH + 2];

    // 当前线程在 tile 中的局部索引
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // 当前线程在全局图像中的像素坐标
    int global_row = blockIdx.y * TILE_WIDTH + local_y;
    int global_col = blockIdx.x * TILE_WIDTH + local_x;

    // 将当前线程需要的数据搬入共享内存（包含 padding）
    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            int r = global_row + offset_y;
            int c = global_col + offset_x;

            // 映射到共享内存中的位置（+1 是为了处理 halo 区）
            int tile_y = local_y + offset_y + 1;
            int tile_x = local_x + offset_x + 1;

            // 边界判断（超出范围的用0填充）
            if (r >= 0 && r < width && c >= 0 && c < width)
                tile[tile_y][tile_x] = input[r * width + c];
            else
                tile[tile_y][tile_x] = 0.0f;
        }
    }

    // 所有线程完成共享内存填充后进行同步
    __syncthreads();

    // 保证线程不会越界写入图像
    if (global_row < width && global_col < width) {
        float result = 0.0f;

        // 在共享内存中执行3x3卷积
        for (int i = 0; i < KERNEL_SIZE; ++i) {
            for (int j = 0; j < KERNEL_SIZE; ++j) {
                float pixel = tile[local_y + i][local_x + j];
                float weight = d_kernel[i * KERNEL_SIZE + j];
                result += pixel * weight;
            }
        }
        output[global_row * width + global_col] = result;
    }
}


// CPU 二维卷积实现
void conv2D_cpu(const float *input, float *output, int width) {
    // 遍历每个像素位置，逐个计算卷积结果
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f; // 当前输出像素的累加结果

            // 遍历3x3卷积核区域（偏移范围为 -1 ~ 1）
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    // 当前核对应的输入像素坐标
                    int r = row + dy;
                    int c = col + dx;
                    // 越界位置补0（zero padding）
                    float pixel = 0.0f;
                    if (r >= 0 && r < width && c >= 0 && c < width) {
                        pixel = input[r * width + c];
                    }
                    // 卷积核的索引换算（E.g., 将 [-1,1] 映射为 [0,2]）
                    int kernelIdx = (dy + 1) * KERNEL_SIZE + (dx + 1);
                    sum += pixel * d_kernel[kernelIdx];
                }
            }
            output[row * width + col] = sum;
        }
    }
}


void initData(float *data, int n) {
    for (int i = 0; i < n * n; i++) data[i] = i % 10;
}

int main() {
    int size = N * N * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_cpu_out = (float*)malloc(size);
    float *h_gpu_out = (float*)malloc(size);
    initData(h_input, N);

    // CPU 计时
    auto start_cpu = std::chrono::high_resolution_clock::now();
    conv2D_cpu(h_input, h_cpu_out, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU 分配
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // GPU 计时
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    conv2D_gpu_shared<<<blocks, threads>>>(d_input, d_output, N);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    cudaMemcpy(h_gpu_out, d_output, size, cudaMemcpyDeviceToHost);

    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time: %.3f ms\n", gpu_time);

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_cpu_out); free(h_gpu_out);
    return 0;
}
