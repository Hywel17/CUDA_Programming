# TensorCore_GEMM_WideInt

本项目旨在探索如何使用 Tensor Core 实现大位宽整数矩阵乘法（例如 128/256/512/1024-bit 等），并通过 CUDA Core 实现进行结果验证与性能对比。项目共分为五个阶段，从基本矩阵乘法逐步过渡到可自定义位宽的高位整数 GEMM，适合逐步掌握 Tensor Core 编程方法。

---

## 各阶段介绍

**Stage 1：Tensor Core 基本矩阵乘法（FP16, 16x16）**
- 文件：`stage1_fp16_16x16_gemm.cu`
- 说明：使用 WMMA API 编写 Tensor Core 的 FP16 tile 矩阵乘法，实现 C = A × B。
- 目标：掌握最基本的 Tensor Core 调用方式。

**Stage 2：大规模 Tile 拼接（FP16, 128x128）**
- 文件：`stage2_fp16_128x128_gemm.cu`
- 说明：将输入矩阵划分为多个 tile，并在全局范围内拼接结果，实现大矩阵乘法。
- 目标：掌握 tile 索引映射与多线程并行处理方法。

**Stage 3：int8 整数乘法测试（16x16）**
- 文件：`stage3_int8_16x16_gemm.cu`
- 说明：用 cuBLAS Tensor Core 支持的 int8×int8→int32 实现基本整数 GEMM。

**Stage 4：大矩阵 + 重复执行（int8, 128x128）**
- 文件：`stage4_int8_128x128_gemm_repeat.cu`
- 说明：对大规模矩阵执行多轮 GEMM，评估 Tensor Core 执行时间。

**Stage 5：高位宽整数乘法（final）**
- 文件：`stage5_final_gemm.cu`
- 说明：
  - 支持任意用户定义的位宽（如 128 / 256 / 512 / 1024-bit）；
  - 每个矩阵元素被拆分为若干 int8 段（segment）；
  - 分段执行 cuBLAS Tensor Core 的 GEMM，再组合结果；
  - 与 CUDA Core 的 int8 分段实现结果进行逐项比较；
  - 支持可调的矩阵规模 M/N/K 和重复执行次数。
- 参数设置：
```cpp
// 矩阵规模（矩阵A: M×K  矩阵B: K×N  矩阵C: M×N）
#define M 128
#define N 128
#define K 128

const int BIT_WIDTH = 1024;  // 相乘的bit宽度（例如 128/256/512/1024-bit 等）
const int REPEAT = 10000;    // 每种实现重复执行次数（用于平均计时）
```
- 编译 & 运行：
```bash
nvcc stage5_final_gemm.cu -arch=sm_87 -lcublas -o stage5
./stage5
```
> 说明：
> - 如果你使用的是其他 GPU 架构（如 `sm_80` 或 `sm_75`），可相应修改 `-arch=sm_XX`；
> - `-lcublas` 用于链接 cuBLAS 库支持 Tensor Core；
---
