# CUDA_PROGRAMMING

本仓库记录了我在学习 CUDA 编程过程中的一些实践代码，包含基础功能实现，以及一个基于 Tensor Core 的高位宽整数乘法实现。

- **Basics**：包含 GEMM、共享内存、卷积、向量加法等基础 CUDA 编程练习，支持与 CPU 结果进行验证与性能对比。
- **TensorCore_GEMM_WideInt**：使用 Tensor Core 实现大位宽整数矩阵乘法（如 128 / 256 / 512 / 1024-bit），支持用户自定义位宽，并与 CUDA Core 实现进行逐项结果验证。

---

## 项目结构
```bash
CUDA_PROGRAMMING/
│
├── Basics/                        # 基础 CUDA 编程练习
│   ├── Conv2d.cu                  # 卷积运算 CUDA 实现
│   ├── GEMM_1_Naive.cu            # 朴素 GEMM 实现（无优化）
│   ├── GEMM_2_SharedMem.cu        # 使用共享内存优化 GEMM
│   └── VectorAdd.cu               # 向量加法 CUDA 实现
│
├── TensorCore_GEMM_WideInt/      # Tensor Core 实现高位宽整数矩阵乘法
│   ├── stage1_fp16_16x16_gemm.cu           # FP16 示例（用于掌握 WMMA API）
│   ├── stage2_fp16_128x128_gemm.cu         # 扩展 FP16 tile 到大矩阵
│   ├── stage3_int8_16x16_gemm.cu           # int8 × int8 → int32 示例
│   ├── stage4_int8_128x128_gemm_repeat.cu  # 扩展到大矩阵，支持重复执行
│   ├── stage5_final_gemm.cu                # 支持任意位宽的 Tensor Core 整数乘法
│   └── README.md
│
└── README.md                   # 本说明文件
