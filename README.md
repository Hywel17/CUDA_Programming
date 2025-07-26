# CUDA_PROGRAMMING

这是我在学习 CUDA 编程过程中积累的代码与实验记录，主要包括以下两个模块：

- **TensorCore_GEMM_WideInt**：使用 Tensor Core 实现高位宽整数乘法（如 128/256/512-bit 等），并与 CUDA Core 进行验证与性能对比。
- **Basics**：一些基础 CUDA 编程练习，包含 GEMM、共享内存、卷积等示例，并与 CPU 结果进行验证与性能对比。

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
├── TensorCore_GEMM_WideInt/      # 分阶段探索 Tensor Core 高位宽整数乘法
│   ├── stage1_fp16_16x16_gemm.cu
│   ├── stage2_fp16_128x128_gemm.cu
│   ├── stage3_int8_16x16_gemm.cu
│   ├── stage4_int8_128x128_gemm_repeat.cu
│   ├── stage5_final_gemm.cu
│   └── README.md              
│
└── README.md            
