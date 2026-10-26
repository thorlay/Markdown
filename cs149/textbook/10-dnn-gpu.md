# 第 10 章｜如何让深度神经网络在 GPU 上高效执行？

> Stanford CS149 Fall 2023，主题 Lecture 10（2023-10-26）。这是一份基于官方课件的**原创教材初稿**。没有完整逐段校对视频 Transcript，故不宣称重现教授全部课堂细节。额外的 LLM 推理讨论会单独注明为课外延伸。

**原始资料：** [官方 Slides PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/dnneval/10_dnneval.pdf) · [Stanford Online 录课](https://www.youtube.com/watch?v=qbKtU0X6-WU)

## 本章解决什么问题？

为什么一个看上去只有几个数学算子的神经网络，在真实 GPU 上运行时，性能却可能远低于峰值 FLOPs？为什么高性能 DNN Runtime 需要同时考虑算子选择、数据布局、线程映射、融合和执行调度？

学完后应能：
1. 区分算子层的 FLOPs、内存流量和端到端执行时间；
2. 理解 GEMM、卷积及其在 GPU 上的常见映射；
3. 解释 Tiling 与 Operator Fusion 为什么影响性能；
4. 判断什么时候 GPU 的运算峰值不是主要瓶颈；
5. 从 Attention 的表达式理解中间 Tensor 的内存成本。

## 一、深度学习计算不是只有乘法

一个简化的神经网络推理过程可能是：

~~~text
Input
  ↓
Matrix Multiplication / Convolution
  ↓
Bias
  ↓
Activation
  ↓
Normalization
  ↓
Output
~~~

在数学层面，逐个算子执行没有问题；在真实 GPU 上，它可能意味着：

1. Kernel A 从设备内存读取 Tensor；
2. Kernel A 把中间结果写回设备内存；
3. Kernel B 再读取刚刚写出的结果；
4. Kernel B 把新的结果写回；
5. 多次重复。

因此**数学上算子少，并不自动意味着执行代价低**。真正的成本还包括 Kernel Launch、数据移动、内存分配、同步以及工作负载规模。

## 二、矩阵乘法为什么值得特殊优化？

令 A 的形状为 M×K，B 为 K×N，输出 C 为 M×N：

~~~text
C[i,j] = sum_k A[i,k] * B[k,j]
~~~

对于常见 Dense GEMM，总乘加工作量大约为 2MNK FLOPs（按一次乘法、一次加法分别记 1 FLOP）。

朴素三重循环：

~~~cpp
for (int i = 0; i < M; ++i)
  for (int j = 0; j < N; ++j) {
    float s = 0.0f;
    for (int k = 0; k < K; ++k)
      s += A[i*K+k] * B[k*N+j];
    C[i*N+j] = s;
  }
~~~

这能描述正确数学关系，但不一定高效：同一段 A、B 数据可能被许多输出元素反复需要。如果每次都从较慢的存储层级重新取数据，就浪费了大量时间和带宽。

### Tiling / Blocking

把输出 C 分成小矩形块。计算其中一个输出 Tile 时，只加载对应的 A、B 子块，并让一批乘加操作重复使用这些输入。这样可以减少来自慢速内存层级的重复读取。

抽象结构是：

~~~text
for each output tile (i, j):
  local_C = 0
  for each k tile:
    local_A = load A[i_tile, k_tile]
    local_B = load B[k_tile, j_tile]
    local_C += local_A @ local_B
  store local_C
~~~

但 Tile 并非越大越好。更大 Tile 可能需要更多 Registers 和 Shared Memory，进而减少并行驻留的 Warps / Blocks，甚至导致寄存器溢出。高性能库通常会根据不同 Shape、dtype 和 GPU 架构使用不同的 Tile 策略。

### 算术密度的直觉

如果没有数据复用，加载数据后只做很少的算术操作，内存传输可能更贵。Tiling 的目标之一，就是让每次从慢内存取回的数据参与更多计算，使 *Arithmetic Intensity = FLOPs / Bytes Moved* 更高。

不过必须注明内存层级：HBM→SM 的 bytes 与 Shared Memory→寄存器的 bytes 不能混算。

## 三、卷积如何映射到高效 GPU 计算？

二维卷积的每个输出位置，通常要对输入的某个局部窗口和 Filter 权重做乘加。许多输出位置在输入或权重上存在大量数据复用机会。

常见方法包括：

- **Direct convolution**：直接安排线程计算输出位置，尽量复用局部窗口或权重；
- **Im2col + GEMM**：把卷积组织成矩阵乘法，从优化成熟的 GEMM 中受益；
- **Implicit GEMM**：用索引映射表达类似 GEMM 的执行，而避免显式物化很大的 im2col 中间矩阵。

没有哪一种方法对所有卷积 Shape 都绝对最好。卷积核大小、输入通道数、Batch、Layout、GPU 特性与库实现都会改变结论。

举例：1×1 convolution 的空间窗口大小为 1，在许多布局下与 GEMM 的关系尤其直接。对于更大卷积核，输入窗口重叠造成的数据复用与索引开销需要同时考虑。

## 四、Operator Fusion：少写中间结果

考虑：

~~~text
Z = A @ B
Y = relu(Z + bias)
~~~

一种朴素实现会分成三个 Kernel：

1. GEMM：计算 Z 并写入 HBM；
2. Bias：读 Z、加偏置、写回；
3. ReLU：再读 Z、写 Y。

如果这三个步骤能在 GEMM 的 Epilogue 中安全融合，就有可能直接将最终输出写入 HBM，从而减少中间结果的写入、读取以及 Kernel Launch。

这种优化背后的主要收益是**减少数据移动**，而不是改变矩阵乘法的理论算术复杂度。

Fusion 也不是万能：
- 一个过于复杂的 Kernel 可能增加 Registers；
- 更高资源占用可能降低 Occupancy；
- 某些中间 Tensor 实际上需要被其他消费者共享；
- 不同算子使用不同并行粒度时，硬融合可能破坏高效的线程布局。

所以融合是否真正有效，必须在目标硬件和目标 Tensor Shape 下测量。

## 五、Transformer Attention 为什么会带来很大的内存压力？

对于单个 Attention Head，可以抽象为：

~~~text
S = Q @ K^T / sqrt(d)
P = softmax(S)
O = P @ V
~~~

设 Q、K、V 的序列长度都是 L，S 和 P 的形状通常为 L×L。标准朴素实现若将完整 S、P 物化在 GPU HBM，单个 Head 就要承担与 L² 成正比的中间存储。

例如只考虑单个矩阵、单个 Head、单个 Batch：

~~~text
L = 4096
L² = 16,777,216 elements

float16: 约 32 MiB / matrix
float32: 约 64 MiB / matrix
~~~

这里还没有计入 Q、K、V、输出、其他 Heads、反向传播中间状态或实际实现的 Workspace。因此在长序列场景中，注意力中间 Tensor 的存储和读写可能很贵。

### 为什么 FlashAttention 思想重要？

通过 Tiling，可以分块处理 Query、Key、Value，同时在片上维护局部归一化所需的统计量，避免把整个 Attention Score / Probability 矩阵作为大型中间对象反复写入 HBM。

核心收益包括：
1. 减少对 HBM 的中间结果访问；
2. 在片上复用更多输入；
3. 结合数值稳定的在线 Softmax 实现完整计算。

**重要限定：** 这并不把 Dense Attention 的所有 O(L²) 数学计算自动变成 O(L)。主要改变的是 I/O 路径和资源利用方式。

此外，原始 Stanford CS149 Fall 2023 的 Assignment 4 主要让学生用 **C++/CPU** 思考 Attention 的 Locality、Blocking、Fusion 和并行优化；这不是让学生直接交一个 GPU CUDA FlashAttention 实现。算法理念有联系，但运行平台和具体瓶颈不同。

## 六、从算子快到系统快：一个实际工程成本模型

假设一个模型的 GPU 执行过程是：

~~~text
Input transfer:      2 ms
GEMM kernels:        6 ms
Elementwise kernels: 2 ms
Other overhead:      2 ms
--------------------------
Total:              12 ms
~~~

若 GEMM 优化 2 倍，得到：

~~~text
Input transfer:      2 ms
GEMM kernels:        3 ms
Elementwise kernels: 2 ms
Other overhead:      2 ms
--------------------------
Total:               9 ms
~~~

GEMM 自己提升 2 倍，端到端却仅提升 12/9 = 1.33 倍。这个示意说明两点：

- 只优化热点子模块，整体速度可能受其他环节限制；
- 真正的性能工程需要持续更新成本模型，随着瓶颈变化调整优化重点。

给模型系统做 Profiling 时，至少要分清输入准备、传输、Kernel Execution、同步、CPU 调度与输出处理。

## 七、与现代 AI Infra 的关系（课外延伸）

CS149 的 GPU DNN 优化可以自然接到这些当前系统议题：

**LLM Prefill**：批量处理输入 Token，较多大型矩阵运算，适合研究 Tensor Cores、GEMM、FlashAttention 以及动态 Shape。

**LLM Decode**：自回归阶段逐 Token 生成，KV Cache 读取、内存带宽、Batch Scheduling 经常成为关键因素，不能简单用 Prefill FLOPs 推断 Token 生成速度。

**Tensor Parallelism**：单 GPU 计算拆分到多个 GPU 时，还需要关注通信与同步成本；本章的内存层级思想会扩展到 NVLink、NCCL 与跨节点网络。

**Inference Runtime**：即使所有算子已经使用高效库，请求合批、KV Cache 管理、排队、Admission Control 仍可能决定用户体验。

这些属于现代工程扩展，不宣称全部内容都在 2023 年课程 Slides 中逐一讨论。

## 八、练习（含简答）

**Q1：** 计算形状为 (1024, 2048) 与 (2048, 1024) 的矩阵乘法，理论 FLOPs 大约多少？

答：2×1024×2048×1024 ≈ 4.29×10^9 FLOPs，即约 4.29 GFLOPs。注意 FLOPs 是**工作量**，FLOP/s 才是执行速率。

**Q2：** 有 8 个 Attention Heads，L=4096，float16。如果每个 Head 都显式保存一份 L×L Probability 矩阵，最少需要多少原始元素内存？

答：8×4096²×2 bytes = 268,435,456 bytes，即 256 MiB。实际训练内存通常远大于这个单一中间 Tensor 的数字。

**Q3：** 既然 Fusion 减少 Kernel Launch，是否把所有算子都合到一个 Kernel 更好？

答：否。过度 Fusion 会造成寄存器与 Shared Memory 压力、布局不匹配、编译成本上升，以及对可并行性的限制。

**Q4：** 为什么同一模型的 Prefill 和 Decode 可能表现出不同瓶颈？

答：Tensor Shape、Batch 结构、缓存读取和数据复用都不同。Prefill 往往具有较大算术工作量；Decode 可能需要反复读取较大的 KV Cache，并伴随动态请求调度。

## 九、建议实践

1. 完成 [CS149 Assignment 4](https://github.com/stanford-cs149/cs149gpt) 中的一版正确 Attention；
2. 做 Blocked / Fused 版本，记录 Correctness、Runtime、峰值内存；
3. 对输入序列长度 L 做倍增实验，例如 256、512、1024、2048；
4. 用曲线区分计算工作量、内存容量与实际时间的变化；
5. 在学习 [CS336](https://cs336.stanford.edu/) 或 [GPU MODE](https://github.com/gpu-mode/lectures) 时，再做 Triton / FlashAttention 方向的 GPU 实验。

## 来源核验

| 来源 | 核验范围 | 状态 |
| --- | --- | --- |
| [Fall 2023 L10 官方 Slides](https://gfxcourses.stanford.edu/cs149/fall23content/media/dnneval/10_dnneval.pdf) | DNN scheduling、Conv/GEMM、Tensor/Operator Fusion 与 Attention 的课程主线 | 官方 PDF 关键内容已核查 |
| [Stanford Online L10 视频](https://www.youtube.com/watch?v=qbKtU0X6-WU) | 与 2023 主题讲次对应 | 视频入口已核实 |
| 完整视频 Transcript | 全部课堂口述与准确时间戳 | **尚未取得和逐段核查** |

> 本章为原创教材，不是课程 Slides 的完整翻译，也不代表 Stanford 官方授权或背书。
