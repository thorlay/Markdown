# 第 02 章｜现代多核处理器：Cache、Multicore、SIMD 与硬件多线程

> Stanford CS149 · Fall 2023 · **Lecture 2 — A Modern Multi-Core Processor**
>
> **状态：Self-contained 中文原创教材（第一版）**。本章依据用户提供的英文自动转写（YouTubeToTranscript，**Timestamp OFF**）以及 Stanford 官方 103 页 Slides 编写。已核对 Slides 的文本、案例和章节顺序，尚未把全部 PDF 图页逐张视觉复核，也未将自动字幕与原始音频逐句校验；不能将本文视作 Stanford 授权教材或完整逐字稿。

**先修知识：** 掌握变量、数组、for 循环、基础 C/C++ 语法即可。不需要先阅读 Lecture 1。本章会解释所有核心术语，并提供可手算的例题和不依赖 GPU 的实践。  
**学习时间：** 2–3 小时阅读与练习，约 1–2 小时完成编程。  
**来源：** [Stanford Lecture 2 页面](https://gfxcourses.stanford.edu/cs149/fall23/lecture/multicore/) · [103 页官方 Slides](https://gfxcourses.stanford.edu/cs149/fall23content/media/multicore/02_basicarch_xX3ssOi.pdf) · [课程视频](https://www.youtube.com/watch?v=CKmNpAO5rS4)

## 0. 本章的问题与学习目标

假设你要对一百万个浮点数执行相同的复杂计算。一台机器拥有多个 CPU Core，每个 Core 还有向量执行单元；为什么一个普通 for 循环未必能使用这些资源？为什么算术单元已经足够多，程序仍可能被内存访问拖慢？

本讲的三条主线分别是：

1. **Multicore**：让多个 Core 同时运行不同指令流，处理不同的工作。
2. **SIMD（Single Instruction, Multiple Data）**：让一条指令在不同数据元素上同时进行相同的操作。
3. **Hardware Multithreading**：一个 Core 维护多个硬件线程上下文；某个线程等待慢内存时，切换去执行另一个线程。

课程同时回顾 **Superscalar（超标量）**，它利用同一指令流里的独立指令。理解本章以后，你应当能够算出 Cache 的 Hit/Miss、SIMD Lane 的有效利用率、隐藏一段访存等待所需的线程数，并写出能验证正确性的多核 / SIMD 程序。

## 1. 先区分“程序要求的语义”与“硬件怎么实现”

从处理器的角度，程序最终体现为指令和状态。**指令（Instruction）**可以做算术、读取或修改寄存器、Load/Store 内存、跳转到下一段代码；**程序状态**包括寄存器值以及可访问的内存。

一个极简的 Core 可以画成：

~~~text
       Instruction Stream（指令流）
                   |
          +----------------+
          | Fetch / Decode |  选择下一步指令
          +----------------+
                   |
          +----------------+
          | Execution Unit |  算术 / Load / Store
          +----------------+
                   |
              Registers
                   |
                 Memory
~~~

这只是教材模型，不是完整的 CPU 微架构。真实处理器还有流水线、缓存、分支预测、寄存器重命名与 Out-of-Order Execution。

### 1.1 什么是 Instruction-Level Parallelism？

考虑：

~~~text
A = x * x
B = y * y
C = A + B
~~~

计算 A 与 B 互不依赖，所以某个 Core 如果拥有多个执行单元，可以在允许的资源和时序下并行执行；C 必须等待 A、B 的结果。这叫 **ILP（Instruction-Level Parallelism，指令级并行）**。

**Superscalar** 主要在一个指令流里寻找这种独立性并利用多个执行单元。它不等于多个 Core：一个 Core 即使能同周期执行多条独立指令，仍然在处理同一指令流里的工作。

### 1.2 为什么两个 Core 不一定比一个 Core 快？

多核芯片给不同 Core 提供各自的指令获取、执行与寄存器上下文。开发者需要提供可以并发运行的任务。如果程序只创建一个线程，那么第二个 Core 可能闲着；假如新 CPU 的单核性能还比较弱，原有串行程序甚至可能变慢。

课程强调一个贯穿全章的事实：**硬件提供并行能力，软件需要暴露可并行的工作。**

## 2. 为什么先讲 Cache？因为计算单元会等待数据

### 2.1 Memory 与 Cache 不应混为一谈

对程序而言，Memory 是“地址 → 值”的抽象。**DRAM** 是实现大容量内存的一种硬件技术。**Cache** 是容量较小、访问更快的存储层级，它保存主存中一部分数据的副本。

理想且正确的 Cache 实现改变程序运行时间，不改变程序要求的结果。

### 2.2 Cache Line 的含义

Cache 通常不是一次只搬一个字节，而是按 **Cache Line（缓存行）**搬运和管理数据。官方课件使用一个简化示例：

- 主存地址从 0x0 到 0xF，共 16 字节；
- Cache 总容量 8 字节；
- 每条 Cache Line 保存 4 字节；
- Cache 最多保留两条 Line；
- 使用 LRU（Least Recently Used，最近最少使用）逐出策略；
- 为了手算，假定两条 Line 可以保存任意两块地址区域。

地址按 4 字节对齐：

| Cache Line 起始地址 | 包含地址 |
| --- | --- |
| 0x0 | 0、1、2、3 |
| 0x4 | 4、5、6、7 |
| 0x8 | 8、9、10、11 |
| 0xC | 12、13、14、15 |

如果一条 Cache Line 有 L 字节，非负地址 a 所属行的起点是：

$$
\operatorname{line}(a)=\left\lfloor \frac{a}{L}\right\rfloor L
$$

所以在本例中访问地址 5，会加载起点为 4 的一整行，而不是单独加载一个字节。真实 x86 处理器的 Cache Line 常为 64 字节；**4 字节只为便于讲解**。

### 2.3 什么是 Hit、Miss 和 LRU？

**Hit**：目标地址所在行已经在 Cache 中，可以从当前层级取到数据。  
**Miss**：当前层级没有该行，需要从较慢的层级获取。  
**Eviction**：Cache 满后需要逐出旧行。  
**LRU**：优先逐出最长时间没访问过的行。

**例子 A：** 按顺序读取：

~~~text
0, 1, 2, 3, 2, 1, 4, 1
~~~

| 访问地址 | 结果 | Cache 当前行 |
| --- | --- | --- |
| 0 | Cold Miss，加载行 0 | 0 |
| 1 | Hit，同一行 | 0 |
| 2 | Hit | 0 |
| 3 | Hit | 0 |
| 2 | Hit，重复使用 | 0 |
| 1 | Hit，重复使用 | 0 |
| 4 | Cold Miss，加载行 4 | 0、4 |
| 1 | Hit，行 0 未被逐出 | 0、4 |

**结论：8 次访问、6 Hit、2 Miss。**

这展示了两种数据局部性：

- **Spatial Locality（空间局部性）**：访问 0 后很可能访问相邻的 1、2、3，一条 Line 带来多个 Hit。
- **Temporal Locality（时间局部性）**：刚使用过的 1、2 又被重复访问。

**例子 B：** 顺序访问整个 16 字节区域，再访问 0：

~~~text
0, 1, 2, 3, 4, 5, 6, 7,
8, 9, 10, 11, 12, 13, 14, 15, 0
~~~

只看造成缓存状态变化的访问：

| 访问 | 结果 | 需要逐出 | Cache 里保留的行 |
| --- | --- | --- | --- |
| 0 | Cold Miss | 无 | 0 |
| 4 | Cold Miss | 无 | 0、4 |
| 8 | Cold Miss | 0 | 4、8 |
| 12 | Cold Miss | 4 | 8、12 |
| 最后再次访问 0 | Capacity Miss | 8 | 12、0 |

其他访问都是 Hit，得到 **17 次访问、12 Hit、5 Miss**。

### 2.4 三种 Cache Miss 必须分清

课堂中教授在问答时曾短暂混淆 **Capacity Miss** 与 **Conflict Miss**，随后自己更正。这里保留正确定义：

- **Cold / Compulsory Miss（冷 / 强制未命中）**：第一次访问某条缓存行，即使 Cache 大到能保存全部数据，也不能凭空让它已经在 Cache 里。
- **Capacity Miss（容量未命中）**：总 Cache 容量不足以保存需再次使用的工作集。例子 B 最后一次读 0 属于这种情况；若 Cache 可以保存全部四条 Line，它就会 Hit。
- **Conflict Miss（冲突未命中）**：Cache 总容量本可能足够，但 Set Associativity / 地址映射限制使不同数据争用同一个位置。我们在本章的简化全相联模型中不模拟它。

Miss 分类帮助解释成因，不意味着 CPU 会给每个 Miss 一个独立的“类型标签”再决定是否从主存读取。

### 2.5 亲自运行 Cache 模拟器

下面是一份仅依赖 Python 标准库的完整 LRU 模拟器。保存为 **cache_lines.py**：

~~~python
from collections import OrderedDict

def simulate(addresses, line_bytes=4, capacity_lines=2):
    resident = OrderedDict()
    seen = set()
    hits = misses = 0
    log = []

    for address in addresses:
        line = (address // line_bytes) * line_bytes
        if line in resident:
            hits += 1
            kind = "hit"
            resident.move_to_end(line)
        else:
            misses += 1
            kind = "cold miss" if line not in seen else "miss after eviction"
            if len(resident) == capacity_lines:
                resident.popitem(last=False)
            resident[line] = None
        seen.add(line)
        log.append((address, kind, list(resident.keys())))
    return hits, misses, log

case_a = [0, 1, 2, 3, 2, 1, 4, 1]
case_b = list(range(16)) + [0]
for name, case, expected in [
    ("A", case_a, (6, 2)),
    ("B", case_b, (12, 5)),
]:
    hit, miss, log = simulate(case)
    assert (hit, miss) == expected
    print(f"{name}: hits={hit}, misses={miss}")
    for address, kind, lines in log:
        print(f"  addr={address:>2} | {kind:<19} | lines={lines}")
~~~

运行：

~~~bash
python3 cache_lines.py
~~~

应看到：

~~~text
A: hits=6, misses=2
B: hits=12, misses=5
~~~

**模型边界：** 代码中的 “miss after eviction” 不自动区分 Capacity 与 Conflict；模拟器不含 Set Associativity、预取、写回、CPU 时钟、访问延迟和多核一致性协议。它只让你验证 Locality、LRU 与 Hit/Miss 的关系。上述代码已在 Python 3 上运行并通过断言。

### 2.6 Cache 层级的性能差距有多大？

官方课件举出 Kaby Lake CPU、4 GHz 的教学数字：

| 访问位置 | 约需 Cycles | 4 GHz 下约需时间 |
| --- | ---: | ---: |
| L1 Cache | 4 | 1 ns |
| L2 Cache | 12 | 3 ns |
| L3 Cache | 38 | 9.5 ns |
| DRAM（较好条件） | 248 | 62 ns |

4 GHz 相当于每个 Cycle 约 0.25 ns。这是**课件中的特定 CPU 示例**，不是最新处理器的统一实测值。

假如某条运算必须先完成 Load，而数据暂时只在 DRAM，后续依赖指令可能等待很多周期。因此优化数据访问模式、让有用数据常驻近端 Cache，有时比增加 ALU 数量重要得多。

**Prefetching（预取）**是另一种手段：硬件预测未来要访问的地址并尝试提前加载。顺序数组容易预测；链表追踪或由刚计算出的数据决定地址则可能不容易预测。预取错误也可能浪费带宽并污染 Cache。

---

## 3. 一条贯穿整章的计算任务：逐元素近似 sin(x)

为了比较 Multicore、SIMD、Hardware Multithreading，课程使用同一个数组计算问题：

~~~text
for i in 0..N-1:
    y[i] = approximate_sin(x[i])
~~~

每个输出 y[i] 仅依赖对应的输入 x[i]。这意味着**不同 i 之间天然可以并行**；同一个元素内部的计算则可能存在依赖链。

### 3.1 Taylor Series 的完整定义

正弦函数在 0 附近可以展开为：

$$
\sin(x)=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\frac{x^7}{7!}+\cdots
$$

n! 表示阶乘，例如 3!=6，5!=120。保留有限项便得到近似。

定义 term₀=x。对 k≥1：

$$
\operatorname{term}_k=\operatorname{term}_{k-1}\frac{-x^2}{(2k)(2k+1)}
$$

取 K 个额外项的近似是：

$$
s(x)=\sum_{k=0}^{K}\operatorname{term}_k
$$

用 x=1、保留前三项手算：

~~~text
term0 = 1
term1 = -1/6    = -0.16666667
term2 = +1/120  = +0.00833333
s(1)  = 0.84166667
sin(1) ≈ 0.84147098
~~~

误差来自截断，不能误以为三项已经精确得到 sin(1)。课件原始示例采用不同的循环变量和分母写法；后续可运行代码使用 double 中间结果的递推形式，避免某些整数阶乘分母溢出问题。

### 3.2 为什么有并行工作，却未必被一个 Core 自动发现？

~~~text
计算 y[0]: term0 -> term1 -> term2 -> ...
计算 y[1]: term0 -> term1 -> term2 -> ...
计算 y[2]: term0 -> term1 -> term2 -> ...
           （行与行独立，行内有依赖）
~~~

Superscalar Core 可以在指令窗口内寻找某些独立指令，但它不保证会自动识别复杂循环里所有输入元素之间的独立性。程序员或编译器明确表达“外层循环各迭代独立”，能够帮助硬件获得更多并行任务。

**核心区分：单个元素内部的指令级并行（ILP）与不同元素之间的数据并行，是两个不同的并行维度。**

---

## 4. Multicore：用多个独立指令流处理不同元素

### 4.1 复制的是什么？

两个 Core 可以分别维护取指/解码、执行单元和寄存器状态，运行不同线程。真实 CPU 的某些 Cache 私有、某些共享，并非每一种存储资源都完整复制。

~~~text
Core 0 / Worker 0: x[0 .. middle) -> y[0 .. middle)
Core 1 / Worker 1: x[middle .. N) -> y[middle .. N)
~~~

记号 [begin,end) 表示包含 begin、不包含 end。输出区间没有重叠，所以各线程不写同一个 y[i]。

课程展示了用 C++ std::thread 分成两半的实现，以及一种叫 **forall** 的**概念性语法**：只声明哪些迭代独立，让编译器或 Runtime 决定如何使用处理器。**课程课件里的 forall 不是标准 C/C++ 关键字**。

若程序写死两个线程，即使机器拥有 16 Core，也可能只有两个线程在计算。Runtime 自适应的抽象更有扩展性，但独立性需要程序本身保证。

### 4.2 完整 C++17 多核实验

下方代码使用固定输入和 Taylor 递推，并同时验证“并行输出与串行输出一致”及“近似接近 std::sin”。代码刻意将线程创建纳入计时，后续可以研究线程池如何减少开销。

保存为 **parallel_sine.cpp**：

~~~cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

double approx_sin(double x, int terms) {
    double term = x;
    double sum = x;
    for (int k = 1; k <= terms; ++k) {
        term *= -x * x / ((2.0 * k) * (2.0 * k + 1.0));
        sum += term;
    }
    return sum;
}

void serial(const std::vector<double>& x,
            std::vector<double>& y, int terms) {
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] = approx_sin(x[i], terms);
}

void parallel(const std::vector<double>& x,
              std::vector<double>& y,
              int terms, unsigned workers) {
    workers = std::max(1U, workers);
    std::size_t chunk = (x.size() + workers - 1) / workers;

    auto work = [&](unsigned w) {
        std::size_t begin = w * chunk;
        std::size_t end = std::min(begin + chunk, x.size());
        for (std::size_t i = begin; i < end; ++i)
            y[i] = approx_sin(x[i], terms);
    };

    std::vector<std::thread> threads;
    for (unsigned w = 1; w < workers; ++w)
        threads.emplace_back(work, w);
    work(0);
    for (auto& t : threads) t.join();
}

int main(int argc, char** argv) {
    unsigned workers = argc > 1
        ? static_cast<unsigned>(std::stoul(argv[1])) : 4U;
    constexpr std::size_t n = 1 << 19;
    constexpr int terms = 8;
    std::vector<double> x(n), baseline(n), result(n);

    for (std::size_t i = 0; i < n; ++i)
        x[i] = 2.0 * static_cast<double>(i % 1024) / 1023.0 - 1.0;

    auto start = std::chrono::steady_clock::now();
    serial(x, baseline, terms);
    auto split = std::chrono::steady_clock::now();
    parallel(x, result, terms, workers);
    auto end = std::chrono::steady_clock::now();

    double max_diff = 0.0, max_math_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_diff = std::max(
            max_diff, std::abs(baseline[i] - result[i]));
        max_math_error = std::max(
            max_math_error, std::abs(result[i] - std::sin(x[i])));
    }

    double ts = std::chrono::duration<double, std::milli>(
        split - start).count();
    double tp = std::chrono::duration<double, std::milli>(
        end - split).count();

    std::cout << std::fixed << std::setprecision(3)
              << "workers=" << workers
              << " serial_ms=" << ts
              << " parallel_ms=" << tp
              << " speedup=" << (ts / tp) << "x\n"
              << "max_diff=" << max_diff
              << " max_sin_error=" << max_math_error << "\n";

    bool ok = max_diff < 1e-12 && max_math_error < 1e-12;
    std::cout << "correct=" << (ok ? "yes" : "NO") << "\n";
    return ok ? 0 : 1;
}
~~~

在 Linux / macOS 的 C++17 工具链上编译：

~~~bash
c++ -O2 -std=c++17 -pthread parallel_sine.cpp -o parallel_sine
./parallel_sine 1
./parallel_sine 2
./parallel_sine 4
./parallel_sine 8
~~~

**预期检查：** 每种 Worker 数量都应打印 correct=yes。串行与并行执行时间取决于机器、负载和 Compiler，**不能根据本章规定一个固定 Speedup**。

该代码已在 Linux/x86_64 以 4 Worker 编译运行并通过正确性检查。不同 Worker 只读同一输入 x、写自己独占的 y 区间；主线程也执行一个 Chunk，随后 join 确保所有 Worker 结束。

如果算法改成 y[i]=y[i-1]+x[i]，各迭代就存在真实依赖，不能如此简单地并行化。**数据并行不由 for 循环的外观决定，而由依赖决定。**

---

## 5. SIMD：一条指令同时处理许多数据

### 5.1 什么被复制，什么被共享？

Multicore 复制的是能够执行不同指令流的 Core。SIMD 则让一个指令控制多个数据 Lane，摊薄指令获取和控制成本：

~~~text
                VADD（一个向量加法指令）
                             |
                       Fetch/Decode
                             |
         +----------+--------+-------+----------+
         |          |                |          |
       Lane 0     Lane 1           ...       Lane 7
       a0+b0      a1+b1                     a7+b7
~~~

如果一个 float32 是 32 bits，而 AVX2 Vector Register 有 256 bits：

$$
V=\frac{256\ \mathrm{bits}}{32\ \mathrm{bits/float}}=8\ \mathrm{floats}
$$

因此一条相应的 AVX2 向量乘法或加法，可以处理 8 个 float。真实微架构中每周期能执行多少向量指令，还取决于执行端口、指令类别与数据是否就绪。

### 5.2 两种机制可以相乘，但结果仍只是理想上限

如果有 16 个 Core，每个 Core 在某个周期能发出一条 8-wide 浮点算术指令：

$$
16\times8=128\ \text{元素算术操作/周期（理想上限）}
$$

如果有 4 个 Core，每 Core 理想情况下能同时执行 3 条独立的 8-wide 算术指令：

$$
4\times3\times8=96\ \text{元素算术操作/周期（理想上限）}
$$

这些数字不是程序一定能达到的 FLOP/s：它们没有考虑数据带宽、指令依赖、Cache Miss、执行端口限制、实际频率和不同运算的吞吐率。

### 5.3 直接运行 AVX2 向量乘法

下面使用 AVX2 Intrinsic 操作 8 个 float，处理 19 个元素时最后 3 个由 Scalar Loop 补齐；不支持 AVX2 的编译环境可编译普通标量版本。

保存为 **avx2_multiply.cpp**：

~~~cpp
#include <cmath>
#include <iostream>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

void multiply(const float* a, const float* b, float* out, int n) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vc);
    }
#endif
    for (; i < n; ++i)
        out[i] = a[i] * b[i];
}

int main() {
    constexpr int n = 19;
    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = 2.0f;
    }
    multiply(a.data(), b.data(), out.data(), n);
    for (int i = 0; i < n; ++i) {
        if (std::fabs(out[i] - 2.0f * (i + 1)) > 1e-5f)
            return 1;
    }
    std::cout << "correct=yes n=19 vector_lane_width="
#if defined(__AVX2__)
              << 8
#else
              << 1
#endif
              << "\n";
}
~~~

先编译普通版本：

~~~bash
c++ -O2 -std=c++17 avx2_multiply.cpp -o multiply_scalar
./multiply_scalar
~~~

**仅在确认 x86/x86_64 CPU 支持 AVX2 时**，再运行向量版本：

~~~bash
c++ -O2 -mavx2 -std=c++17 avx2_multiply.cpp -o multiply_avx2
./multiply_avx2
~~~

预期输出分别为：

~~~text
correct=yes n=19 vector_lane_width=1
correct=yes n=19 vector_lane_width=8
~~~

这两个版本已在 Linux/x86_64 编译运行、通过正确性测试。Apple Silicon 不能直接执行 AVX2 指令。这里使用 unaligned load/store，因此没有额外的 32-byte 对齐要求。数据量只有 19，**不应把这个实验当作有效的性能 Benchmark**；此外普通版本也可能被编译器自动向量化。

**Explicit SIMD** 可以通过 Intrinsic 或 Compiler 生成的向量指令实现；在 GPU 的 **SIMT / Implicit SIMD** 模型中，程序员通常编写多个逻辑线程，硬件再将它们组织成共同执行的组。CPU AVX Register 与 GPU Warp 有联系，但不属于完全相同的编程抽象。

---

## 6. SIMD 最大的现实难题之一：Divergence

### 6.1 一个 8-wide 分组里出现不同 if 分支

假设一个 8-wide SIMD 组的条件结果是：

~~~text
Lane index:   0 1 2 3 4 5 6 7
Condition:    T T F T F F F F

执行 if：     X X . X . . . .
执行 else：   . . X . X X X X
~~~

其中 X 代表有效 Lane，点代表 Mask 掉的 Lane。若同一批计算资源需要先执行 if 路径、再执行 else 路径，部分 Lane 在某些阶段就没有做有用工作。

称同组工作需要相似指令流为 **Coherent Execution**；不同 Lane 需要不同路径为 **Divergence**。只有足够的控制流一致性，SIMD 的宽向量计算资源才容易得到有效利用。

### 6.2 用数字计算有效 Lane-Instruction 利用率

假设 V 个 Lane 中 n 个走 if，其他走 else；if 路径有 A 条向量算术指令，else 有 B 条。假设两条路径分别执行且没有额外优化，则一个教学用利用率模型为：

$$
U=\frac{nA+(V-n)B}{V(A+B)}
$$

分子是真正需要完成的 Lane-Instruction 数量；分母是执行两条路径时使用的总 Lane Capacity。

例如 V=8、n=3、A=3、B=2：

$$
U=\frac{3\times3+5\times2}{8(3+2)}
=\frac{19}{40}=47.5\%
$$

这是简化的 Mask 模型，不是所有真实 CPU/GPU 上的性能计数器数值。

### 6.3 课堂提问：如何接近 1/8 的最低利用率？

关键不是让 1 个 Lane 走 if、7 个走 else 就自动达到 1/8；还必须考虑**两条分支的成本**。

设只有 1 Lane 执行很长的 if 路径（100 条指令），7 Lane 执行极短的 else（1 条）：

$$
U=\frac{1\times100+7\times1}{8(100+1)}
=\frac{107}{808}\approx13.24\%
$$

当昂贵路径相对更长时，利用率趋近于 1/8=12.5%。如果两条路径成本一样，不能从“1 个 Lane 走 if”直接推断出 1/8 的整体利用率。

使用 Predication 或把条件变成 0/1 乘法也不是无成本的万能优化：如果依然执行两套昂贵运算，改写语法并不能减少有用计算之外的指令占用。

**真正需要优化的是：同组工作是否执行相似计算，以及为解决分歧而重新组织数据是否值得其搬运成本。**

---
