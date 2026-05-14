# Paged Attention Unroll Tiling Tuning Guide

## 当前 Case1 基准

```
batch=256, num_heads=16, head_dim=128, block_size=128, context_len=8192
bn_this_batch = ceil(8192/128) = 64 blocks, q_loop = ceil(16/16) = 1
```

Kernel 模板参数由三个值决定：

```
QK:  <q_tile, head_dim,   block_size>   当前 <16, 128, 128>
PV:  <q_tile, block_size, head_dim>     当前 <16, 128, 128>
SF:  <q_tile, block_size>               当前 <16, 128>
UP:  <q_tile, head_dim>                 当前 <16, 128>
```

---

## 方案 A：调 N_UNROLL（1 行改动，无需改 kernel）

**文件**：`kernels/orchestration/paged_attention_orch.cpp`

```cpp
// 第 32 行，改这一个数字即可
#define N_UNROLL 8   // 当前 64
```

| N_UNROLL | 迭代数 | task/scope | 单 task blocks | sij_buf |
|----------|--------|-----------|---------------|---------|
| 64（当前） | 1 | 4 | 64 | 512KB |
| 32 | 2 | 8 | 32 | 256KB |
| 16 | 4 | 16 | 16 | 128KB |
| 8 | 8 | 32 | 8 | 64KB |

**约束**：太小 → 单 task 过轻，swimlane 稀疏。

---

## 方案 B：调 block_size（test params + 3 kernel 模板）

`block_size` 是系统参数（KV cache 分页大小），不是模型参数，可自由调整。
总计算量不变（相同的矩阵乘总量，只是切块方式不同）。

### B.1 修改文件清单

#### 1) test params

`test_paged_attention_unroll.py` Case1:

```python
# 改前
"block_size": 128,
# 改后
"block_size": 64,
```

#### 2) QK matmul — `kernels/aic/aic_qk_matmul.cpp`

kernel_entry dispatch 加分支：

```cpp
uint64_t q_tile_size = static_cast<uint64_t>(qi->shapes[0]);
uint64_t head_dim = static_cast<uint64_t>(qi->shapes[1]);
uint64_t block_size = sij_buf->shapes[1] / n_blocks;

if (q_tile_size == 16 && block_size == 64) {
    qk_matmul_n_impl<16, 128, 64>(...);     // 新增
} else if (q_tile_size == 16) {
    qk_matmul_n_impl<16, 128, 128>(...);    // 原有
} else {
    qk_matmul_n_impl<64, 128, 64>(...);     // 原有
}
```

#### 3) PV matmul — `kernels/aic/aic_pv_matmul.cpp`

```cpp
uint64_t q_tile_size = static_cast<uint64_t>(pij_buf->shapes[0]);
uint64_t block_size = pij_buf->shapes[1] / n_blocks;

if (q_tile_size == 16 && block_size == 64) {
    pv_matmul_n_impl<16, 64, 128>(...);     // 新增
} else if (q_tile_size == 16) {
    pv_matmul_n_impl<16, 128, 128>(...);    // 原有
} else {
    pv_matmul_n_impl<64, 64, 128>(...);     // 原有
}
```

#### 4) Softmax — `kernels/aiv/aiv_softmax_prepare.cpp`

```cpp
uint64_t q_tile_size = static_cast<uint64_t>(sij_buf->shapes[0]);
uint64_t block_size = sij_buf->shapes[1] / n_blocks;

if (q_tile_size == 16 && block_size == 64) {
    softmax_prepare_n_impl<16, 64>(...);    // 新增
} else if (q_tile_size == 16) {
    softmax_prepare_n_impl<16, 128>(...);   // 原有
} else {
    softmax_prepare_n_impl<64, 64>(...);    // 原有
}
```

#### 5) Online update — 不需要改

模板是 `<q_tile, head_dim>`，head_dim 不变。

#### 6) Orchestration — 不需要改

block_size 从 tensor shape 动态读取，sij/pij shape 动态计算。

### B.2 效果对比

| | block_size=128 | block_size=64 |
|---|---|---|
| 总 blocks | 64 | **128** |
| N_UNROLL=64 → groups | 1 (4 tasks) | **2 (8 tasks)** |
| QK B tile | (128,128)=32KB | **(128,64)=16KB** |
| PV A tile | (16,128)=4KB | **(16,64)=2KB** |
| SF sij tile | (16,128)=8KB | **(16,64)=4KB** |

### B.3 Dispatch 说明

三个 kernel 需要在运行时区分 block_size。最简方式：
从已有 tensor shape 推导 `block_size = buf->shapes[1] / n_blocks`，
无需修改 orchestration 传参。

如果不需要兼容 block_size=128，可以直接把模板参数改掉，不加分支。

---

## 方案 C：调 q_tile（orchestration + 4 kernel 模板）

仅对 Case2/3 生效（num_heads=64），Case1 的 q_tile=16 已是 CUBE 下限。

### C.1 修改文件清单

#### 1) Orchestration

`kernels/orchestration/paged_attention_orch.cpp` 第 108 行：

```cpp
// 改前
uint64_t q_tile = std::min(num_heads, static_cast<uint64_t>(128));
// 改后（下限 clamp 到 16，防止 Case1 被减到 8）
uint64_t q_tile = std::max(
    std::min(num_heads, static_cast<uint64_t>(128)) / 2,
    static_cast<uint64_t>(16));
```

#### 2) 4 个 kernel 各加一个 q_tile=32 分支

```
QK:  加 <32, 128, 64>
PV:  加 <32, 64, 128>
SF:  加 <32, 64>
UP:  加 <32, 128>
```

### C.2 效果

Case2/3: q_tile 64→32, q_loop 1→2（scope 数翻倍）。
Case1: 无影响。

---

## 方案 D：调 head_dim（test params + 3 kernel 模板）

head_dim 是模型参数，改动等于换模型配置。仅用于实验探索。

### D.1 修改文件清单

#### 1) test params

```python
"head_dim": 64,   # 当前 128
```

#### 2) 3 个 kernel 加分支

```
QK:  加 <16, 64, 128>   (K=head_dim 变)
PV:  加 <16, 128, 64>   (N=head_dim 变)
UP:  加 <16, 64>         (N=head_dim 变)
SF:  不变（模板参数不含 head_dim）
```

Dispatch 可用 `qi->shapes[1]`（即 head_dim）区分。

---

## 组合速查表

| 组合 | QK | PV | SF | UP | blocks | 改动量 |
|------|----|----|----|----|--------|--------|
| 当前 bs=128 hd=128 | `<16,128,128>` | `<16,128,128>` | `<16,128>` | `<16,128>` | 64 | — |
| **bs=64** hd=128 | `<16,128,64>` | `<16,64,128>` | `<16,64>` | `<16,128>` | 128 | 3 kernel |
| bs=128 **hd=64** | `<16,64,128>` | `<16,128,64>` | `<16,128>` | `<16,64>` | 64 | 3 kernel |
| **bs=64 hd=64** | `<16,64,64>` | `<16,64,64>` | `<16,64>` | `<16,64>` | 128 | 4 kernel |

## 推荐顺序

1. **方案 B**（block_size=64）— 改动小、收益确定、不改模型语义
2. **方案 A**（N_UNROLL）— 配合方案 B 精调 task 粒度
3. **方案 C**（q_tile）— 仅 Case2/3 受益
4. **方案 D**（head_dim）— 实验性质
