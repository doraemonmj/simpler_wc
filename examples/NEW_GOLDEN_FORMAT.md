# Golden Script 使用指南

完全与用例解耦的测试脚本编写指南。

## 核心理念

**用户只需修改配置和计算逻辑，框架自动处理其余部分。**

## 必需的4个模块

### 1. 配置声明

```python
__outputs__ = ["result"]              # 输出 tensor 名称列表
TENSOR_ORDER = ["a", "b", "result"]   # tensor 顺序（匹配 C++ 函数签名）
PARAMS_LIST = [{"size": 16384, "dtype": "float32", "seed": 42}]  # 测试参数
RTOL, ATOL = 1e-4, 1e-4              # 精度容差（可选，默认 1e-5）
```

**参数说明：**
- `__outputs__`: 声明哪些 tensor 是输出（用于验证）
- `TENSOR_ORDER`: **必需且关键**！必须严格匹配 orchestration C++ 函数的参数顺序
- `PARAMS_LIST`: 参数字典列表，每个字典对应一个测试用例
- `RTOL/ATOL`: 数值比较的相对/绝对误差容忍度

### 2. 生成 Tensors

```python
def generate_inputs(params: dict) -> dict:
    """一次性生成所有 tensors（包括输入和输出）"""
    size = params["size"]
    dtype = params["dtype"]

    # 设置随机种子（如果有）
    if "seed" in params:
        np.random.seed(params["seed"])

    return {
        # 输入 tensors
        "a": np.random.rand(size).astype(dtype),
        "b": np.random.rand(size).astype(dtype),

        # 输出 tensors（用 zeros 初始化）
        "result": np.zeros(size, dtype=dtype),
    }
```

**说明：**
- 一次性返回所有 tensors（inputs + outputs）
- 输出 tensor 用 `np.zeros()` 初始化
- 从 `params` 读取参数，实现灵活配置

### 3. 计算 Golden

```python
def compute_golden(tensors: dict, params: dict) -> None:
    """计算期望结果（就地修改）"""
    a = torch.from_numpy(tensors["a"])
    b = torch.from_numpy(tensors["b"])

    # 你的计算逻辑
    result = a + b

    # 就地修改输出 tensor（必须用 [:] 赋值）
    tensors["result"][:] = result.numpy()
```

**说明：**
- **就地修改**：使用 `tensors["result"][:] = ...` 而不是 `tensors["result"] = ...`
- 函数无返回值（`None`）
- 可使用 `params` 获取测试参数

### 4. 运行测试

**硬件平台（NPU）：**
```bash
python examples/test_runner.py \
    --kernels examples/your_example/kernels \
    --golden examples/your_example/kernels/golden.py \
    --platform a2a3 \
    --device 0
```

**仿真平台（CPU）：**
```bash
python examples/test_runner.py \
    --kernels examples/your_example/kernels \
    --golden examples/your_example/kernels/golden.py \
    --platform a2a3sim \
    --device 0
```

**平台参数说明：**
- `--platform a2a3`: 真实硬件（NPU），使用 ccec 编译器
- `--platform a2a3sim`: 仿真平台（CPU），使用 g++ 编译器，无需硬件
- `--device 0`: 设备 ID（0-15），仿真时可忽略但参数仍需要

## 完整示例（30行）

```python
import numpy as np
import torch

# 1. 配置
__outputs__ = ["f"]
TENSOR_ORDER = ["a", "b", "f"]
PARAMS_LIST = [{"size": 16384, "dtype": "float32", "seed": 42}]
RTOL, ATOL = 1e-4, 1e-4

# 2. 生成 tensors
def generate_inputs(params: dict) -> dict:
    s, dt = params["size"], params["dtype"]
    if params.get("seed"):
        np.random.seed(params["seed"])
    return {
        "a": np.random.rand(s).astype(dt),
        "b": np.random.rand(s).astype(dt),
        "f": np.zeros(s, dtype=dt),
    }

# 3. 计算 golden
def compute_golden(tensors: dict, params: dict) -> None:
    a = torch.from_numpy(tensors["a"])
    b = torch.from_numpy(tensors["b"])
    tensors["f"][:] = ((a + b + 1) * (a + b + 2)).numpy()
```

运行：
```bash
python examples/test_runner.py  # 默认使用 a2a3 硬件
python examples/test_runner.py --platform a2a3sim  # 仿真平台
```

## 配置解耦

### 参数化配置

所有可变参数放在 `PARAMS_LIST` 中，代码逻辑从 `params` 读取：

```python
PARAMS_LIST = [{"size": 16384, "value_range": (0, 1)}]

def generate_inputs(params: dict):
    size = params["size"]              # 从配置读取
    vmin, vmax = params["value_range"]  # 从配置读取
    return {"a": np.random.uniform(vmin, vmax, size).astype("float32")}
```

### 显式 Tensor 顺序

`TENSOR_ORDER` 明确声明 tensor 顺序，匹配 C++ 函数签名：

```python
# C++ 函数: void func(float* a, float* b, float* result, ...)
TENSOR_ORDER = ["a", "b", "result"]  # 与 C++ 参数顺序一致
```

### 多测试用例

一次运行自动测试所有用例：

```python
PARAMS_LIST = [
    {"size": 128*128, "seed": 42},    # 用例 1
    {"size": 256*256, "seed": 123},   # 用例 2
    {"size": 32*32, "seed": 456},     # 用例 3
]
# 运行一次，自动测试所有 3 个用例
```

## 支持任意 Shape

不仅限于 vector，支持标量、矩阵等：

```python
def generate_inputs(params: dict) -> dict:
    return {
        "scalar": np.array([3.14], dtype="float32"),           # 标量
        "vector": np.random.rand(128).astype("float32"),       # 向量
        "matrix": np.random.rand(128, 128).flatten().astype("float32"),  # 矩阵
        "output": np.zeros(128*128, dtype="float32"),
    }
```

**注意**: Orchestration 接收 flatten 后的指针，复杂 shape 需要 `.flatten()`。

## 常见用法

### 固定随机种子（推荐）

确保测试结果可重现：

```python
PARAMS_LIST = [{"size": 16384, "seed": 42}]  # 每次运行结果相同
```

### 使用固定值

不使用随机值，用固定值测试：

```python
def generate_inputs(params: dict) -> dict:
    size = params["size"]
    return {
        "a": np.full(size, 2.0, dtype="float32"),  # 全部填充 2.0
        "b": np.full(size, 3.0, dtype="float32"),  # 全部填充 3.0
        "out": np.zeros(size, dtype="float32"),
    }
```

### 测试不同数据类型

```python
PARAMS_LIST = [
    {"size": 16384, "dtype": "float32", "seed": 42},   # FP32
    {"size": 16384, "dtype": "float16", "seed": 42},   # FP16
]
```

### 测试不同大小

```python
PARAMS_LIST = [
    {"size": 32*32, "seed": 42},      # 小规模快速验证
    {"size": 128*128, "seed": 42},    # 标准测试
    {"size": 512*512, "seed": 42},    # 压力测试
]
```

## 常见问题

### Q: 测试时而成功时而失败？

**A**: 两个原因及解决方案：

1. **未设置 seed** - 每次随机值不同
   ```python
   PARAMS_LIST = [{"size": 16384, "seed": 42}]  # 添加 seed
   ```

2. **精度容差过严** - 浮点计算误差
   ```python
   RTOL, ATOL = 1e-4, 1e-4  # 放宽容差（默认 1e-5）
   ```

### Q: TENSOR_ORDER 怎么确定？

**A**: 查看 orchestration C++ 函数签名：

```cpp
// C++ 代码
void example_orch(void* a, void* b, void* out, size_t size_a, ...)
                  ^^^^^^  ^^^^^^  ^^^^^^
                  按这个顺序
```

```python
# Python 配置
TENSOR_ORDER = ["a", "b", "out"]  # 与 C++ 一致
```

### Q: 如何只测试某个用例？

**A**: 注释掉其他用例：

```python
PARAMS_LIST = [
    {"size": 128*128, "seed": 42},      # 只测试这个
    # {"size": 256*256, "seed": 123},   # 注释掉
]
```

### Q: 硬件和仿真有什么区别？

**A**:

| 特性 | 硬件 (`a2a3`) | 仿真 (`a2a3sim`) |
|------|--------------|------------------|
| 运行环境 | NPU 硬件 | CPU 线程 |
| 编译器 | ccec | g++ |
| 需要硬件 | 是 | 否 |
| 速度 | 快（硬件加速） | 慢（软件模拟） |
| 用途 | 正式测试 | 开发调试 |

**使用场景：**
- 开发阶段：用 `--platform a2a3sim` 快速验证逻辑
- 正式测试：用 `--platform a2a3` 在真实硬件上验证性能

### Q: 能否自定义参数名？

**A**: 可以，`params` 是字典，任意 key 都支持：

```python
PARAMS_LIST = [
    {
        "batch_size": 32,
        "hidden_dim": 512,
        "learning_rate": 0.001,
        "seed": 42,
    }
]

def generate_inputs(params: dict):
    bs = params["batch_size"]
    hd = params["hidden_dim"]
    # 使用自定义参数
```

## 框架工作流程

框架自动执行以下步骤：

1. 读取 `PARAMS_LIST`，遍历每个 `params`
2. 调用 `generate_inputs(params)` 生成 tensors
3. 调用 `compute_golden(tensors, params)` 计算期望结果
4. 保存期望结果副本
5. 将 tensors 按 `TENSOR_ORDER` 传给 runtime
6. Runtime 在设备上执行
7. 比较实际结果与期望结果（使用 `RTOL/ATOL`）
8. 报告测试结果（PASS/FAIL）

**你只需提供 4 个模块，框架处理所有其他工作。**

## 示例代码

完整示例见：
- 单输出: [examples/host_build_graph_example/kernels/golden_torch_example.py](../host_build_graph_example/kernels/golden_torch_example.py)

运行示例：
```bash
# 默认（硬件）
python examples/test_runner.py

# 仿真
python examples/test_runner.py --platform a2a3sim

# 自定义
python examples/test_runner.py \
    --kernels examples/your_kernels \
    --golden examples/your_golden.py \
    --platform a2a3sim \
    --device 0
```
