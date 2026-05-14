# Scheduler Dispatch 负载均衡分析

## 当前 Dispatch 数据流

```
completion fanout
       |
       v
  local buffer (per-thread)  ──优先──>  pop  ──>  dispatch to 自己的 idle core
       |                                 ^
       | overflow                        |
       v                                 |
  global ready queue (MPMC)  ──────────-─┘
```

每轮 dispatch 循环（`resolve_and_dispatch`）：
1. 从 local buffer 优先 pop，不够再从 global ready queue pop
2. 只能 dispatch 到 `core_trackers_[thread_idx]` 拥有的 core
3. 用不完的 local buffer 内容 push 回 global queue

## 不均衡来源

### 1. 每个线程只能 dispatch 到自己的 core

`dispatch_shape()` 通过 `tracker.get_dispatchable_cores()` 只返回本线程
拥有的 core。线程 A 有 idle core 而线程 B 没有时，B 的 pop 能力被浪费。

### 2. Local buffer 造成依赖链自锁

task A 在 thread 0 complete → A 的后继进入 thread 0 的 local buffer →
thread 0 下轮 dispatch 先从 local buffer pop → dispatch 到自己的 core →
complete 后后继又回到 local buffer。

结果：依赖链被"锁"在同一个线程上，其他线程 idle core 吃不到任务。

### 3. Local buffer 只在溢出时才进 global queue

当线程 idle core 足够多时，local buffer 内容全被自己消耗，其他线程在
global queue 上 pop 不到任何 task。

## 方案

### 方案 A：限制 local buffer 自消费比例（推荐先试）

改动点：`get_ready_tasks_batch()`

每轮 dispatch 只允许从 local buffer 取一定比例（如 1/2），剩余立刻 push
到 global queue，让其他线程有 task 可 dispatch。

- 改动量：极小（一个函数）
- 均衡效果：中等
- 额外开销：低（多一些 global queue atomic 操作）

### 方案 B：Cross-thread dispatch

去掉"只能 dispatch 到自己 core"的约束。线程自己没有 idle core 时，尝试
claim 其他线程的 idle core 来 dispatch。

需要将 `CoreTracker::core_states_` 改为 `atomic<uint64_t>`，或增加
per-core CAS claim 机制。`CoreExecState` 已是 per-core 的，completion
回路不受影响。

- 改动量：大
- 均衡效果：最好
- 额外开销：中（per-core CAS）

### 方案 C：去掉 local buffer，全走 global queue

`release_fanin_and_check_ready()` 中不再走 local buffer，所有 ready
task 直接 push 到 global MPMC queue。任务分配完全由 MPMC 特性保证公平。

- 改动量：小
- 均衡效果：好
- 额外开销：中（queue contention 增加）

## 推荐顺序

1. **方案 A** — 用 profiling 的 `local_dispatch_count` vs global dispatch
   count 验证各线程 dispatch 数量是否更均匀
2. **方案 C** — 如果 A 不够，用全 global queue 作为对照实验
3. **方案 B** — 如果队列公平性已够但仍有 idle core 浪费，再上 cross-thread
