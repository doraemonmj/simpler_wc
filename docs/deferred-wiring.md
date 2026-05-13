# SMT 同 Die 调度优化：WFE 等待 + Deferred Drain

## 背景

在 SMT 同 Die 场景下，orchestrator 和 scheduler 共享物理核。
原始设计中 scheduler 线程在等待 orchestrator 编排完成期间
直接进入 dispatch loop 做空循环（检查 running cores、
ready queue 等），所有操作都是空操作但消耗 CPU。在 SMT
同核上这些空转抢占 orchestrator 的流水线资源，反而拖慢编排。

## 设计决策

### Wiring 逻辑不变

Wiring（fanout 边建立、dep_pool 分配、readiness 检查）保留
在 orchestrator 的 `submit_task()` 中通过 SPSC wiring queue
推送给 scheduler。**Orchestrator 代码零改动**，和 8bb837c
基准完全一致。

### Scheduler 侧改动：WFE + Deferred Drain

唯一改动在 scheduler 侧：新增 `wait_orchestration_and_wire()`
方法，在 dispatch loop 之前调用。

## 核心改动

### 1. WFE 替代空自旋

所有 scheduler 线程在 orchestrator 编排期间统一使用 ARM WFE
低功耗等待（包括 thread 0）。orchestrator 写入
`orchestrator_done_` 时产生 cache invalidation 事件，
自动唤醒 WFE。

- 编排期间：所有 scheduler 线程休眠，零 CPU 开销
- 编排完成时：被动唤醒
- SMT 同核场景：WFE 释放执行资源给兄弟线程（orchestrator）

### 2. Deferred Drain（编排后批量 drain）

Orchestrator 编排期间 scheduler 全员休眠，无人消费 wiring
queue。编排完成后，thread 0 在进入 dispatch loop 之前
批量 drain 所有积压的 task：

```cpp
// wait_orchestration_and_wire() 中:
while (!orchestrator_done_) {
    __asm__ volatile("wfe" ::: "memory");
}
if (thread_idx == 0) {
    while (sched_->drain_wiring_queue(true) > 0) {}
}
```

Thread 1+ 唤醒后直接进入 dispatch loop。Dispatch loop 中
原有的 Phase 3（`drain_wiring_queue`）保留不变，继续处理
bulk drain 未完成的剩余 task。

## 改动清单

| 文件 | 改动 |
| ---- | ---- |
| `scheduler_context.h` | 新增 `wait_orchestration_and_wire()` 声明 |
| `scheduler_dispatch.cpp` | 实现该方法；`resolve_and_dispatch` 中 one-time init 之后、dispatch loop 之前调用 |
| `pto_orchestrator.cpp` | 无改动 |

## 时序

```
Orch thread              Sched threads
    |                        |
    | wait_init_complete()   | one-time init
    |   (SPIN_WAIT_HINT)     |   profiling, PMU...
    |                   <----| init_complete_ = true
    |                        |
    | submit_task × N        | wait_orchestration_and_wire():
    |   queue.push × N       |   WFE 休眠（全员沉睡）
    |                        |
    | orchestrator_done_  ---+-> WFE 唤醒
    |                        |
    |                   thread 0: bulk drain_wiring_queue
    |                   thread 1+: 直接往下走
    |                        |
    |                   sched_start_ts = now
    |                   dispatch loop (Phase 1~4)
```

## 约束与注意事项

### Wiring queue 容量

编排期间无人 drain，所有 task 积压在 wiring queue 中。
队列容量为 `PTO2_WRIRING_QUEUE_SIZE`（当前 2048）。
如果单次编排提交的 task 数超过此值，orchestrator 的
`queue.push()` 会 spin-wait 等待空位，但无人消费 →
**死锁**。需要确保 `PTO2_WRIRING_QUEUE_SIZE >= max task count`，
或根据场景加大容量。

### Thread 0 泳道图偏移

Thread 0 在 WFE 唤醒后需要 bulk drain 全部积压 task，
每个 task 需要 `lock_fanout` + `dep_pool.prepend` +
`unlock_fanout`。因此 thread 0 的 `sched_start_ts` 会
**晚于 thread 1+**，在泳道图上表现为 thread 0 起点偏右。
偏移量与 task 数量成正比。

### wait_init_complete 保持原样

Orchestrator 在编排前通过 `sched_ctx_.wait_init_complete()`
等待 scheduler 完成 one-time init（profiling 初始化）。
当前保持原始的 `SPIN_WAIT_HINT()` 不变（非 WFE），以便
和 orch/sched 并行基准进行性能对比。

此处 orch 的 spin-wait 会在 SMT 同核上抢占 sched 做 init
的流水线资源。如果 init 耗时成为瓶颈，可以考虑：
1. 将 `SPIN_WAIT_HINT()` 改为 WFE（让 orch 休眠释放资源）
2. 或调整顺序让 orch 跳过 `wait_init_complete` 直接编图
   （前提：编图不依赖 profiling 初始化，仅在
   `enable_l2_swimlane=false` 时安全）

## 性能打点

```
resolve_and_dispatch():
  1. one-time init           (init_complete_ = true)
  2. wait_orchestration_and_wire()
     - WFE sleep             (不计入 sched 打点)
     - thread 0: bulk drain  (不计入 sched 打点)
  3. sched_start_ts = now    <-- 干净起点
  4. dispatch loop (Phase 1~4, Phase 3 继续 drain 剩余)
  5. sched_end_ts = now      <-- 干净终点
```

`sched_start_ts` 到 `sched_end_ts` 只包含 dispatch +
completion 时间（thread 0 包含少量残余 drain）。
