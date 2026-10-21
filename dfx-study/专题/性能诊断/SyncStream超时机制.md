# SyncStream 超时机制与运行路径

本文按 simpler 的 main@545c05a43 说明 onboard 路径。先梳理与 SyncStream 直接相关的三种超时：设备侧 scheduler 检查调度是否持续无进展，STARS / CANN 限制 op 执行时间，Host 的 ACL 等待限制单次 event 或 stream 同步。它们的计时起点不同，并非从 Worker.run() 开始的一只总计时器。

~~~text
Worker.init() → runner 绑定设备 → 配置超时预算
Worker.run() → submit(...).wait()
  ├─ submit：排入 AICPU、AICore 内核及本次 run 的完成 event
  │   ├─ scheduler：检查最近一次调度进展距今多久
  │   └─ STARS / CANN：监控 op 执行
  └─ wait：Host 等完成 event，必要时再同步整条 stream
~~~

默认预算依次是 **20 s / 45 s / 50 s**。可以分别用 SIMPLER_SCHEDULER_TIMEOUT_MS、SIMPLER_OP_EXECUTE_TIMEOUT_US、SIMPLER_STREAM_SYNC_TIMEOUT_MS 覆盖，单位依次为毫秒、微秒、毫秒。Host 在 runner 首次绑定设备时解析配置，要求 scheduler < op-execute < stream-sync，且 stream-sync > scheduler + 1500 ms；无效单项值使用默认值，组合顺序无效时 Host 的 op-execute 和 stream-sync 回退默认值，scheduler 覆盖不下发。[配置解析（第 84–137 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp) · [顺序校验（第 141–204 行）](../../../src/common/platform/include/host/runtime_timeout_config.h)

## 1. Scheduler 无进展超时

### 判断前：预算与进展记录

Host 初始化时把有效的 SIMPLER_SCHEDULER_TIMEOUT_MS 传给 AICPU；没有有效覆盖时，设备使用默认的 **20 s**。以 A2/A3 tensormap_and_ringbuffer 为例，每个调度线程进入 resolve_and_dispatch() 时记录 last_progress_ts，并将预算换算为设备计数器周期数。[预算传递（第 1857–1863 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp) · [计时起点（第 993–1008 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp)

每轮先检查任务完成，再尝试派发；任务完成、派发或回收推进会刷新最近进展时间。关键代码简写如下：[调度循环（第 1010–1049、1299–1305、1381–1422 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp)

~~~cpp
bool made_progress = false;  // 每轮重新观察
check_running_cores_for_completion(/* ... */, made_progress, /* ... */);
dispatch_ready_tasks(/* ... */, made_progress, /* ... */);
if (made_progress) last_progress_ts = get_sys_cnt_aicpu();
~~~

### 判断点：每轮调度末尾的无进展分支

完成检查、虚拟任务处理和派发尝试后，才到这轮的超时判断。代码简写如下：[判断代码（第 1381–1474 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp)

~~~cpp
if (made_progress || sched_->drain_pending_ring_advances()) {
    last_progress_ts = get_sys_cnt_aicpu();
} else {
    uint64_t idle_elapsed = get_sys_cnt_aicpu() - last_progress_ts;
    if (idle_elapsed > scheduler_timeout_cycles) {
        bool self_owns = self_owns_running_task(thread_idx);
        bool global_stuck = !self_owns && total_tasks_ > 0 &&
                            completed_tasks_.load(std::memory_order_relaxed) < total_tasks_ &&
                            no_thread_owns_running_task();
        if (self_owns || global_stuck) {
            timeout_rc = handle_timeout_exit(/* ... */);
            break;
        }
        last_progress_ts = get_sys_cnt_aicpu();
    }
}
~~~

判断的是**距最近一次进展的时间**，不是 run 总耗时。self_owns 表示本线程持有 RUNNING 任务但无进展；global_stuck 表示还有任务未完成、且所有线程都没有 RUNNING 任务。若其他线程还在运行任务，当前空转线程重新计时；预算过半的诊断日志只是预警。

### 超时后如何处理

handle_timeout_exit() 的核心顺序如下：[设备侧处理（第 433–505 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp)

~~~cpp
StallClassification cls = classify_stall_reason();
LOG_ERROR("TIMEOUT_EXIT ... sub_class=...");  // AICPU 设备日志
if (latch_scheduler_error(header, thread_idx, SIMPLER_ERROR_SCHEDULER_TIMEOUT)) {
    // 首次报错者将停滞类别、任务数和卡住的位置写入共享状态。
}
if (begin_emergency_shutdown()) {
    log_shutdown_stall_snapshot(/* ... */);  // 记录各线程、任务和核心状态
    signal_emergency_shutdown(runtime);  // 请求所有 AICore 退出
}
return -SIMPLER_ERROR_SCHEDULER_TIMEOUT;  // -100
~~~

停滞现场有三个去向：

- **设备日志（plog）**：TIMEOUT_EXIT 和 SHUTDOWN_SNAPSHOT 记录停滞类别、任务状态、各线程负责的核心状态等详细诊断, 由LOG_ERROR 等日志调用写入。[现场日志（第 343–373 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp)
- **本次 run 的结果记录**：共享状态只保存错误码和少量摘要；AICPU 收尾时将它复制到本次 run 专属的结果区，Host 读取后打印 sub_class、任务数和卡住的位置。header 保存的是错误摘要，收尾时会复制进本次 run 的结果记录。[结果快照（第 1091–1118 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp) · [Host 读取（第 1099–1133 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp)
- **可选 ArgsDump**：启用时，在请求 AICore 退出前抓取运行中任务的部分输出；调度线程退出时冲刷缓冲区，Host 的错误回收路径尝试导出诊断文件。[抓取（第 464–490 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp) · [错误回收（第 981–995 行）](../../../src/a2a3/platform/onboard/host/device_runner.cpp)

调度循环的 break 只结束循环：线程先冲刷诊断缓冲区，再从 resolve_and_dispatch() 返回 -100；调用者随后完成核心退出和本次 run 的收尾。[调度线程返回（第 1500–1559 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp) · [调用者收尾（第 1027–1088 行）](../../../src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp)

## 2. STARS / CANN op-execute 超时

### 判断前：配置 op 执行预算

**设置调用点在初始化阶段**：runner 首次绑定自有设备时，走 `attach_current_thread() → configure_aicore_op_timeout() → aclrtSetOpExecuteTimeOutV2()`。它解析 SIMPLER_OP_EXECUTE_TIMEOUT_US（默认 **45 s**）并设置设备全局的 op 执行预算；这一步不等待或判断超时。借用调用方设备上下文时不改写该设备的预算。[初始化与设置（第 1581–1607、1657–1671 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp)

~~~cpp
uint64_t actual_timeout = 0;
int rc = aclrtSetOpExecuteTimeOutV2(timeout_config_.op_execute_timeout_us, &actual_timeout);
// 成功时记录 requested 和 actual；失败时记录设置错误。
~~~

### 判断点：设备执行 op 期间

这里的 **op** 可以简单理解为一次 launch 下发的设备任务。以 AICPU 为例，下发的 `simpler_aicpu_exec` 会调用 `aicpu_execute()`，在其中处理多个 Simpler task；AICore 内核则是另一次 launch。[AICPU 入口（第 150 行）](../../../src/a2a3/platform/onboard/aicpu/kernel.cpp) · [AICore 下发（第 3293 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp)

op 提交后，由 **STARS / CANN** 判断其执行是否超过预算；Simpler 不自行计时，也不使用 scheduler 的 `last_progress_ts`。超时后底层可终止 op 并报告错误，设备日志可能出现 `HandleTaskTimeout`；AICPU op 被终止时还可能出现 `kill aicpu-sd`。这说明 op 执行超时，不能单凭此认定死锁。[超时测试内核（第 27–35 行）](../../../tests/st/aicore_op_timeout/kernels/aic/kernel_hang.cpp)

### 超时后如何处理

**等待调用点在 run 回收阶段**：`simpler_wait_run() → drain_execution() → reap_run() → wait_run_fence()`。最后通过 `RunCompletionFence::wait()` 调用 **aclrtSynchronizeEventWithTimeout()**，等本次 run 的 AICPU、AICore 完成 event。每次 event 等待默认最多 **50 s**，可能收到设备错误，也可能等到自己的超时；完成 event 不齐或执行结果仍不明时，改用 stream 同步。[回收入口（第 2297–2321 行）](../../../src/common/platform/onboard/host/c_api_shared.cpp) · [ACL 接口绑定（第 183–187 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp)

这两个接口分工不同：`aclrtSetOpExecuteTimeOutV2()` 让底层监控并处理单个 op 超时；`aclrtSynchronizeEventWithTimeout()` 让 Host 有界等待并接收结果。**没等到 event 不等于某个 op 执行超时**，所以后者不能代替前者。等待可能返回 507000、507018 等设备错误，也可能在 stream 同步时返回 507046；具体原因仍需结合 ACL error detail 和设备日志。

`reap_run()` 收到非零等待结果后走通用失败路径，简写如下：[等待与失败处理（第 964–1003 行）](../../../src/a2a3/platform/onboard/host/device_runner.cpp)

~~~cpp
int rc = wait_run_fence(prepared);
if (rc != 0) {
    log_run_failure(prepared, rc);
    recover_device_or_mark_unusable(rc);
    // 收尾诊断资源后，返回原始错误码。
    return rc;
}
~~~

恢复时先禁止这个 runner 接收新任务。自有设备上下文尝试有界的设备同步，随后在 `finalize()` 中强制重置；借用上下文只将 runner 标记为不可用，设备恢复由调用方负责。这也是**等待失败的通用恢复路径**，并非凭错误码确认了 op 超时。[恢复处理（第 1077–1118 行）](../../../src/a2a3/platform/onboard/host/device_runner.cpp)

## 3. Host 完成 event / SyncStream 等待超时

### 判断前：记录本次 run 的完成位置

Host 在 runner 首次绑定设备时解析 SIMPLER_STREAM_SYNC_TIMEOUT_MS，默认等待预算是 **50 s**。`Worker.run()` 提交 AICore、AICPU 内核后，在各自的 stream 上记录完成 event；event 标记的是**本次 run 的位置**，整条 stream 还可能有后续工作。[内核提交与 event 记录（第 860–909 行）](../../../src/a2a3/platform/onboard/host/device_runner.cpp)

### 判断点：run 回收时的 ACL 等待

`wait_run_fence()` 先检查本次 run 的两个完成 event 是否齐全。齐全时，`RunCompletionFence::wait()` 依次等待 AICPU、AICore event；其 `ops_.wait` 最终调用 `aclrtSynchronizeEventWithTimeout()`。每个 event 都单独使用最多 **50 s** 的等待预算，超时由 ACL 判断，Simpler 不比较 run 的总耗时。[event 等待（第 271–288 行）](../../../src/common/platform/include/host/run_completion_fence.h)

后续分支简写如下：[等待与结果判定（第 3714–3799 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp)

~~~text
if (没有完整的完成 event) return sync_stream_pair(/* AICPU, AICore */);
if (event 等待返回错误) return 错误码;
读取本次 run 的结果记录;
if (记录明确成功) return 0;
return sync_stream_pair(/* AICPU, AICore */);
~~~

`sync_stream_pair()` 依次调用 `aclrtSynchronizeStreamWithTimeout()` 等 AICPU、AICore stream，每次也是独立的 50 s 预算。结果记录读取失败时，仍会同步 stream，但保留并返回原先的读取错误。正常成功路径优先用 event 和结果记录判断本次 run，避免为后续工作等待整条 stream。[stream 同步（第 3555–3588 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp)

### 超时后如何处理

event 等待超时返回 `ACL_ERROR_RT_EVENT_SYNC_TIMEOUT`（507047）；stream 同步超时返回 `ACL_ERROR_RT_STREAM_SYNC_TIMEOUT`（507046）。Host 记录等待预算、设备或 stream 信息及 ACL error detail，然后将错误交给 `reap_run()`：记录本次 run 失败、标记 runner 不可用并尝试恢复，最后返回原错误码。[等待错误处理（第 3730–3744 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp) · [run 回收（第 964–1003 行）](../../../src/a2a3/platform/onboard/host/device_runner.cpp)

这里的 **50 s 是单次 event 或 stream 等待上限**，不是整个 `Worker.run()` 的总时限。等待超时只说明 Host 没在预算内等到完成，不能据此认定 op 执行超时；还要看设备侧是否先出现 scheduler 或 STARS / CANN 错误。SIMPLER_STREAM_SYNC_TIMEOUT_MS 作用于 runner 保存的 run 等待预算；部分初始化、注册等控制流仍直接使用编译常量。[控制流同步（第 1871–1887、2350–2374 行）](../../../src/common/platform/onboard/host/device_runner_base.cpp)

三种默认预算按 **20 < 45 < 50 s** 排列，让设备侧先有机会留下诊断，再由 Host 的有界等待收尾；由于起点不同，实际先出现的错误不由数字大小单独决定。

## 4. 张量数据依赖等待超时

## 5. 任务环与资源池回收等待超时

## 6. RunHandle.wait(timeout) 调用方等待超时

## 7. 初始化与注册阶段的控制 stream 等待超时
