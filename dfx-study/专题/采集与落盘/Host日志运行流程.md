# Host 与 Device 日志运行逻辑

本文说明 C++ `LOG_*` 日志从调用点到输出端的主路径。这里的 Device 指使用 `unified_log_device.cpp` 的 AICPU 日志；芯片泳道、依赖图等 DFX 数据有各自的采集流程，不属于这条日志调用链。

## 1. 统一入口

业务代码调用 `LOG_ERROR("code=%d", code)` 时，`common/unified_log.h` 中的宏补上当前函数、文件名和行号，再调用 `unified_log_error()`：

```cpp
unified_log_error(__FUNCTION__, "[%s:%d] code=%d", __FILENAME__, __LINE__, code);
```

`unified_log_error()` 在 Host 和 AICPU 目标中分别链接到不同实现。Host 实现直接用 C/C++ 标准的 `va_start` 把 `...` 整理成 `va_list`；AICPU 实现先判断等级，通过后才整理参数。两者调用下层函数后都用 `va_end` 收尾。`common/unified_log.h` 中其他 `LOG_*` 级别使用相同结构。

```text
LOG_ERROR(fmt, ...)
  └─ unified_log_error(func, fmt, ...)
       ├─ Host 版本：va_start → HostLogger::vlog(ERROR, func, fmt, args)
       │    ├─ is_enabled(ERROR) 为 false → 返回，不产生日志
       │    └─ is_enabled(ERROR) 为 true → emit() → Host 输出端（见第 2 节）
       └─ AICPU 版本：is_log_enable_error()
            ├─ false → 返回，不产生日志
            └─ true → va_start → dev_vlog_error(func, fmt, args)
                 ├─ sim：HostLogger::vlog(ERROR, ...)
                 │         ├─ 再检查 is_enabled(ERROR) 为 false → 返回
                 │         └─ 再检查 is_enabled(ERROR) 为 true → emit() → Host 输出端
                 └─ onboard：vsnprintf() → CANN dlog_error(AICPU, ...)
                                └─ CANN 设备日志
```

Host 和 sim 的 `is_enabled()` 读取进程共享阈值：消息级别达到阈值且阈值不是 `NUL` 时才通过。正常初始化后默认阈值是 `TIMING`，因此 `DEBUG`、`INFO` 被过滤，`TIMING`、`WARN`、`ERROR` 可以通过。sim 的 `is_log_enable_error()` 也读取这个阈值；真机的同名函数读取初始化时由 CANN `CheckLogLevel(AICPU, DLOG_ERROR)` 得到的开关。图中以 `ERROR` 为例；其他级别使用对应的检查函数，真机 `TIMING` 使用单独设置的开关并以 CANN `WARN` 输出。

## 2. Host：队列、writer 与 emit()

上一节的 Host 分支中，`unified_log_error()` 把 `va_list` 交给 `HostLogger::vlog()`。`vlog()` 检查进程共享的日志阈值，通过后才调用 `emit()`。

**有哪些对象：** `HostLogger::get_instance()` 首次使用时创建 logger 对象，它本身不是队列。每个进程的 `HostLogAsyncSink` 包含一个多生产者、单消费者（MPSC）有界队列、一个 writer 线程，以及唤醒 writer 的信号量。同进程中调用日志的多个线程是生产者，writer 是唯一消费者；没有单独的日志进程。Host 动态库各有自己的 `HostLogger` 对象，但绑定同一进程日志状态，共用这套 sink。

**何时初始化：** 层级 `Worker.init()` 先设置日志阈值，创建本地子进程时暂不启动 writer；完成最后一次本地 fork 后调用 `HostLogger::start_writer()`。它创建 `HostLogAsyncSink`（初始化队列和信号量）、启动 writer 线程，再把 sink 的提交函数和上下文发布到进程共享日志状态。fork 后的子进程各自启动自己的队列和 writer；直接初始化芯片运行时且无需继续 fork 时，可以立即启动 writer。

**`emit()` 如何使用：**

1. `emit()` 取得单调时钟时间和线程 ID，把级别、函数名及按 `fmt`、`args` 格式化的正文组成一条记录。整条记录按 `_POSIX_PIPE_BUF` 限长，超长时以 `~` 标记截断。
2. writer 尚未启动且允许接收时，`emit()` 同步调用 `write_record_now()`；writer 已启动时，`emit()` 从共享状态取出 `sink_enqueue` 回调和 sink 上下文，将完整记录交给 `HostLogAsyncSink::enqueue()`。它把记录放入队列并唤醒 writer，调用日志的线程不执行文件写入。
3. writer 从队列取出记录，调用 `write_record_now()`：未绑定目录时写 `stderr`；有非空 `CallConfig.output_prefix` 的运行会触发临时目录绑定，此后每个进程追加写自己的 `host.<pid>.log`。队列满或写入失败会计入丢弃数。

进程树可共用日志目录，但每个 PID 各有队列和 writer。普通 Host 日志不会按单次 `run()` 或设备 rank 自动拆分。Python `simpler` logger 初始化后也可转发到这套 Host 日志。

## 3. AICPU：sim 与真机的后端不同

AICPU 版本的 `unified_log_error()` 先调用 `is_log_enable_error()` 检查开关，再把可变参数交给 `dev_vlog_error()`。

- sim AICPU：`dev_vlog_error()` 调用 `HostLogger::vlog(ERROR, ...)`，与所在 Host 进程的日志合流。
- 真机 AICPU：`dev_vlog_error()` 用 `vsnprintf()` 格式化消息，再调用 `dlog_error(AICPU, ...)`，由 CANN 设备日志系统处理。

模拟 AICPU 运行在 Host 进程中，级别检查也读取绑定的 Host 日志状态。真机 AICPU 使用设备侧开关和 CANN `dlog`，不进入 HostLogger 的队列，也不写 `host.<pid>.log`。真机代码的 2048 字节临时缓冲区最多保留 2047 字节正文和结尾的 `\0`，不能据此推断 CANN 最终日志的完整长度。

## 4. 其他入口与边界

`Worker.init()` 从 Python `simpler` logger 取得当前阈值。sim AICPU 使用绑定的 Host 日志状态；真机 AICPU 初始化设备侧日志开关后，由 CANN 负责输出。

上面的代码树以 `common/unified_log.h` 的 `LOG_ERROR` 为例。还有几种入口最终汇入相同后端：

- Python `simpler` 日志通过 `_UnifiedLogHandler` 转发到 HostLogger；未绑定 Host 日志目录时只由 Python 原有 handler 输出，避免向 `stderr` 重复写入。
- Host `[STRACE]` 通过 `unified_log_host_span()` 进入 `HostLogger::log_host_span()`，最后也走 Host 的 `log()`、`vlog()`、`emit()`。
- 部分 orchestration 代码把 `LOG_*` 定义为 `current_runtime()->ops->log_*()`；运行时的函数指针再指向对应的 `unified_log_*()`。

因此，本图覆盖了 `simpler` 自身 C++ `LOG_*` 的主要分流和输出；其他库直接写 `stdout`、`stderr` 或使用自己的日志接口时，不会自动进入这条链。

## 代码入口

- [LOG_* 宏和统一接口](../../../src/common/log/include/common/unified_log.h)
- [Host 接口适配](../../../src/common/log/unified_log_host.cpp)与 [HostLogger 实现](../../../src/common/log/host_log.cpp)
- [AICPU 接口适配](../../../src/common/platform/shared/aicpu/unified_log_device.cpp)
- [sim AICPU 后端](../../../src/common/platform/sim/aicpu/device_log.cpp)与 [真机 AICPU 后端](../../../src/common/platform/onboard/aicpu/device_log.cpp)
- [Python 日志初始化与目录绑定](../../../python/simpler/task_interface.py)
- [Worker 初始化](../../../python/simpler/worker.py)、[Python 日志转发](../../../python/simpler/_log.py)、[Host STRACE 入口](../../../src/common/log/include/common/strace.h)
- [orchestration 日志入口示例](../../../src/a5/runtime/tensormap_and_ringbuffer/orchestration/orchestration_api.h)
- [完整的日志架构与跨动态库状态说明](../../../docs/logging.md)
