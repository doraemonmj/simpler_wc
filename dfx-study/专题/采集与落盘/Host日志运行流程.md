# Host 日志运行逻辑

本文第 1～4 节记录主线的 Host 日志流程：谁产生日志、怎样进入进程内队列、由谁写文件，以及 fork、flush 和退出时发生什么。这里的 Host 日志包括原生 `LOG_*`、`[STRACE]` 和转发后的 Python `simpler` 日志；`chip_swimlane_records.json`、`deps.json` 等走各自的 DFX 采集路径。第 5 节记录独立实现分支 `codex/simpler-output-root` 的目录方案与仍待讨论的问题。

## 1. 三种不同的生命周期

| 层级 | 含义 | 当前产物归属 |
| ---- | ---- | ------------ |
| 进程及其子进程 | `Worker.init()` 建立并复用 Worker、芯片子进程、设备上下文和注册资源，最后 `Worker.close()`；同一进程还可能先后创建多个 Worker | **每进程一个日志队列和一个 writer 线程**，没有单独的“日志进程”；进程树通常共用一个目录，每个 PID 写自己的 `host.<pid>.log` |
| 顶层运行 | 一次 `Worker.submit()` / `Worker.run()`，可产生一个 `RunHandle` 和多个下游任务；串行与重叠运行都可能发生 | 本轮 `CallConfig.output_prefix` 决定诊断产物根目录，但不改变已绑定的 Host logger 目录 |
| 芯片采集 | L3 一次下游 dispatch 在某个设备 rank 上的采集；本地编号 `dN` 由服务该设备的 ChipWorker 进程递增 | 原始文件位于 `<output_prefix>/rankN/dN/`；`dispatch_identity.json` 保存顶层 `run_id`、任务与分组信息及执行该采集的 `host_pid` |

`rankN` 表示设备的逻辑 rank，并非 PID，也不一定等于物理设备 ID：本地 L3 对 `device_ids` 按顺序编号，`rank0` 对应列表中的第一张卡。当前本地 L3 在 `Worker.init()` 时为每张卡 fork 一个 ChipWorker 进程；进程进入任务循环，持续服务同一 Worker 的多次 `run()`，`run()` 不会重新 fork，异常退出也没有自动补建路径。`Worker.close()` 通知其退出并回收。`dispatch_identity.json` 的 `host_pid` 才指向执行某次采集的进程。`d0` 也不是一次顶层 `run` 的同义词；一次顶层运行可含多个 `dN`。不同 rank 的同组采集要靠 `run_id`、`task_slot`、`group_index` 等身份信息关联，不能仅靠 `dN` 名称配对。`pipeline_slot` 和 `pipeline_generation` 标识复用槽位中的代际。SceneTest 会给每个 case 创建 `outputs/<case>_<timestamp>/`，但这个 case 目录不是 Host logger 的生命周期边界。调用方直接指定 `output_prefix` 时，采集文件以该路径为根；API 并未要求多个顶层 run 共用一个输出根目录。

## 2. 从初始化到退出

```text
Worker.init()
  └─ 初始化进程级 Host logger，安装 Python simpler 日志转发
      └─ 本地最后一次 fork 后启动该进程的异步 writer
Worker.submit()/run()，本轮有 output_prefix
  └─ 首次绑定进程树的临时 Host 日志目录
      ├─ 根进程写 host.<root-pid>.log
      └─ 各子进程写 host.<child-pid>.log
芯片任务结束 / Worker.close()
  └─ 有界 flush 各自进程的已接收记录
根进程正常退出
  └─ atexit 清理自动创建的临时目录
```

1. `Worker.init()` 在创建层级子进程前调用 `_initialize_host_log(..., defer_writer=True)`，并把 Python `simpler` logger 接到同一套原生日志。writer 在该进程最后一次本地 fork 之后启动；无 sink 且允许接收时，初始化记录同步写出。若一个已有 writer 的进程后来需要创建新的层级 Worker，`prepare_to_fork()` 会先停止接收新记录、排空并回收线程。fork 子进程继承已绑定的目录，但不会继承一个可运行的 writer 线程；`start_writer()` 为该 PID 新建队列和 writer，发现 `sink_process_pid` 属于父进程时会清零继承来的 pending 和 dropped 计数。
2. 尚无目录时，原生 Host 记录写到 `stderr`。Python `simpler` 记录此时仍由 Python 根 logger 输出，绑定目录后才额外转发到原生 Host 日志；根 logger 的传播仍在。
3. 有 `output_prefix` 的运行会触发 `_bind_host_log_session_directory()`。默认目录是 `${TMPDIR:-/tmp}/simpler-host-logs-<root-pid>-<uuid>/`，由导入 `task_interface.py` 的进程生成路径，实际首次绑定时才 `mkdir(mode=0700)`。如果此前显式调用原生 `_set_host_log_directory(path)`，自动绑定会保留既有位置。原生 `set_log_directory()` 采用“第一个非空路径生效”，后续运行不能用新的 `output_prefix` 改变它；是否绑定目录取决于 `output_prefix` 非空，并不等价于开启所有诊断项。
4. 每个 PID 追加自己的 `host.<pid>.log`。生产者将记录送入该进程的有界队列，后台 writer 写到固定目录；绑定后若文件打不开或写失败，记录计入丢弃数，不转去 `stderr`。队列已满、争用、未接收也分别计数。`flush` 有超时，只等待调用进程已接收的记录；芯片任务完成、`ChipWorker.finalize()`、`Worker.close()` 和部分子进程 `os._exit()` 路径有相应的 flush 点。它不等于整个进程树或某个顶层 run 的全局完成屏障。
5. 自动临时目录只在其创建者进程**正常解释器退出**时清理；外部显式绑定的目录不由这一清理函数删除。异常终止时不应假定 `atexit` 已执行。

这解释了当前的留存问题：采集原始文件可以长期保存在 `output_prefix` 下，而完整 Host 进程日志可能仍在临时目录，并在正常退出后消失。多个串行或并行运行可同时把记录写入同一个 `host.<pid>.log`，文件名本身不标识运行。

### 从调用点到文件的具体路径

```text
Python simpler logger ── _UnifiedLogHandler（目录已绑定才转发） ─┐
C++ LOG_* / STRACE ── 阈值过滤 ──────────────────────────────┤
                                                            ↓
         HostLogger：采样单调时钟、格式化为一条完整记录
                                                            │
              ┌─ 无 sink 且允许接收：同步 write_record_now
              └─ writer 已启动：本进程 HostLogAsyncSink::enqueue
                                     ↓  MPSC 环形队列
                      本进程的 writer 线程：pop → write_record_now
                                     ├─ 未绑定目录：stderr
                                     └─ 已绑定目录：<目录>/host.<pid>.log
```

原生 `_task_interface` 持有进程共享状态；Host 侧动态库各自编译有 `HostLogger`，通过 `simpler_host_log_bind_state` 指向同一 `SimplerHostLogState`。只有状态所有者创建该进程的 sink 和 writer，动态库中的 logger 通过共享的 `sink_enqueue` 回调提交。同一进程里**调用 simpler 日志接口**的多个线程和 Host 动态库因此汇入同一队列；fork 后的子进程另建队列和 writer。其他库直接写 `stdout`、`stderr` 或自己的日志系统，不会自动进入 simpler 的队列。

### 队列的生产、消费和顺序

1. **筛选与格式化。** `LOG_*` 按进程阈值判断是否产生日志；`[STRACE]` 需要 `TIMING` 级别且编译时启用 Host trace。`HostLogger::emit()` 取单调时钟和线程 ID，组成一整行记录；`log_host_span()` 也把 span 格式化为一条记录。Python `simpler` handler 仅在 Host 日志目录已绑定时转发，且 Python logger 的传播仍可让原有控制台 handler 收到记录。
2. **提交。** 当进程尚无 sink 且接收未关闭时，`emit()` 直接调用 `write_record_now()`，不经过队列；fork/停机的过渡阶段若未能获准提交，则按 `not_admitted` 计数。writer 建立后，生产线程通过共享的 `sink_enqueue` 提交到本进程 `HostLogAsyncSink`。`QueuedRecord` 只有 `size` 和完整文本 `data`，没有 `run_id`、`output_prefix`、rank 或目标文件。每进程环形队列有 4096 个槽，单条记录按 `_POSIX_PIPE_BUF` 限长；普通超长记录以 `~` 标记截断。
3. **占位与发布。** 多个生产者用序号及 CAS 争取队列位置，最多尝试 1024 次。占到位置后复制记录、增加 `pending_record_count`，再以 release 顺序发布该槽，并通过信号量唤醒 writer。队列满或争用超出预算时直接计数丢弃；稳态生产者不做文件 I/O，也不等待 writer 排空。
4. **消费与写入。** 每进程唯一的 writer 线程按队列位置取出记录，调用 `write_record_now()`。生产者可能先占到较早位置、却比后来的生产者更晚发布；writer 会等待较早位置就绪，保持队列位置顺序。记录的时间戳在入队前取得，因此跨线程文件行序不必严格等于时间戳顺序。写完或写失败后减少 `pending_record_count`，通知等待 flush 的线程。

writer 查看进程共享状态中已绑定的目录：未绑定则写 `stderr`；绑定后追加到 `<目录>/host.<pid>.log`。文件以 `O_APPEND` 打开并缓存 FD。状态里只有一个 `log_directory[1024]` 和一个 `log_directory_bound`；`set_log_directory()` 用原子状态 `0 → -1 → 1` 发布路径，读者只认完成态 `1`，避免读到写了一半的路径。首次绑定后目录不会迁移。绑定后写失败按 `output_failed` 记损失，不回退到标准错误输出。

`pending_record_count` 是已接收而 writer 尚未处理完的记录数，`dropped_record_count` 汇总 `queue_full`、`claim_exhausted`、`output_failed`、`not_admitted`。`flush(timeout_ms)` 等生产者退出提交区且 `pending_record_count == 0`；返回成功表示本进程这批已接收记录处理完，**不保证没有丢弃，也不能把混在同一队列中的两轮运行分开**。停止 writer 的边界会用 `[HOSTLOG_DROPS]` 报告增加的丢弃数；读取文件时应连同计数看完整性。日志阈值默认 `TIMING`，更高阈值可能使对齐所需的 `[STRACE]` 根本不产生。

## 3. 采集目录里已有的 Host 相关文件

同一个 case 的一次 L3 采集，可能同时涉及以下两个位置（方括号表示满足条件时才生成）：

```text
${TMPDIR:-/tmp}/simpler-host-logs-<root-pid>-<uuid>/
├── host.<root-pid>.log
├── host.<服务rank0的进程PID>.log
└── host.<服务rank1的进程PID>.log

outputs/<case>_<timestamp>/             # SceneTest 的 output_prefix 示例
├── rank0/d0/
│   ├── chip_swimlane_records.json
│   ├── dispatch_identity.json
│   └── [host_clock_alignment.<pid>.log]
├── rank1/d0/
│   ├── chip_swimlane_records.json
│   ├── dispatch_identity.json
│   └── [host_clock_alignment.<pid>.log]
└── [l3_swimlane.json]                    # SceneTest 后处理产物
```

这里的两个 `d0` 分别是两个设备 rank 的本地采集编号；`rank0`、`rank1` 标识设备位置，`host.<pid>.log` 按执行进程的 PID 命名。Host 进程日志保存在另一处，并不会因为 case 目录已创建而自动复制到 case 下。

成功执行并完成验证的 HBG L3/L4 采集，在 Host phase records 已采集完、输出前缀存在且日志阈值允许 `TIMING` 时，可在本轮采集目录导出 `host_clock_alignment.<pid>.log`。Level 3 还需要 `SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE=1`；Level 4 会自动启用。`needs_clock_alignment()` 的实际条件是泳道级别至少为 `SCHED_PHASES`、`output_prefix` 非空、Host phase records 已 armed 且 finished；最终导出还要求设备已启动、执行与验证成功。实现先记下本次调用开始前 `host.<pid>.log` 的文件偏移，结束时 flush，再从该偏移起按 `pid` 和 `inv` 筛选 `chip.run`、`chip.run.runner_run`、`device_wall`、launch 与 phase 标记；要求前三类窗口齐全，最后通过临时文件加 rename 发布。

这个文件仅含当前芯片调用所需的时钟对齐标记，**不是**普通日志和完整 Host 调用树的副本。TMR 当前没有相同的采集目录导出路径；只保留 `rankN/dN` 的 JSON 文件，不能据此推断完整 Host 原始日志也已留存。`[STRACE]` 中 `(pid, inv)` 标识一次芯片侧 `simpler_run` 调用；它与顶层 `run_id` 不同。

SceneTest 在做多 rank 泳道后处理时，若各采集目录都有上述对齐文件，就让转换器发现这些局部文件；否则会在进程仍运行时 flush 当前进程的日志，并从已绑定目录收集 `host.*.log` 传给转换器。转换器的默认发现范围是输入根下的 `host.*.log`、根目录及 `rank*/d*/` 下的 `host_clock_alignment.*.log`；也接受显式 `--host-log`。它用采集 sidecar 的身份、`host_pid` 与 Host 时间窗口确定跨 rank 放置关系。`[STRACE]` 的 Host 时间用 `CLOCK_MONOTONIC` 纳秒，同一主机上的进程可以比较；标有 `clk=dev` 的设备相对时间不能直接跨调用比较。SceneTest 对一个转换目标默认生成 `l3_swimlane.json`，对多个目标生成 `l3_swimlane_run<run_id>_task<task_slot>.json` 等文件；这些后处理结果可能用过即将被清理的 Host 原始数据，成功生成不等于原始日志已随输出目录保存。

## 4. 多个运行时的实际输出

一个进程的 Host logger 只有一个已经绑定的目录；`output_prefix` 是本轮诊断配置，当前只用于触发首次绑定，不会在后续轮次切换日志文件。writer 依 PID 写 `host.<pid>.log`，不会按 `run_id`、rank 或 `dN` 拆分；队列中尚未写出的记录也没有随记录保存的输出目录。

例如，同一 Worker 先后提交 `A(output_prefix=A)` 和 `B(output_prefix=B)`，两轮各自的采集文件写到对应的 `output_prefix`，服务同一设备的 ChipWorker 进程在这两轮间不重启，其 Host 记录继续追加到相同 PID 的进程日志。该进程的 `diagnostic_capture_index` 也持续递增，第一次诊断采集为 `d0`，第二次为 `d1`，不会因为更换 `output_prefix` 重置。若两轮重叠，记录在文件中也可能交错；即使串行，初始化、注册、close 等记录也不属于某个芯片采集。`[STRACE]` 的芯片调用可以用 `(pid, inv)` 分组，但普通 Host 日志行的通用前缀只有单调时间、线程、级别和函数名，没有统一的顶层 `run_id` 字段。

## 代码入口

- [Python 初始化、绑定与退出清理](../../../python/simpler/task_interface.py)
- [Worker 初始化、提交、芯片任务和关闭](../../../python/simpler/worker.py)
- [进程级共享日志状态](../../../src/common/log/include/common/host_log_state.h)与[写入实现](../../../src/common/log/host_log.cpp)
- [原生 Python 绑定](../../../python/bindings/task_interface.cpp)与[Python 日志转发](../../../python/simpler/_log.py)
- [采集目录的 Host 时钟对齐日志导出](../../../src/common/platform/include/host/host_clock_alignment_log.h)
- [SceneTest 输出目录与泳道后处理](../../../simpler_setup/scene_test.py)及[转换器的 Host 日志发现逻辑](../../../simpler_setup/tools/swimlane_converter.py)
- [STRACE 格式和读取说明](../../../docs/dfx/host-trace.md)

## 5. 独立实现分支的目录方案

本节对应 `codex/simpler-output-root` 工作树中的未提交实现，不属于本 study 分支或主线。上文第 1～4 节描述主线现状；底层 Host 队列和单进程固定 sink 的机制没有改变。

调用方可在创建 Worker 时提供 `output_root`，`Worker.init()` 在 fork 前把进程级 Host sink 绑定到 `<output_root>/host_logs/<session-id>/`。每个进程继续写自己的 `host.<pid>.log`，正常退出后保留。若同进程先绑定了别的目录，或后续 Worker 选择不同根目录，会明确报错。未设置 `output_root` 时，原先由非空 `CallConfig.output_prefix` 触发临时目录并在退出时清理的行为继续存在。

启用诊断且没有显式 `CallConfig.output_prefix` 时，Simpler 自动建立 `<output_root>/captures/<worker-id>/`。L3 同一 Worker 的多次顶层 run 共用这个目录，芯片采集仍分在 `rankN/dN/`；L2 每次 run 在其下建立独立的 `run-<id>/`，防止固定文件名覆盖。SceneTest 的独立运行入口仍为每个 case 生成 `outputs/<case>_<timestamp>/`，但其 Worker 的日志写入 `outputs/host_logs/<session-id>/`。

L3 每份 `dispatch_identity.json` 新增相对于该采集目录的 `host_log_directory`，离线转换器可按所选采集的身份文件定位原始 Host 日志。同一会话的多个 run 仍共享进程文件，因此普通日志行无法仅凭这个目录引用精确归属某个 run；只读 stderr 的测试仍需考虑进程 sink 已转到文件，这也是 [#2524](https://github.com/hw-native-sys/simpler/issues/2524) 所讨论的独立问题。

### 改动前方案草案（留作对照）

以下保留改动前的**方案草案**，其中目录示意不是当前实现。目标是从顶层、某个 Worker、某个 rank 或单次 `dN` 采集目录进入，都能找到相应数据。

### 5.1 区分目录树与日志生命周期

在旧实现中，诊断目录由上层 `output_prefix` 逐层派生：有父级的 Worker 会追加 `<层级名称><worker_id>/`，芯片采集再追加 `rankN/dN/`。下层看到的 `output_prefix` 已经是上层目录的子目录，不能把每一层拿到的路径视为互不相干的根。顶层 `Worker.run()` 的输出目录由调用方传入；不同 run 可能使用不同顶层目录。

```text
<明确指定的 output_root>/
└── sessions/<session_id>/                 # 拟新增：进程树共用的日志会话
    ├── host_logs/
    │   ├── host.<pid-A>.log               # 同一进程可包含多次 run 的记录
    │   └── host.<pid-B>.log
    └── runs/
        ├── <run-A>/                       # 拟由调用方或运行时分配唯一目录
        │   └── node0/rank0/d0/
        │       └── dispatch_identity.json
        └── <run-B>/
            └── node0/rank0/d1/           # 同一 ChipWorker 可继续编号
                └── dispatch_identity.json
```

图中的 `node0` 仅示意有父级的 L3 Worker 命名空间；直接以 L3 为顶层时，可以直接出现 `rankN/dN/`。`runs/` 也是建议的组织层，不是现有 SceneTest 的实际目录。现有 SceneTest 仍按 case 创建 `outputs/<case>_<timestamp>/`；若要统一到上图的 `output_root`，需明确由谁分配和传递这个根，不能靠下层路径向上猜父目录。

### 5.2 先解决留存与定位

1. 在可能产生诊断日志、创建子进程之前确定一个明确的 `output_root` 与唯一的 `session_id`，把进程级 Host 日志绑定到 `sessions/<session_id>/host_logs/`。日志文件仍按 PID 追加，避免在并行 run 之间切换同一进程的日志目录。若调用方只在每次 `run()` 提供彼此无关的 `output_prefix`，就无法可靠选出一个覆盖全部 run 的共同根；需要新增会话级根目录约定，或接受日志存于独立位置并用引用关联。
2. 在采集目录的身份文件中记录会话身份和可解析的日志引用，保留现有 `run_id`、任务身份、rank、`dN` 和 `host_pid`。日志引用还应区分 PID 复用后的进程实例，并考虑移动整棵输出树后仍可解析。并行执行时要给 run／会话目录唯一名称，身份文件以原子发布方式写入；从任一层目录打开时，沿其子树中的身份信息发现采集及对应日志，不根据固定父目录深度猜测。
3. 进程级日志目录的生命周期不能简单等同于一个 Worker：同一 Python 进程可能先后创建多个 Worker，现有 Host logger 又是首次非空目录绑定后不可更改。若要为每个 Worker 单独绑定目录，需要先改变该进程级约束；首阶段应把 `session_id` 定义到实际共享 Host logger 的生命周期，并让相关 Worker/run 引用同一会话。

### 5.3 再解决精确筛选

仅凭 `host_pid` 可以找到进程日志，不能把其中普通日志精确归给某个 `run` 或 `rankN/dN`：同一进程可跨 run 复用，并行任务的记录还可能交错。要支持“只看本次芯片采集的日志”，需在**产生日志时**带上稳定的 run／dispatch／采集身份，并沿任务下发、线程与子进程边界传播；异步队列中也必须保留该身份。没有运行上下文的初始化、注册和关闭日志应归入会话级，而不是硬塞进某个 run。只按时间窗口筛选可作辅助定位，不能作为并行场景的精确归属依据。

接口、绑定时机和目录引用已按本节开头所述实现。后续仍需讨论 IDE 的入口与发现规则，以及是否为普通日志记录传播 run 身份。现有 HBG 的 `host_clock_alignment.<pid>.log` 只是局部对齐标记，不能代替完整 Host 日志。
