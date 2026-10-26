# A5 onboard L0 泳道：分阶段实施方案

**范围**：当前 A5 onboard 代码和这台 36-cluster EVB；不包含 sim。参考现有 PMU、chip swimlane 等 DFX collector 的 run 管理，增加独立的 L0 硬件数据源。测试全程使用 device-0 锁与超时，不执行服务器或设备重启、reset。

**当前实现状态（2026-10-26）**：代码已接通普通 ST、108 核自管 GM、L0 raw/JSON 导出及 chip swimlane task 关联。单任务 MIX 用例已完整通过；中等规模 paged attention 的计算和 golden 已通过，但其 L0 capture 因 channel 17 无法再次启动而未完成。PA 多任务关联只经过离线构造数据测试，尚无真实 PA L0 数据证明。每核固定 1 MiB，当前写指针检查不能排除硬件回卷覆盖；大数据量采集完整性未验收。以下首版设计和 2026-10-23 记录保留为历史方案，实际行为以代码及 [PA 实验记录](l0_paged_attention_attempt_2026_10.md)为准。

## 2026-10-26 全物理核实测

- 现有宏开关与逐 run `enable_l0` 路径已接入普通 onboard ST。专用 manual ST 改为一个 36-block MIX cohort，AIC、AIV0、AIV1 分别写独立输出槽并发出 region 10/11 标记。ST golden 检查 108 个输出槽，channel 20 的运行通过（`task_20261026_093634_716644`，6.53 s）。
- **先预检并配置 108 个物理子核的自管 GM，再调用 `prof_drv_start`**，启动后逐核回读 `gen/base/len`，run fence 后逐核撤销并确认，才读取按写指针界定的原始前缀。这次 108/108 核都配置成功、都产出 L0 原始数据：导出前缀合计 48,544 B，其中 19,744 B 为初始化 `A5` 前缀，**本次新写入的连续记录为 28,800 B**；解出 360 条 region 10/11 事件，其中 36 个 AIC 各 2 条、72 个 AIV 各 4 条；每条事件的 block 都对应执行的 cluster。108 个从 START_STAMP 起始的实际写入窗口独立解码后，与 live JSON 的事件逐份一致。完整原始字节、逐核元数据、SHA-256 清单和 [窗口审计](../../../src/a5/platform/testdata/l0_onboard/channel20_all_physical_20261026/l0_window_audit.json)均已归档。
- 先前的“启动 channel 再配置”顺序，在 channel 24 的 core 0 配置返回 `-2`；channel 26 和 23 均见固定 73 核在启动后 `gen=1,len=0,base=0`，只允许其余 35 核安全配置和采集。channel 23 对 core 0 等待约 2 秒仍为 `gen=1`，排除了短暂启动瞬态。相关两次原始数据也已归档。此现象证明**当前 EVB 上配置顺序影响全核覆盖**，并不能单凭这几次实验断言驱动为何把 73 核置位。
- 全核**原始采集与 pipe 标记解码**已验证；精确 runtime task ID 关联、跨 run channel 重开、环形回绕和 viewer 泳道仍未验证。channel 20/23/24/26 都已使用，当前驱动不保证它们可以再次启动。此节覆盖下方 2026-10-23 的首版历史状态，后续章节是原计划与未完成事项。
- **跨会话写指针残留**：channel 20 的 35 个带 `A5` 前缀的核，前缀长度均等于 channel 23 同核上次写指针；channel 23 对 channel 26 也是 35/35，三个更早的单核样本也相符。当前代码已改为启动后记录每核 `wptr`，完成后按 `[start,end)` 判断本次新数据，同时保留 `[0,end)` 原始文件供复核。该修正尚未再次上板验收；硬件为何未按写 0 的意图复位指针仍待查。

## 2026-10-23 首版实施状态

- `SIMPLER_A5_ONBOARD_L0` 默认关闭；开启后，普通 `SceneTestCase` 可用逐 case 的 `config.enable_l0` 请求单核采集。`CallConfig.enable_l0` 为保持 Python/native wire ABI 一致，在各平台结构中保留该配置字段；真正的 channel、GM、AICPU MMIO 及 host 生命周期代码只在 A5 onboard 宏开启时编译。A5 onboard 宏关闭时会拒绝请求。
- 已接入单 run、单物理 AIV 核 106、channel 27 的前置配置与完成后导出：run 前同步预检、启动 channel 并配置自管 GM；run fence 后撤销 GM 地址、保存有效 `.bin` 和解码 `.json`。新数据的 JSON 标记 `source=live`。若前置采集不可用且确认没有悬挂 GM 地址，普通 ST 继续运行，同时写 `l0_status.json`；若无法确认撤销则阻止后续工作并保留 GM。当前版本不做 task ID 归属、多核、环形缓冲或 viewer 展示。
- 这台 EVB 上普通 A5 ST 在新构建中通过；L0 专用 ST 两次在 `prof_drv_start(0,27)` 返回失败，未进入 kernel，也没有新的 L0 字节。由于前面 channel 27 已用于独立实验，不再反复启动它。L0 专用 ST 已设为 manual。归档的 128 字节原始记录用同一解码器复核，生成明确标记 `source=archive` 的 [事件文件](../../../src/a5/platform/testdata/l0_onboard/channel27_core106_two_tasks.json)：8 条 pipe 标记，两次独立实验任务各有 MTE3 region 起止对。
- 已确认 A5 宏开启构建成功、普通 onboard ST 通过、归档解码成功。**普通 ST 中实时 L0 数据的端到端验收尚未完成**；需要一个可正常启动的 channel 会话才能核实采集与撤销路径。本文后续“大步骤”仍是扩展计划，不代表这些步骤已经完成。

## 已有证据与边界

- 当前代码上的单核实验已经证明：host `prof_drv_start` 成功后，AICPU 给 AIV perfmon 配置自管 GM 缓冲区，HWTS 在 kernel kickstart 时置 `en`，硬件把 `mark_stamp` 记录写入该 GM。直接 `prof_channel_read` 为 0 并不代表没有数据。
- 同一次 channel 27 启动、同一个 core 106 缓冲区，连续两个 72-block AIV 任务分别把写指针推进到 64 和 128 字节。128 字节解出 8 条标记，两个任务的 MTE3 起止标记都完整，按**字节窗口**归属与 region ID 相符。[归档原始记录](../../../src/a5/platform/testdata/l0_onboard/channel27_core106_two_tasks.bin)、[归档说明与校验值](../../../src/a5/platform/testdata/l0_onboard/README.md)；完整寄存器与清理记录见 [实验报告](../../../src/a5/platform/docs/l0_evb_onboard_probe_2026_10.md)。
- 这证明的是**一个受监控 AIV 核、两个顺序任务**。真实 ST 的调度任务 ID、多个核、并发任务、环形回绕及 FIN 时在途 DMA，均未在当前驱动上验证。旧版 #905 的全核结果是设计参考，不能直接视作当前 EVB 的结论。
- 当前 `DeviceRunner` 在普通 run 中先提交 AICore worker，再提交主 AICPU 调度核。因此 L0 的缓冲区配置必须有一个**独立于主调度入口的前置完成点**，放在 AICore 首个任务可运行之前。

## 首个可交付版本：单核、单 run、同步采集

这个版本只要求贯通 **普通 onboard ST → perfmon 自管 GM → 原始文件 → 解码事件**。它不是全核 collector：只监控一个已验证的物理 AIV 核，使用有明确 `mark_stamp` 的 72-block ST 保证该核实际执行。ST 的成功结果与 L0 的 raw、JSON 放在同一 `output_prefix` 下；同时打开现有 chip swimlane 时，尝试用同核任务记录给事件关联真实 task ID，匹配不唯一就保留 unmatched。

1. **开关和入口**：只用 `SIMPLER_A5_ONBOARD_L0` 一个新编译宏，默认 `0`；宏 `1` 时由独立的逐 run L0 请求触发采集，不借 `enable_chip_swimlane` 或 PMU 开关隐式触发。沿用 `CallConfig`/`DfxRunConfig` 的配置路径时须同步检查 Python、native 和远程传输的 wire 布局，避免只改一端造成 ABI 错位。非 A5 onboard 明确拒绝该请求。宏 `0` 编译时不引用 L0 源码、结构字段和 driver 接口。
2. **一个小型 L0 session**：放在 A5 onboard host 侧，复用现有 DFX 的 run 身份与输出目录，但暂不继承 `ProfilerBase`、不开管理/采集线程。host 只启动一个对应受监控核的 channel、分配一块固定大小 GM；借用已验证的专用 AICPU 控制入口，在第一个 AICore 任务可运行前同步写入并回读 `base/len/wptr/gen`。这一步必须在普通 run 的提交边界内完成，不依赖后启动的主调度 AICPU。
3. **完成后导出**：普通 run fence 确认完成后，由 AICPU 先关闭门控、清空 `base/len` 并回读确认；读取硬件写指针，只复制有效字节前缀，保存 `.bin` 和包含 run/core/channel、写指针及清理状态的元数据，再用现有连续缓冲解码器写 `.json`。原始字节始终保留，即使解码或任务归属失败。采集错误单独报告；撤销硬件地址无法确认时不释放被引用的 GM，也不为此 reset 设备。
4. **边界和验收**：第一个版本只支持一个 L0 run、一个受监控核、一个不回绕的缓冲区；第二次启用、并发 L0 run、指针越界/回绕均明确拒绝或标为不完整，不静默输出。用一个普通 ST 验证任务 pass、受监控核确实运行、raw 非空、解出预期 region/pipe、GM 清理回读正确；再用两个顺序任务验证两组标记和任务记录的关联。宏 `0` 的普通 ST 构建与运行路径也要检查。上板前必须隔离当前 runner 的自动 force-reset 路径，所有运行经设备锁、限时且不调用 `npu-smi info`。

后续再扩为多核与环形缓冲、按 task 增量 drain、精确时钟对齐、STATE 活动解码、现有 viewer 呈现及跨 run 复用。特别是这台驱动上重复 `prof_drv_start` 曾失败，首版不能假定 `prof_stop=0` 意味着 channel 可再次打开。

## 大步骤

### 1. 确定 DFX 边界和单一编译宏

**参考现有 DFX**：沿用 `DfxRunConfig` 的逐 run 配置、`DeviceRunnerBase` 的 admission/close、run epoch、`output_prefix` 和错误回滚约定；参考 A5 `PmuCollector` 的 host/device 分工、设备拷贝及原始数据保存，参考 chip swimlane 的任务时间记录和输出关联。`ProfilerBase` 的 ready-queue/free-queue 协议服务于 AICPU 发布的缓冲区；L0 数据由硬件直接写入自管 GM，先实现独立 collector，不为了复用基类而改变硬件数据通路。L0 的 channel、GM、MMIO 和字节解码归 L0 模块所有，普通 ST 与现有 collector 只通过 run 生命周期和可选的任务记录交互。

**单一编译宏**：用 `SIMPLER_A5_ONBOARD_L0`，默认 `0`，仅 A5 onboard 的 host、AICPU、AICore 构建可设为 `1`，并要求 `SIMPLER_DFX=1`。由同一构建配置生成三端一致的宏值；所有新增 L0 硬件源文件、设备入口、调用点及输出接线都受这个宏控制。`0` 时无 L0 符号引用、驱动调用、GM 分配、寄存器访问或产物；逐 run `CallConfig.enable_l0` 字段在跨进程 wire 中始终保留，以免 Python 扩展和运行时采用不同宏值时产生布局错位。`1` 但该 run 未请求 L0 时也不操作硬件。已有的独立实验工具和归档原始样本不进入普通 ST 构建。

**运行期解耦**：L0 的逐 run 请求与 `enable_chip_swimlane`、`enable_pmu` 分开，不能因打开 chip swimlane 就隐式打开 L0。L0 可独立保存 raw；任务泳道关联在 chip swimlane 任务记录可用时添加，缺失时保留未归属的 L0 事件。具体公开配置入口在接入普通 ST 时按现有 DFX 配置方式确定，不额外引入环境变量或第二个编译宏。L0 与现有 PMU 可能共用 perfmon 控制资源；先核对寄存器/driver 互斥关系，未证实可共存前在 admission 阶段拒绝同时请求。

**构建验收**：验证宏 `0` 的 A5 onboard 和非 A5 构建不依赖 L0 接口，宏 `1` 的 host/AICPU/AICore 值一致；`SIMPLER_DFX=0` 与 L0=`1` 在配置阶段明确报错。检查普通 ST 在 L0 未请求时不发生 channel/MMIO/GM 操作。

### 2. 在普通 ST 的诊断生命周期中接入 L0 collector

**复用部分**：L0 collector 接入现有 DFX admission、run 开始/结束、回滚及输出目录，不重做 ST 执行或另建 run 框架。现有 `ChipSwimlaneCollector` 的四类 ready-buffer 是任务和调度记录，原始 BIU 字节不塞进现有任务记录类型。

**L0 特有部分**：定义每核 GM 缓冲区、物理核到寄存器地址的映射、channel 会话与采集状态。host 开 channel，AICPU 前置配置 `base/len/wptr/gen`，AICore 启动后由 HWTS 控制 `en`。正常完成后先停本地门控并清除 `base/len`，确认回读，再复制 GM、停 channel、释放缓冲区。遇到无法确认撤销的情况，保留仍被寄存器指向的 GM，报告错误；不为采集失败重启或 reset 设备。

**EVB 安全门槛**：当前 `DeviceRunner` 的某些致命错误恢复分支会调用设备 force reset。任何改造后的 ST 上板前，必须确认 L0 诊断错误不会进入该分支，并为这台 EVB 的测试运行禁用或隔离自动 reset；否则只做离线编译和解码验证。

**接入点**：当前 `DeviceRunner::launch_execution_impl` 的 AICore 提交之前、`drain_execution` 的 run fence 完成之后；现有 `KernelArgs`/AICPU SO 加一条专用配置入口或等效控制路径。复用 `host_regs.cpp` 已建立的物理核寄存器表，避免另建不同的映射规则。

**通过条件**：一条普通 onboard ST 在打开 L0 时仍 pass；指定核 `en/gen/base/len/wptr` 的前、中、后状态符合单核实验；关闭 L0 时普通 ST 行为不受影响。先只用一个已验证的受监控核，不同时开启所有 channel。处理这台驱动上 channel 可能无法二次启动的问题：优先按一次会话复用，不把“`prof_stop=0`”当作可重新打开的证明。

### 3. 把字节归属到真实 ST 任务

**2026-10-26 进展**：已在 channel 17 的普通 MIX ST 中同时采到 108 核 L0 数据与 108 条 chip swimlane AICore task 记录，并将离线关联器接入 `SceneTestCase` 收尾流程。360 条事件均能以“同 run、同逻辑核唯一 task、block 一致”关联到真实 `task_token=0`；原始 cycle 无法直接按统一偏移落入 task 时间窗，因此多 task 边界关联仍待时钟标定或新增边界标记。详见 [demo 与限制](l0_task_join_demo_2026_10.md)。

**要做**：先复用 chip swimlane 的 `ChipSwimlaneAicoreTaskRecord`，其中已有 `task_token`、每核 `reg_task_id`、`start_time/end_time`；确认从逻辑核到 perfmon 物理核的映射与两个 cycle 时钟的关系。host 解码 `START_STAMP`、pipe/region、`STATE`，按同核 cycle 窗口匹配任务；无法唯一匹配的记录标为 unmatched，不强行归属。顺序任务的写指针增量可用作校验，不能在并发调度下单独充当任务 ID。只有现有记录不足时才补充新的 AICPU FIN 元数据。

**接入点**：现有 chip swimlane collector 的 AICore task 记录与 run 归属；必要时再接 AICPU FIN 路径。旧版 `L0TaskFinMarker`/`l0_perf_collector_aicpu.cpp` 可供参考。当前分支已有 [连续缓冲区解码器](../../../src/a5/platform/include/common/l0_trace_decode.h)，需补环形流和边界校验。

**通过条件**：至少一个有两个不同 runtime task ID 的普通 ST，输出记录含真实 task ID、physical core、block、cycle、pipe、region；两个任务的标记都归到正确任务；raw、decoded 和 ST 完成结果相互一致。

### 4. 扩展覆盖到 36-cluster 机器

**要做**：按物理 AIC/AIV 子核分别管理缓冲区及读指针，从一个已验证核扩到同簇，再扩到多个受监控簇，最后验证 36-block MIX / 72-AIV 普通 ST。每一步都记录“实际运行核、已配置核、有数据核、无数据核”，分辨调度未命中、硬件未启用和采集丢失。旧版在另一环境观察到一次 `prof_drv_start` 可促使更广覆盖；当前机器必须实测再决定是否需要启动更多 channel。

**通过条件**：每个实际运行且声明覆盖的核都有可解析的 START_STAMP 和预期 region；未覆盖核有明确原因。检查每核缓冲区容量、写指针回绕、`samp_wrt` 增量及溢出，避免把旧数据当成本次运行数据。

### 5. 接入诊断输出与查看器

**要做**：保留原始字节用于复核，输出稳定 schema 的 L0 事件文件，再把任务内 pipe 标记转成泳道展示。记录采集窗口、设备/核/任务标识、字节范围、解码错误和丢样统计。明确 `mark_stamp` 的时间语义：它来自 scalar 派发处，MTE 标记可能自带 barrier，不能直接解释为每条 pipe 的实际占用率。

**通过条件**：普通 ST 的输出目录里同时有成功结果、原始采集文件和可打开的 L0 泳道；从原始字节重新解码可复现事件数量、顺序和任务归属。

### 6. 稳定性与性能验收

**要做**：覆盖多次 run、不同调度核、采集开关切换、错误中途退出、缓冲区接近满、FIN 后在途写入以及 channel 状态残留。把诊断采集失败和普通任务执行失败分开报告；记录内存占用与额外耗时。前五步证据足够后，再决定是否扩大采集覆盖范围。

**通过条件**：限时 onboard ST 连续运行，结果和资源状态可复核；没有悬挂 GM 地址、错误归属、静默溢出或因 L0 诊断导致的普通 ST 失败。全程无需设备或服务器 reset。

## 推荐推进顺序

先按“首个可交付版本”完成一次普通 ST 的单核同步采集及 raw/JSON 输出，再验证两任务关联。下面的步骤 **4 → 6** 是后续扩展，不能阻塞首版交付；每一步都可以独立审查结果。
