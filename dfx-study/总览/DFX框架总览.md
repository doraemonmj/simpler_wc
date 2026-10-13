# simpler DFX 框架总览

本文按设计目的、数据采集、落盘后的处理与分析、外部查看插件、DFX 议题树五部分组织，关注各环节为什么存在、怎样衔接。

## 1. 思考与设计目的

DFX的目标是什么？

- 让一次运行可诊断：当结果错误、执行卡住、耗时变长或资源紧张时，开发者能把现象定位到具体阶段、任务和执行层，并找到足以指导修复的原因。判断DFX是否达到目标：在哪里发生了什么、为什么发生

DFX的设计逻辑是什么？

- 设计思想：围绕“慢在哪一段、任务为什么等待、资源或参数是否异常”组织互补证据，形成从整体阶段到任务、必要时到核内细节的观察层次。高开销采集按需启用，控制对运行的扰动；记录要能归属到同一次运行，并能核对完整性，后续的关联和归因才有可信依据。
- 技术实现：在事实产生的位置分层记录，通过公共机制或模块自身的路径交给 Host，再在落盘后关联、分析与展示；采集与分析解耦，避免运行时承担复杂的归因工作。

DFX怎么分模块，怎么分组件的，处理逻辑是什么？

- 按诊断问题分模块：STRACE / Device phases 看阶段耗时，Chip Swimlane / dep_gen 看任务执行与依赖，PMU / Args Dump / Scope Stats 看硬件计数、参数与资源；Core Swimlane 在落盘后针对单个任务做核内重放。
- 按处理职责分组件：运行配置控制开关，Host / AICPU / AICore 在各自掌握事实的位置记录；日志、固定时间槽或公共 Profiling Framework 把记录交给 Host，模块 Collector 整理落盘，仓库工具与查看器再完成关联分析和展示。
- 业务上是“选择问题 → 采集证据 → 解释结论”，架构上是“配置 → 记录 → 交接 → 落盘 → 分析”。

## 2. 数据采集

### 2.1 控制链与一次运行

编译期决定诊断代码是否存在，进程期设定全局策略，单次运行配置决定真正启用的能力与输出位置。因此，“功能已编译”不等于“每轮都采集”。

```text
单次运行配置
  → 校验并按需分配缓冲
  → 将采集状态传给 Host / AICPU / AICore
  → 运行时记录
  → 停止生产、排空尾部数据
  → 对账并导出本轮产物
```

每轮必须有明确归属。后台整理文件可以与下一轮设备运行重叠，但最终仍需确认本轮记录已消费、发布完成；否则尾部数据可能遗漏。

### 2.2 公共 Profiling Framework

公共框架负责持续记录的交接，子模块只决定在哪里打点、记录什么、怎样解释记录。核心缓冲流程是：

```text
free queue → Device 写当前缓冲 → ready queue
                                  ↓
Host 回收 ← Collector 解析/保存 ← Host drain
    └────────→ free queue
```

一块缓冲在任一时刻只属于一方。Device producer 负责满缓冲发布和换缓冲；Host 管理队列、地址映射或复制，并由具体 Collector 整理原始产物。停止时先排空，随后让 Collector 处理最后一批数据，再核对“产生、采集、丢弃”的计数。

Host 消费不足时，ready 队列可能满、空闲缓冲可能耗尽。背压让生产者等待 Host 排空和补充；长时间无进展才保护性丢弃，并记录数量。a2a3 主要通过共享映射读取缓冲，a5 使用 Host shadow 和显式复制；上层记录语义尽量保持一致。

### 2.3 DFX 子模块怎样产生证据

各类记录的产生方式不同；持续产生、需要缓冲交接的记录由公共 Profiling Framework 送到 Host：

```text
旁路采集：STRACE → 日志；Device phases → 固定时间槽
公共采集：Chip Swimlane（AICore / AICPU）/ PMU / Args Dump / Scope Stats
          → Device 缓冲 → Host Collector → Host 整理与导出（见 2.4）
按 runtime 分流：dep_gen → 设备提交记录经公共 Collector 后重放，或 Host 构图时直接记录
```

| 子模块 | 记录发生在哪里、怎样交给 Host | 采集边界与原始产物 |
| ------ | ----------------------------- | ------------------ |
| STRACE / Device phases | Host 作用域退出时写 `[STRACE]` 日志；少量设备阶段写固定槽，运行后读回 | 不走公共流式 Collector；适合阶段耗时，不提供完整逐任务时间线 |
| Chip Swimlane | AICore 写每核执行时间，AICPU 写任务 dispatch/finish 和调度阶段；两类缓冲都经公共 Profiling Framework 交给 Host Collector | AICore 缓冲轮转有专门机制，但交给 Host 的通道仍是公共框架；Collector 将各类原始记录分别写入 `chip_swimlane_records.json`，此时尚未按任务关联 |
| dep_gen | 设备侧构图时，AICPU 在任务提交处写输入记录，经公共 Collector 拉到 Host；Host 构图时直接记录依赖 | 设备侧记录先核对完整性、再在 Host 重放生成 `deps.json`；Host 直录不经过设备 Collector，设备侧采集可能扰动提交时序 |
| PMU | AICore 在 kernel 窗口计数，AICPU 在任务完成时归属并提交记录，再经公共缓冲/Collector 导出 | 每任务一组硬件计数，形成 `pmu.csv`；不同架构的事件含义需分别解释 |
| Args Dump | AICPU 在 dispatch 前和 FIN 后采集参数，记录经公共缓冲/Collector 到 Host | 元数据与可选 payload 分离，供参数检查或单任务重放 |
| Scope Stats | AICPU 在 scope 进入和退出时写资源 frontier，经公共缓冲/Collector 导出 | 形成 scope 资源记录；主要适用于相应的 ring/scope 资源模型 |

公共框架负责缓冲所有权、背压和最终排空；各模块负责自己的采样时机与记录解释。依赖边由 dep_gen 在依赖语义仍完整的位置取得。Swimlane 原始记录及依赖图落盘后，再由离线工具关联。

### 2.4 Host 收集后怎样落盘

缓冲到达 Host 并不等于立即得到最终文件。公共框架负责交接和回收缓冲；各子模块的 Collector 负责按记录类型保存内容。运行结束时先排空尾部，再核对采集与丢弃计数、完成导出。部分模块在运行中已开始写盘，因此这些步骤不一定严格串行。

```text
Device 缓冲 → Host Collector 接收、分类 → 内存记录或运行中写盘
                                     → 停止生产、排空、对账 → 最终产物
```

| 子模块 | Host 到最终产物之间的处理 |
| ------ | ------------------------- |
| Chip Swimlane | 按运行归属和记录类型收集，结束时合并采集分片、核对计数，在同一 `chip_swimlane_records.json` 中分别写入 AICore 与 AICPU 原始记录及元数据；任务级关联留到落盘后 |
| dep_gen | 设备提交记录先留在 Host 内存；确认采集完整后，在 Host 重放提交和依赖计算、校验结果，再写 `deps.json`。Host 构图路径则直接形成依赖数据 |
| PMU | Collector 在运行中把计数记录写入 CSV 分片，结束时合并分片并核对计数，形成 `pmu.csv` |
| Args Dump | Collector 依据元数据取回参数 payload，运行中异步写二进制内容；结束时汇总、排序元数据并导出索引文件 |
| Scope Stats | Collector 暂存资源记录，结束时核对计数，连同运行元数据导出 `scope_stats.jsonl` |

STRACE 日志和 Device phases 固定槽不经过这条公共 Collector 流程。这里的“整理”仍属于原始产物的形成；跨记录关联、时间转换和指标分析放在第 3 节。

## 3. 数据落盘之后的处理与分析

### 3.1 从原始产物到视图

落盘文件是证据，仍需确认运行归属、完整性、任务标识和时钟域，再按问题处理：

```text
日志 ───────────────→ 阶段树 / 耗时表
AICore / AICPU 原始记录 → 按任务关联 → 时间线
时间线 + 依赖图 ────────→ 调度开销 / 关键路径
依赖图 ─────────────→ 任务关系图
资源记录 / PMU ──────→ 水位报告 / 计数表
参数 Dump ───────────→ 参数检查 / 核内重放

仓库工具负责计算、转换或生成 HTML；外部查看器/插件负责其支持的交互视图或进一步分析。
```

时间线回答“何时发生”，依赖图回答“受谁约束”，统计表回答“代价多大”。一个结论可能需要同时看几种证据。

### 3.2 各子系统落盘后怎样处理

下面只看仓库内的处理链。箭头表示脚本读取产物并生成结果，不表示所有脚本必须依次运行。

**STRACE / Device phases：** Host 日志中的 `[STRACE]` 标记 → `strace_timing.py` 按进程、运行轮次和嵌套深度还原调用树 → 汇总阶段耗时、每轮表，或导出 Host 时间线。Host 构图的 bind 阶段还能交给 `hbg_bind_phases.py` 区分冷/热运行并统计各阶段耗时分布；`phase_time_split.py` 利用日志里的线程 CPU 时间，把阶段墙钟时间拆成运行和等待。

**Chip Swimlane：** `chip_swimlane_records.json` → `swimlane_converter.py` 的读取层关联 AICore 与 AICPU 任务记录，并把设备周期换算成时间 → 转换器按核、任务和调度阶段生成 Perfetto trace；可用 `name_map*.json` 标注内核名称，有 `deps.json` 时再添加依赖箭头。另有两条分析分支：

```text
任务时间记录 + deps.json → sched_overhead_analysis.py → 就绪/下发/执行区间与调度开销报告
AICore 原始记录 + deps.json + name_map*.json → critical_path.py → 静态关键路径与实际执行归因路径
```

`sched_overhead_analysis.py` 借用转换器的读取层，以前驱结束时间估算任务就绪点，再看空闲核上是否有就绪却未下发的任务。`critical_path.py` 直接读原始 AICore 记录，分别追踪依赖约束下的最长路径和实际执行中的等待。依赖采集会扰动提交时序，做调度开销分析时通常使用同一计算图的另一次运行所得 `deps.json`。

**dep_gen：** `deps.json` → `deps_viewer.py` 整理任务、前后驱和边的来源，输出可检索文本或 Graphviz 生成的独立 HTML；需要时按传递关系折叠可省的显示边。`wait_reduction_sim.py` 则从同一文件抽出 WAIT 边，按提交顺序比较完整依赖图约简与有限窗口位图算法的删边覆盖率；输出是模拟报告，不是设备实测收益。

**PMU：** 当前仓库没有承接 `pmu.csv` 的专用离线分析脚本。仓库内流程止于 Collector 按任务、核和事件列导出 CSV；后续解释计数要结合对应架构的事件定义。

**Args Dump：** `args_dump.json` 记录参数位置和类型，`args.bin` 保存已采集的 payload。`dump_viewer.py` 按任务、采集阶段、参数角色或序号筛选，再按偏移读取二进制内容并导出文本。`core_swimlane.py` 结合测试文件中的 kernel 元信息与 Dump 参数生成独立重放工程，编译后调用模拟器采集单任务核内 trace；这是隔离重放结果。

**Scope Stats：** `scope_stats.jsonl` → `scope_stats_plot.py` 按 ring 和 scope 位置配对 begin/end 记录，计算资源增量、积压和相对容量的峰值 → 生成自包含 HTML 报告。报告中的高水位用于定位资源压力，不等同于逐时刻的实际占用曲线。

### 3.3 其他查看方式

`swimlane_converter.py` 生成的 trace 可交给 Perfetto 等外部查看器；仓库工具生成的独立 HTML 可直接用浏览器打开。VS Code 插件的交互视图另见第 4 节。

Host 与 Device 时间戳可能来自不同 clock，不能直接相减。若记录缺失、时钟未对齐或采集开销明显，图中的空洞也要结合这些边界解释。

## 4. 外部查看插件：pypto-tools_pypto3

目标仓库是 [majin0824/pypto-tools_pypto3](https://gitcode.com/majin0824/pypto-tools_pypto3)。它是 VS Code 中的 PyPTO Toolkit：读取已有运行结果，把控制流、计算图、泳道图、性能与精度信息组织成可交互视图，不承担 simpler 的设备采集。下面按已拉取的 `master` 提交 `e409fb9`（插件版本 `2.0.19`）概括当前实现。

### 4.1 仓库框架

下图按职责展示主要目录；`pypto_toolkit` 中的 TypeScript 在 VS Code 扩展端运行，Vue 页面在 Webview 中运行。

```text
pypto-tools_pypto3/
├── vscode_plugins/pypto_toolkit/           VS Code 扩展工程
│   ├── package.json                         扩展声明与构建脚本
│   ├── src/                                 扩展端：入口、数据服务与处理
│   │   ├── extension.ts                     激活插件，装配服务、命令和视图
│   │   ├── command/                         文件右键等打开命令
│   │   ├── services/run-data/               扫描并索引运行结果
│   │   ├── view/                            Webview 路由与服务
│   │   ├── handler/                         泳道等视图的数据处理
│   │   ├── converter/                       泳道格式转换
│   │   ├── process.ts                       整理时间线与统计数据
│   │   └── core/                            通信协议、存储与缓存
│   └── media/vue-project/src/              Webview 前端
│       ├── apis/                            请求扩展端数据
│       └── components/                      控制流、计算图、泳道图、三栏联动等页面
└── docs/zh/                                 插件使用文档
```

其中控制流和计算图主要解释“程序如何表达与编译”，泳道图解释“运行时怎样执行”；三栏联动通过已有语义标识把这几种视角接起来。

### 4.2 插件从启动到打开视图

```text
VS Code 激活插件 → 注册命令、数据服务、视图路由和 Webview 通信
                    ├─ 运行结果入口：扫描 PyPTO 运行目录 → 侧边栏选择结果
                    └─ 单文件入口：从文件右键打开 → 根据文件内容选择视图
  → 扩展端读取、转换和缓存数据 → Webview 请求并接收数据 → Vue 绘图与交互
```

运行结果入口默认扫描 `~/.pypto/run`，设置了 `PYPTO_HOME` 时改为其下的 `run` 目录。每次运行的 `rundata.json` 指向源码、控制流、计算图和泳道图等产物；插件把有效记录存入本地索引，侧边栏按运行时间列出，打开某一结果时再组织具体文件。三栏联动依赖同一运行结果里的代码、图与泳道路径。单文件入口则直接打开一个已有 JSON，无须先有 `rundata.json`，但跨视图跳转需要额外的关联信息。

各视图共用“VS Code 扩展端处理文件、Vue Webview 负责展示”的方式：控制流从 `program.json` 等结果还原执行结构；计算图读取图 JSON，提供节点检查、搜索及编译图对比；泳道图按时间组织执行记录；OpCheck 和 TaskCheck 读取精度对比结果。插件在这些已有产物上提供查看与交互，不会启动 simpler 的采集流程。

### 4.3 泳道图在插件内怎样处理

```text
主文件 → 识别格式 → 转成或读取 trace events
                  → 按进程/线程/任务整理，建立依赖与统计信息
                  → 缓存并分批发送 → 时间轴、泳道、详情和性能面板
```

插件当前有两条输入路径：`merged_swimlane.json` 等已有 `traceEvents` 的文件直接进入整理流程；带顶层 `tasks` 的 `perf_swimlane.json` 格式先由插件转换成 trace events。后一条路径会尝试读取同目录的 `deps.json` 和 `name_map*.json`，补充依赖边与函数名。插件还按自己的产物约定查找 `tilefwk_prof_pmu.csv`、`dyn_topo.txt` 等辅助文件。

扩展端把事件整理成进程、线程和任务层级，计算泳道布局、依赖连接及可用的 PMU 信息。处理结果按源文件及辅助文件的修改时间复用缓存；大 trace 有压缩处理路径，普通路径也会分批把任务数据送往 Webview。Vue 页面再提供缩放、搜索、节点详情、依赖查看、时间范围统计和与计算图联动等操作。

### 4.4 当前与 simpler 产物的衔接

| simpler 产物 | 当前关系 |
| ------------ | -------- |
| `chip_swimlane_records.json` | 是分流保存的原始记录，不符合插件当前直接识别的顶层 `tasks` 或 `traceEvents` 输入结构 |
| `merged_swimlane.json` | 可由 simpler 的 `swimlane_converter.py` 生成；插件按 trace events 读取并绘制泳道 |
| `deps.json`、`name_map*.json` | simpler 转换原始记录时可用来补依赖和名称；插件只在处理自己的 `perf_swimlane.json` 路径时另行查找同目录文件 |
| `pmu.csv`、`CPM_*.json` | 当前源码没有对应的直接读取路径；插件查找的 PMU 文件名是 `tilefwk_prof_pmu.csv` |

因此，查看 simpler 泳道的可行链路是“原始记录 → simpler 转换器 → `merged_swimlane.json` → 插件单文件视图”。simpler 的运行目录也不会仅凭 `chip_swimlane_records.json` 自动出现在插件侧边栏；侧边栏采用 PyPTO 的 `rundata.json` 索引。第 3 节的脚本负责原始记录到分析结果或标准 trace 的处理，插件负责交互查看。

以上是对当前源码路径的核对，尚未用一组 simpler 产物在 VS Code 中做端到端打开验证。

### 4.5 怎样打包成 VS Code 扩展

扩展工程在 `vscode_plugins/pypto_toolkit/`，其 `package.json` 声明入口为 `dist/extension.js`。打包分成构建和封装两步：TypeScript 编译扩展端到 `dist/`，Vite 编译 Vue 页面到 `webview-dist/`；随后 `@vscode/vsce` 按 `.vscodeignore` 筛选文件，生成可安装的 `.vsix`。源码、测试、前端工程源码不会作为运行时文件装入包，编译产物和所需资源会装入。

```text
完整仓库 + 两处 npm 依赖
  → vsce package
  → 自动执行 vscode:prepublish
      → 复制根目录 LICENSE / README → 检查非 DEBUG 配置 → git lfs pull
      → npm run compile：清理旧 dist，构建 webview-dist 与 dist
      → postcompile：复制 sql.js 的 WASM 运行文件
  → 依据 package.json 与 .vscodeignore 封装 pypto-toolkit-<版本>.vsix
```

在完整检出的仓库中，先为扩展工程和 `media/vue-project/` 分别安装 npm 依赖，再从扩展工程目录执行 `vsce package`（或 `npx @vscode/vsce package`）。`vsce` 不在该仓库的依赖列表中，需要另行提供；[VS Code 官方打包说明](https://code.visualstudio.com/api/working-with-extensions/publishing-extension#packaging-extensions)也说明它会在打包时执行 `vscode:prepublish`。该预处理脚本依赖仓库根目录的许可与 README、Git LFS 文件以及 Node.js 环境；`package.json` 要求 Node.js `>=22.13`。生成 `.vsix` 是本地扩展包，发布到 Marketplace 是后续独立步骤。

## 5. DFX 议题树

- **Top-down：从诊断问题出发**
  - DFX 的目标与设计思路
  - 时间花在哪里
    - 阶段耗时：STRACE 与 Device phases
    - [Chip Swimlane：任务执行时间线](../专题/芯片泳道任务时间线.md)
    - 任务依赖与调度归因：dep_gen、调度开销、关键路径
  - 执行状态怎么样
    - 任务硬件计数：PMU
    - 资源水位：Scope Stats
  - 单任务发生了什么
    - 参数采集与检查：Args Dump
    - 核内隔离重放：Core Swimlane
- **Bottom-up：从实现链路出发**
  - 一次运行的 DFX 控制与产物归属
  - Host、AICPU、AICore 的分层记录
  - 公共 Profiling Framework 与其他数据交接路径
  - Host Collector 落盘与完整性核对
  - 仓库工具的离线处理与分析
  - 外部查看器与插件的交互展示
