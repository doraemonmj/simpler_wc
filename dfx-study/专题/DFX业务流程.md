# DFX 业务流程：一次运行从配置到图

用三张图串讲约 8 分钟：先看数据怎么流转，再认清模块分工，最后走一遍慢运行的分析路径。

## 图一：一次运行的完整链路

~~~mermaid
flowchart TB
    CFG["本轮配置<br/>能力与输出位置"] --> RUN["一次运行"]
    RUN --> STAGE["STRACE / Device phases<br/>阶段记录"]
    RUN --> DATA["Chip Swimlane / PMU<br/>Args Dump / Scope Stats"]
    RUN --> DEP["dep_gen<br/>依赖记录"]
    DATA --> PIPE["公共采集通道"] --> COL["Host Collector"]
    DEP -->|"设备构图"| PIPE
    DEP -->|"Host 构图"| DIRECT["Host 直接记录"]
    STAGE --> FINALIZE["收尾、对账、导出"]
    COL --> FINALIZE
    DIRECT --> FINALIZE
    FINALIZE --> RAW["本轮原始产物"]
    RAW --> TOOL["仓库工具<br/>关联、转换、统计"]
    TOOL --> VIEW["时间线 / 依赖图 / 报告"]
    VIEW --> SHOW["浏览器 / Perfetto / 插件"]
~~~

讲图时沿箭头走一遍：配置决定本轮采什么；运行时各模块记录自己掌握的事实；结束时收齐并核对；仓库工具把原始记录变成可读结果。阶段记录和依赖记录有自己的路径，查看器消费结果。

## 图二：模块各自回答什么

~~~mermaid
flowchart LR
    Q["想看什么？"] --> T["时间"]
    Q --> D["关系"]
    Q --> S["状态"]
    Q --> A["内容"]
    T --> T1["阶段：STRACE / Device phases"]
    T --> T2["任务：Chip Swimlane"]
    D --> D1["任务依赖：dep_gen"]
    S --> S1["硬件计数：PMU"]
    S --> S2["资源水位：Scope Stats"]
    A --> A1["任务参数：Args Dump"]
    A1 -.-> A2["单任务重放：Core Swimlane"]
~~~

时间图先告诉我们“慢在哪”；依赖、计数和资源记录再补充“受什么约束”。Core Swimlane 属于落盘后的单任务分析。

## 图三：把链路用在慢运行上

~~~mermaid
flowchart TD
    SLOW["同一场景运行变慢"] --> PHASE["看阶段耗时"]
    PHASE --> DEVICE{"设备执行段变长？"}
    DEVICE -->|"否"| OTHER["继续查其他阶段"]
    DEVICE -->|"是"| LANE["看任务时间线"]
    LANE --> SHAPE{"主要变化"}
    SHAPE -->|"任务条变长"| PMU["结合 PMU 看任务指标"]
    SHAPE -->|"任务间空档"| DEP["结合依赖与资源记录"]
    PMU --> CHECK["核对本轮记录与对照运行"]
    DEP --> CHECK
~~~

图上的任务空档只是现象。判断等待原因前，先确认记录完整、属于同一轮；做耗时对比时还要考虑采集本身的影响。

## 8 分钟讲述节奏

- **0:00–0:45：** 提出主问题：一次运行的 DFX 结果从哪里来？
- **0:45–3:15：** 沿图一从配置走到查看，重点指清三条记录路径和运行结束时的对账。
- **3:15–5:15：** 用图二介绍模块；按“时间、关系、状态、内容”四组讲各自回答的问题。
- **5:15–7:30：** 走图三的慢运行例子，展示一张图如何引出下一类证据。
- **7:30–8:00：** 回到图一收束：本轮事实经过收集、整理才成为可解释的视图。

实现链路详见 [DFX 框架总览](../总览/DFX框架总览.md)；任务时间线的专题见 [芯片泳道任务时间线](芯片泳道任务时间线.md)。
