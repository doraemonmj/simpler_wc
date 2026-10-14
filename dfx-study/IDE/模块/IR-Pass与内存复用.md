# IR Pass Trace 与 Memory Map

## IR Pass Trace

右键 `passes_dump*` 目录打开 IR Trace。`dist/services/ir-trace/discovery.js` 寻找 `00_frontend.py`，再读取 `01_<Pass>.py`、`02_<Pass>.py` 等连续编号快照；缺号或重复编号会报错。同名 `.log` 文件可带 warning。`dist/view/servers/ir-trace-server.js` 把快照交给前端，比对相邻 Pass 的行和字符变化。

这里的输入是 **Python IR 文本文件序列**，不是 `program.json` 或计算图 JSON。目录结构和命名见[数据结构](../数据结构.md#ir-pass-快照)。用户文档说明视图是只读预览，不会回写 Pass 文件。

## Memory Map

右键匹配 `*after_AllocateMemoryAddr.py` 的快照，或从 Chip Swimlane 的关联入口打开 Memory Map。`dist/view/servers/memory-map-server.js` 读取 Python IR 文本，`dist/converter/memory_map_parser.js` 识别分配后的 MemRef，把每个函数的 Tile 地址区间和生命周期组织为矩形。横轴是内存地址，纵轴是语句生命周期；同一地址在不重叠生命周期重复使用时可直观看到复用。

解析器针对 PyPTO Pass dump 的代码生成格式，而不是任意 Python 源码。安装包会从同级 `ptoas` 目录的 `.pto` 文件尝试识别 `pto.target_arch` 以选择空间容量；找不到时使用默认架构。容量表在该安装包的 `dist/converter/memory_map.js` 中，是界面分析参数，不应把它当成芯片实测容量。具体文本契约需要对照实际 Pass dump 和解析器再核实。
