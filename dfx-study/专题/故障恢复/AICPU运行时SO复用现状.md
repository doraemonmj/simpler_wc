# AICPU 运行时 SO 复用现状

本文梳理 main 的 **AICPU runtime SO**（设备执行器 `simpler_inner_*.so`）从 Host 到 Device 的加载流程。Host 上的源文件与 Device 上写出的文件内容相同，但它们是两份独立的文件；Host 不执行这份 AICPU SO。

## main：设备 SO 的加载与收尾

| 顺序 | 位置 | 实际动作 |
| ---- | ---- | -------- |
| 1. 读取 | Host | 每个 `ChipWorker` 初始化时，从本地分别读取 dispatcher SO 和 AICPU runtime SO 到内存；`DeviceRunner` 暂存这两份字节。Host 上的源文件不删除。 |
| 2. 判断是否上传 | Host | 计算 runtime SO 的 fingerprint，查本进程的 `(fingerprint, device_id)` 记录。命中则跳过本次 dispatcher 和 runtime SO 的 Host→Device 传输；不同 device ID 各上传一次。 |
| 3. 首次传输 | Host→Device | 未命中时，把 dispatcher 和 runtime SO 的字节分别拷到 Device 内存，启动一次 dispatcher。用于传输的 Device 内存缓冲在这次启动完成后释放。 |
| 4. 写入文件 | Device | dispatcher 把收到的 runtime SO 字节写入临时文件，再 `rename` 为 `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<device_id>.so`。dispatcher SO 本身不落设备文件。 |
| 5. 注册加载 | Host 发起，CANN 在 Device 加载 | **每个 `DeviceRunner` 都会**生成指向该设备文件的 Host 侧 JSON，调用 `rtsBinaryLoadFromFile`，再用 `rtsFuncGetByName` 取得函数句柄。即使第 2 步跳过传输，新 runner 也会执行这一步。 |
| 6. 执行任务 | Host 发起，Device 执行 | 同一 runner 后续任务通过已取得的函数句柄调用 `rtsLaunchCpuKernel`，无需再次传 SO、写设备文件或注册加载。初始化完成后，runner 释放 Host 内存中的两份 SO 字节缓冲。 |
| 7. 关闭 runner | Host 发起 | 调用 `rtsBinaryUnload` 卸载该 runner 的 CANN 句柄，并删除 Host 侧 JSON；**不删除设备上的 runtime SO 文件**。 |

**再次初始化时：**同一 Host 进程顺序创建的新 runner，若设备和 fingerprint 相同，会跳过第 3～4 步，但仍重新注册加载；并发初始化的两个 runner 可能都在记录写入前完成检查，因而各自上传。新 Host 进程没有这条内存记录，会重新传输并写设备文件。main 的 Device 写入路径不检查同名文件是否已存在，即使目标文件已有相同内容，也会先写临时副本再覆盖。因而这里的进程内复用是 **Host 跳过上传**，不是 Device 根据已有文件自行复用；设备文件也不会随进程结束而回收。[Host 读取](../../../src/common/worker/chip_worker.cpp) · [初始化与释放缓冲](../../../src/common/platform/onboard/host/device_runner_base.cpp) · [上传与加载](../../../src/common/aicpu_loader/host/load_aicpu_op.cpp) · [设备写入](../../../src/common/aicpu_loader/device/aicpu_dispatcher.cpp)。

同一 Host 进程内，runtime SO 与 orch SO 的复用情况如下：

| 场景 | 实际复用情况 |
| ---- | ------------ |
| 同一个 callable 多次运行 | 复用已加载的函数句柄，不再加载 SO。Host 编排的 orch SO、Device 编排的 orch SO 都是这样；当前 runner 的 runtime SO 也复用 CANN 函数句柄。 |
| 新建 `DeviceRunner`，runtime SO 内容和 device ID 相同 | Host 命中上传记录，跳过再次传输和写文件；新 runner 使用已有的 Device runtime SO 文件，但仍向 CANN 重新加载，取得自己的句柄。 |
| 注册另一个内容相同的 orch callable | 不复用已加载的 orch SO 句柄，仍按新的 callable 注册。 |

## 设备空间与 PR 对照

| 项目 | main | PR #2651 |
| ---- | ---- | -------- |
| Host→Device 传输 | 同进程命中时跳过；新进程重传 | 每个新 runner 都传完整 SO |
| 设备文件 | 同名也重写；不同 fingerprint 持续累积 | 同一设备文件系统中，v2 文件内容相同时复用，跳过 tmpfs 重写 |
| 删除 | 没有回收 | 下次 bootstrap 删除无租约的其他 v2 文件；活跃文件受锁保护 |
| 边界 | 旧文件可长期残留 | 不清理 main 的旧文件；长期活跃进程仍可能保留多份 v2 文件 |

故障现场的设备 tmpfs 为 **4 GiB**，978 个历史 SO 占用约 **4.08 GB**，新文件写入返回 `ENOSPC`；同名重写也需要临时副本空间。`npu-smi` 的 HBM 使用量正常不能排除 tmpfs 配额耗尽。[现场数据](https://github.com/hw-native-sys/simpler/pull/2651)。

本机受控测试中，约 19 KiB SO 的设备文件复用未测出启动收益；约 4.1 MiB SO 节省约 **0.9 ms**，完整进程约 **2.05 s**。这些是合成 SO 的启动测试，不代表生产任务耗时。原始 CSV 在本机仓库外的 `../so-cache-bench/results2/raw.csv`（从仓库根目录起算）。
