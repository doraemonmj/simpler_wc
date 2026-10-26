# A5 EVB L0 onboard probe, October 2026

The tests below ran on device 0 of an `Ascend950PR_9599` EVB using CANN 9.1.0
and driver `7.0.t9.0.B848`, against simpler main `199572b87`. This machine has
no working `npu-smi info`. Each profiling test used one BIU channel at a time.
The initial probes used at most one kernel block; a later 72-block attempt
is recorded below. No test wrote perfmon registers or reset the device.

| Probe | Result |
| --- | --- |
| Enumerate driver channels | `prof_drv_get_channels(0)=0`, 35 channels returned; IDs 11–28 are listed, but their name fields are empty. |
| Channel 11, no kernel | `prof_drv_start=0`, `prof_channel_read=0` bytes, `prof_stop=0`; start and stop each took about 20 ms. |
| Channel 11, no kernel, retest 2026-10-23 | `prof_drv_start=-1` in about 21 ms. The isolated driver log again has subevent 96 / `prof_urma_start` failure. No read or stop followed because start failed. |
| Channel 12, single-block vector-compiled kernel registered with generic AICore ELF magic, no ACL profiling session or `PROF_INSTR` switch | `prof_drv_start=0`; launch and 5 s bounded sync succeeded. The kernel wrote `core_id=0`, `block_idx=0`, marker `0x905a5`. Channel read returned 0 bytes. `halProfDataFlush` returned `65534` (not supported); stop and all normal cleanup succeeded. Later controls showed that this registration did not verify AIV execution. |
| Channel 13, no kernel, 2026-10-23 | `prof_drv_start=0` in about 20 ms; `prof_channel_read=0` bytes; `prof_stop=0` in about 14 ms. No kernel was run, so the empty read is expected. |
| Channel 13, follow-up 2026-10-23 | A fresh one-block kernel probe reached `prof_drv_start(13)=-1` before the profiling switch or kernel. A second minimal no-kernel retry also returned `-1`; both have the same subevent-96 response failure. |
| Channel 15, first use with one generic-magic block, 2026-10-23 | `prof_drv_start(15)=0`, then `rtProfSetProSwitch(START)=0`, kernel launch/sync/copy all returned 0 and the marker was `0x905a5`. Ten channel-read attempts over about 1 s returned 0 bytes; `halProfDataFlush=65534` and the read after flush also returned 0. `prof_stop(15)=0`, switch STOP=0, and ordinary cleanup returned 0. The probe exited 1 solely because it requires nonempty samples. Runtime debug log reports `ReportSoftwareSqEnableToMsprof: ... ret=6` despite switch START returning 0. This run did not verify AIV execution. |
| Channel 16, first use with full direct-channel sequence, 2026-10-23 | ACL profiling init/start, `PROF_INSTR` START, `prof_drv_start(16)`, single-block kernel launch/sync, and copy all returned 0; kernel marker was `0x905a5`. Ten channel-read attempts and a read after unsupported `halProfDataFlush=65534` returned 0 bytes. Channel stop, switch STOP, ACL profiling stop/finalize, and other cleanup returned 0. Despite `aclprofStart=0`, its internal channel 52 start failed with subevent-96 response `-1` in the debug log. The probe exited 1 only because no samples were read. |
| Channel 14, same-process reopen, 2026-10-23 | With no kernel or ACL profiling session, first `prof_drv_start(14)=0` (~20 ms), `prof_channel_read=0` bytes, and `prof_stop(14)=0` (~15 ms). After a 2 s pause in the **same process**, second `prof_drv_start(14)=-1` (~13 ms) with the same subevent-96 response failure. `aclFinalize=0`; device lock released. |
| Channel 12, `PROF_INSTR` switch only | Switch START returned 0, but `prof_drv_start=-1`. Switch STOP and all normal cleanup succeeded. |
| Channel 12, ACL profiling session plus `PROF_INSTR` | `aclprofInit`, `aclprofStart`, and switch START returned 0; `prof_drv_start=-1`. Session and switch teardown all returned 0. The profiler log also shows channel 52 failing to start. |
| Channel 12, ACL profiling session without `PROF_INSTR` (2026-10-23) | One bounded retry using the same single-block probe: `aclprofInit=0`, `aclprofStart=0`, then `prof_drv_start=-1` before kernel launch. All ordinary cleanup calls returned 0; no device process holder remained. No L0 data was collected. |
| Channel 12, no ACL session, intended switch after channel start (2026-10-23) | Bounded one-channel retry on latest main: `aclInit`, device bind, one 64-byte allocation, stream creation, and kernel registration succeeded; `prof_drv_start=-1` before the switch or kernel launch. The log again reports `subevent_id=96` / `prof_urma_start` for channel 12. Kernel unregister, stream destruction, memory free, and `aclFinalize` all returned 0; the device lock was released. No data read was reached. |
| Channel 12, no kernel or stream setup | `prof_drv_start=-1`. This is not directly comparable with the native-kernel probe because its setup differs. |

For the failed channel-12 starts, driver logs report
`prof_event_get_submit_sync_result: result=-1, subevent_id=96`, followed by
`prof_urma_start: ret=-1, chan_id=12`. The full-session run reports the same
URMA error on channel 52 (`CHANNEL_STARS_SOC_PROFILE_BUFFER`). The successful
native-kernel run produced no L0 channel data, so this EVB has **no verified
working L0 trace path** under these tested conditions. The results do not show
that channel 12 is permanently unavailable: it started once, before the
profiling-switch tests. The zero-byte read does not establish why capture was
empty: this tiny kernel had no final `bar.all`, and its generic ELF registration
did not target the AIV cores as the later AIVEC-magic control did.

The installed `ascend_hal_define.h` maps subevent 96 to
`DRV_SUBEVENT_PROF_START_MSG`. In the ACL-session runs, channel 52 fails at
this event submission before our explicit channel-12 start fails at the same
point. This localizes the failure to the profiling start handshake, before
any kernel or `mark_stamp` collection. The available host log does not give
the underlying reason for the event's `-1`; it does not establish a hardware,
firmware, URMA transport, or parameter root cause.

The probe sources are `src/a5/platform/tools/l0_biu_channel_probe.py`,
`l0_biu_kernel_probe.cpp`, and `l0_probe_core.cpp`. The native kernel emits two
`mark_stamp<PIPE_MTE3>` calls and writes only a 64-byte result allocation.
The C++ probe currently reproduces the ACL-session-plus-switch failure and has
an `L0_PROBE_SKIP_PROF_SWITCH=1` option for the ACL-session-only check. Built binaries
and per-process logs are under ignored `build/l0_probe/` and `l0_probe_logs/`.
`torch_npu` was also tried as a small workload, but its initialization rejected
this SoC (`Unsupported soc version: Ascend950PR 9599`), before any kernel ran.

The early `core_swimlane_principle.md` describes `aclrtSetDeviceResLimit(6/12)`
as a prerequisite, but the later `prototype/l0swimlane-v2` DeviceRunner
explicitly removed it because limiting core counts cannot select the six
physical clusters monitored by the channels. We did not change device resource
limits on this EVB. Issue #905 records successful self-managed buffer writeback
in its earlier environment; these small channel probes do not replicate that
AICPU register programming and buffer path.

The 72-block timeout was subsequently isolated to the probe's generic ELF
registration and fixed with AIVEC magic. Further profiling work should
explain the URMA `subevent_id=96` reopening failure and determine whether
perfmon `en/gen` and write counters change during a correctly targeted AIV
kernel. Only then can raw record decoding, DMA completion, and buffer
ownership be tested for
[issue #905](https://github.com/hw-native-sys/simpler/issues/905).

## Process-level channel recovery probe, 2026-10-26

The host had not rebooted since October 22. Under the device-0 lock, a bounded
probe used the previously consumed BIU channel 12 without launching a kernel,
allocating GM, or writing perfmon registers (`task_20261026_112650_765462`):

| Operation | Return value |
| --- | ---: |
| `prof_drv_start(0, 12)` | `-1` |
| `prof_stop(0, 12)` after failed start | `30` |
| `prof_drv_start(0, 12)` after stop | `-1` |
| `aclrtResetDevice(0)` | `0` |
| `aclrtSetDevice(0)` after reset | `0` |
| `prof_drv_start(0, 12)` after reset | `-1` |
| Final `aclrtResetDevice(0)` and `aclFinalize()` | both `0` |

Thus an extra stop and a successful ACL process-level reset did not restore
channel 12. The meaning of stop return value `30` is not documented in the
installed public profiling header. Disassembly of this driver build shows
that `prof_chan_stop` returns `30` at its local channel-state check, before
calling the registered stop handler; this attempt did not submit a remote
stop event. It does not prove whether the remote channel remained open.
[CANN's `aclrtResetDevice` documentation](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_0040.html)
describes release of the caller's contexts and streams; process exit already
does this. This probe does not establish whether profiling state persists in
the driver, firmware, or device service. This first probe did not perform a
force, PCIe, or server reset. A subsequent ordinary vector ST passed in 8.17 s
under the device lock (`task_20261026_112930_771718`), confirming the normal
onboard path still worked after the probe.

A second bounded channel-12 probe changed only `prof_start_para.real_time`
from `PROF_REAL` to the public `PROF_NON_REAL` mode, using the same BIU user
data and no kernel (`task_20261026_113422_775472`).
`prof_drv_start(0, 12)` still returned `-1`; `aclrtResetDevice` and
`aclFinalize` both returned `0`. Mode selection did not recover the consumed
channel. The installed public HAL header exposes start, stop, read, poll, and
flush for profiling channels, but no per-channel reset or remote-state query.
The private `prof_urma_stop` symbol requires internal session pointers, so it
was not called directly. A full device/PCIe reset was not attempted on this
fragile EVB.

With no other process holding `/dev/davinci0`, one locked probe also tested
the public ACL force variant (`task_20261026_113605_777185`). Baseline
`prof_drv_start(0, 12)` returned `-1`; `aclrtResetDeviceForce(0)` returned
`0` in 4.94 s; rebind returned `0`; another `prof_drv_start(0, 12)` returned
`-1`. Final normal ACL cleanup returned `0`. A subsequent ordinary vector ST
passed in 6.48 s (`task_20261026_113739_782020`). Therefore neither ACL
reset variant restored the consumed BIU channel in this test. The force API
releases runtime device resources; it is not a demonstrated reset of remote
profiling-channel state. No PCIe/hardware or server reset was performed.

## Ordinary onboard health check, 2026-10-23

Ran the existing small vector scene test on device 0:

```bash
.venv/bin/python -m pytest \
  examples/a5/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
  --platform a5 --device 0 --pto-session-timeout 1200 -q
```

Result: **1 passed in 6.48 s**, process exit code 0. Its isolated process logs
contain no `prof_urma_start`, `subevent_id=96`, or `aclrtResetDeviceForce`
entry, and `/dev/davinci0` had no process holder after the test. This confirms
the ordinary A5 onboard path worked at that time. It does not validate the L0
profiling path that failed above.

## Historic L0 prototype attempt, 2026-10-23

Checked out `prototype/l0swimlane-v2` at `cadf8dd3a` in a separate worktree,
built it in a worktree-local virtual environment against the installed CANN
9.1.0, and changed only `vector_mul_stamp`'s test config to
`block_dim=1, aicpu_thread_num=1`. This bounds a subsequent old perfmon probe
to three 64 MiB buffers (192 MiB) instead of the original 108 buffers
(approximately 6.9 GiB). No old perfmon experiment was enabled in the first
onboard run; this was a prerequisite baseline test.

That old baseline **failed before profiling was enabled**. Device logs show a
one-second `simpler_aicpu_exec` timeout and AICPU task exception. The host then
reported `rtKernelLaunchWithHandleV2=207001`, after the device context was
already failing; that code alone is not evidence of a real host-memory shortage.
Current main documents a related one-second op-timeout/slow-launch problem
and its later three-second timeout plus launch-order fix in
`docs/investigations/2026-06-pa-unroll-207001-optimeout-window.md`; this is
consistent with the old baseline failure, but the old run's exact root cause
was not isolated.
All old-hardware tests were stopped at this point. In particular, the old
experiments that program perfmon MMIO, redirect `base_addr`, or enable 108
cores were **not run**, so none of their data-path claims was reproduced on
this EVB. The attempt's logs are under the separate worktree's
`l0_repro_logs/baseline/` directory.

After this failure, the current main worktree's ordinary single-case vector
ST was run once through the device lock and **passed** (`1 passed in 6.44 s`).
The device lock was released, no process held `/dev/davinci0`, and the two UB
links reported ACTIVE. This is a bounded health check, not an L0 result.

## Read-only analysis of `prof_drv_start=-1`, 2026-10-23

The installed `libascend_hal.so` still contains local symbols. Disassembly of
`prof_event_submit_event_sync` shows it calls `halEschedSubmitEventSync` and,
only when that call returns zero, enters `prof_event_get_submit_sync_result`.
The error log at `prof_event_comm.c:166` comes from the latter function: the
synchronous event submission completed, but the returned profiling response
contained `-1`. `prof_urma_send_start_event` packs subevent 96
(`DRV_SUBEVENT_PROF_START_MSG`) and propagates that result. Thus the observed
failure is in the profiling start response, rather than a direct failure to
submit the synchronous event. The response does not identify why the remote
profiling handler returned `-1`.

The same event/response failure occurred for channel 52, started internally
by `aclprofStart`, before the explicit channel-12 call. Channel 12 also failed
without an ACL profiling session. This makes a channel-12-specific user-data
mistake or profiling-switch order a weak explanation, but does not rule out a
shared profiling-service prerequisite or device/driver state issue.
The later one-channel controls distinguish per-channel behavior: 11 now fails
with the same event response, while 13 starts and stops successfully. The
profiling event path therefore works for at least one BIU channel on this EVB;
these results do not establish a global profiling-service outage.
The subsequent channel-13 retries failed in fresh processes, while previously
unused channel 15 started on its first attempt. This repeated first-success /
later-failure pattern raises the possibility of per-channel state left behind
after stop, but the available response code cannot establish its cause.
The channel-15 zero-byte read does not prove that perfmon produced no data:
this one-block vector-compiled kernel was registered with generic AICore ELF
magic and did not verify execution on the monitored AIV subcore. The run also
omitted the ACL profiling session. The runtime
`ReportSoftwareSqEnableToMsprof` error is another sign that the switch alone
did not establish a verified full instr-profiling route.
The untouched channel-16 full-sequence attempt shows that the BIU channel can
start and the small kernel can run while the ACL profiling session reports
success. However its asynchronous/internal channel-52 start fails, so the
profiler prerequisites were not demonstrably complete. Its generic-magic
single block did not verify execution on channel 16's monitored AIV subcore.
These two limits prevent assigning the zero-byte read to a single cause.

## Stop semantics, channel 52, and empty reads (read-only analysis)

The first-use-then-failure sequence is now visible for BIU channels 11, 12,
and 13. Each successful probe reported `prof_stop=0`; later fresh processes
got `-1` in the response to start subevent 96. Fresh channels 15 and 16 could
still start. This pattern is consistent with per-channel state persisting
after a stop, but is not proof that a channel remains open. There is no
reported busy/occupied-channel status or device-side state query in the
available evidence.
The targeted channel-14 test strengthens that pattern: start → read → stop
returned 0, yet a second start 2 s later in the same process received `-1`.
Thus incomplete process-exit cleanup alone cannot explain the failure.
It still does not reveal whether the remote channel is literally open, its
resources were not released, or the remote profiler rejects reopening for
another state-machine reason.

On October 23, a separate tiny probe treated previously used BIU channel 12
as if it were still open: after normal ACL/device and one-block kernel
registration, it called `prof_channel_read(0, 12, ..., 4096)` directly,
without `prof_drv_start`, a profiling switch, or a kernel launch. The call
returned `-4`. The installed driver header names this result
`PROF_STOPPED_ALREADY`; the probe stopped there and released the device lock.
This rules out directly reusing channel 12's old host read path through the
public API. It does not determine whether the remote/device-side profiling
handler still considers the channel open, since the old process and its
host-side consumer buffer have already exited.

## All-AIV coverage attempt and ordinary 36-block control, 2026-10-23

The machine configuration and the ordinary runtime report 36 clusters and 72
AIV cores. To test whether earlier empty reads were caused by the one-block
kernel landing outside a monitored core, a new minimal AIV kernel wrote one
64-byte marker per task block. With a fresh BIU AIV0 channel 18, ACL profiling,
and `PROF_INSTR`, all setup and `prof_drv_start(18)` returned 0. Launching 72
AIV blocks returned 0, but the 5-second stream sync returned `507046`
(`ACL_ERROR_RT_STREAM_SYNC_TIMEOUT`). The probe therefore did not copy block
markers or reach `prof_channel_read`. `prof_stop(18)`, switch STOP, and ACL
profiling teardown returned 0. Stream destruction did not finish before the
task's 180-second limit; the task exited 130 and the device-0 lock was
released. This attempt gives **no profiling data** and no verified AIV
coverage. The timeout's cause is not established by this result. Its isolated
process log again shows channel 52 failing at profiling start subevent 96;
the runtime timeout snapshot names kernel `l0_probe_0` on stream 61. The same
ACL session collected 96 bytes on channel 50 (`stars_soc.data`), which is
STARS metadata rather than a BIU L0 trace.

After that timed-out attempt, the existing ordinary onboard
`available_aicore_counts` ST passed twice under the device-0 lock (8.28 s and
6.41 s). On the second run, the test printed its device-reported counts:
`cluster_count=36, aiv_count=72`. It launched one MIX cohort of exactly 36
blocks; every block's AIC, AIV0, and AIV1 wrote their expected output slots,
which the test checked. Both runs exited 0 and released the lock. This
confirms ordinary full-width execution still works on the EVB; it does not
exercise the direct profiling capture path or explain the 72-block probe
timeout.

## Registration-magic control and AIV-covered capture, 2026-10-23

The same vector-compiled marker kernel was first run without any profiling
channel, ACL profiling session, or `PROF_INSTR` switch. With the original
generic `RT_DEV_BINARY_MAGIC_ELF` registration, 1 and 36 blocks completed
(`valid_blocks=1/1` and `36/36`), while 72 blocks again timed out with
`507046`; its 60-second task limit terminated the hung cleanup and released
the lock. A subsequent ordinary 36-block MIX ST passed. The repository's
small AIV multiblock ST (4, 16, 24, 48, then 96 AIV blocks) also passed.
Thus full-width ordinary execution and AIV scheduling remained healthy.

The installed runtime header distinguishes generic AICore ELF magic from
`RT_DEV_BINARY_MAGIC_ELF_AIVEC`. With only the registration magic changed to
the AIVEC value, the **same kernel** completed at 1 and 72 blocks with no
profiling. The 72-block output verified all 72 task slots and distinct AIV
core IDs `18..53` and `72..107`; all ordinary cleanup calls succeeded. This
isolates the previous 72-block timeout to the probe's inappropriate generic
registration/launch mode, rather than the profiler channel or a general
full-width AIV hardware fault. The early generic-magic one-block probes did
not establish that work ran on AIV.

An untouched AIV1 BIU channel 19 was then started without ACL profiling or
the `PROF_INSTR` switch (`prof_drv_start(19)=0`). The correct AIVEC-registered
72-block kernel completed with all 72 AIV markers, but channel reads over
about 1 second returned **0 bytes**. `halProfDataFlush=65534` remained
unsupported; `prof_stop(19)=0` and cleanup succeeded. One further untouched
AIV0 channel 21 used the same setup with a 10-second read window. It also
started, ran all 72 AIV blocks, and returned **0 bytes** throughout; stop
and cleanup succeeded. Both task runs released the device-0 lock. These
controls exclude wrong core placement and a roughly one-second wait as the
sole explanations for these two empty reads. They do not prove whether the
hardware perfmon engine was enabled or wrote bytes into its target buffer;
that requires observing `en/gen` and write counters or a self-managed buffer.
Because these runs omitted ACL profiling, channel-52 failure was not their
direct cause. No MMIO perfmon configuration was attempted on this EVB.

## Existing PMU collection control, 2026-10-23

As a low-impact check of the separate AICPU counter collection and host-output
path, the existing onboard `dfx/pmu/test_pmu.py` ran with `--enable-pmu 2` under
the device-0 lock. It passed in 8.20 s and produced a new `pmu.csv` with data
rows. The first AIV row reported `pmu_total_cycles=11451` and nonzero MTE,
scalar, and vector busy counters; all cleanup completed and the lock was
released. This proves the ordinary PMU collector can read and export nonzero
per-task counters in the current environment. PMU is a different register
source from BIU/perfmon L0 instruction trace, so this result does not resolve
the zero-byte BIU reads.

Disassembly of the installed `libascend_hal.so` shows the URMA stop path sends
a synchronous stop event (subevent 97). In `prof_urma_chan_stop`, the result of
`prof_urma_stop` is stored but then overwritten with a zero return after
`prof_buff_wait_read_empty`; `prof_stop=0` therefore cannot, by itself, prove
that the remote stop succeeded or that remote channel state was cleared. No
stop-event failure was printed in the per-process debug logs for the successful
runs, so a failed remote stop is not established either.

Channel 52 is `CHANNEL_STARS_SOC_PROFILE_BUFFER`, distinct from the BIU AIV
channels. The `aclprofStart` profiling job starts it internally along with
CCU, STARS, AICPU, and other channels. In the channel-16 full-flow log, these
other starts largely succeeded, but channel 52's `DrvPeripheralStart` received
the same subevent-96 response `-1`; `aclprofStart` still returned zero. This
was already the result of the first observed ACL-session attempt in these
local logs (October 22, 20:29). No successful start of channel 52 is recorded
here, so the "channel 52 also works only once" hypothesis is unverified.
Channel 52's failure is a concrete sign that the ACL session did not complete
all of its internal data channels, but it does not by itself prove which BIU
trace prerequisite is missing.

`prof_channel_read=0` means no bytes had reached the host consumer buffer at
read time. For channels 15/16, the probe requested physical cluster 9 based
on the old mapping, but the one-block generic-magic kernel did not verify
execution on that cluster / AIV subcore; the mapping itself has not
been reconfirmed on this EVB.
The channel-16 ACL session also had the channel-52 failure. Finally, the tiny
kernel has no trailing `bar.all`, the read window was about one second, and
`halProfDataFlush` returned not-supported; delayed or unflushed trace data
cannot be excluded. These are distinct possible reasons for the empty read,
and the current probe does not discriminate among them. No further device
operation was performed for this analysis.

System journal entries after process exit include UDMA ungrant/free errors
(`-22`), but the earlier process whose channel 12 started successfully had
the same cleanup errors. They are therefore not evidence for this start
failure. No active profiling process was found during the read-only check.
Global device logs under `~/ascend/log/debug` are root-only on this host, and
the available per-process device log has no corresponding profiling-handler
diagnostic. The underlying rejection reason remains unproven. No new device
probe, reset, or service operation was performed in this analysis.

### Could the first successful start have left device state behind?

The successful channel-12 run was process 124887 at 20:26 on October 22; it
reported `prof_stop(12)=0` and normal process cleanup. The next channel-12 run,
process 126837 at 20:27, enabled `PROF_INSTR` before channel start and received
the subevent-96 response `-1`. A fresh process without ACL profiling or a
profiling switch still failed at channel start at 10:27 on October 23. Thus
process-local stale state and a switch-order-only explanation are insufficient;
persistent device-side profiling state remains a possibility.

There is no visible busy/occupied-channel error, and the available API/logs do
not expose remote profiling-handler state. Channel 13 succeeding while 11 and
12 fail leaves per-channel residual state or request-dependent handling as
possibilities. A zero return from `prof_stop` is
evidence that the stop call completed, not proof that every device-side state
was cleared. The same URMA cleanup warnings followed both successful and
failed runs, so they do not distinguish a residual-state mechanism. No reset
was performed. Only the bounded channel-11 and channel-13 start/read/stop
controls above were added at this stage, each under the device-0 lock. Later,
one single-block channel-15 run also used that lock; no reset or service
operation was performed.

## Read-only AIV1 perfmon register control, 2026-10-23

The new `l0_probe_regs_aiv.cpp` runs 72 tiny AIV blocks with the correct
`RT_DEV_BINARY_MAGIC_ELF_AIVEC` registration. It reads nine 32-bit perfmon
registers through `ld_dev` **only on core ID 73**, the AIV1 subcore of physical
cluster 18. The host maps that cluster once with
`halResMap(RES_AICORE, PROCESS_CP1)`; the reported mapping is 3 MB, and AIV1's
register base is the mapped address plus 2 MB. The kernel only reads MMIO; it
does not configure perfmon or write registers. Every run below completed all
72/72 AIV blocks and released the device-0 lock.

| Probe | `en` | `global_en` | `buf_len` / base / `wptr` | `samp_crt` / `samp_wrt` |
| --- | ---: | ---: | --- | ---: |
| No channel, before channel 22 (`task_20261023_155552_296047`) | 0 | 0 | all 0 | `0x40` / `0x40` |
| Fresh AIV1 channel 22 started (`task_20261023_155611_296287`) | **1** | **1** | all 0 | `0x40` / `0x40` |
| No channel, after `prof_stop(22)` (`task_20261023_155645_296541`) | 0 | 0 | all 0 | `0x60` / `0x60` |
| Second no-channel control (`task_20261023_155704_296776`) | 0 | 0 | all 0 | `0x60` / `0x60` |

Channel 22's `prof_drv_start=0`, kernel launch/sync/copy all returned 0, and
the register snapshot was taken inside the target kernel after its first
`mark_stamp`. The one-second channel read and read after unsupported
`halProfDataFlush=65534` both returned **0 bytes**; `prof_stop(22)=0` and
normal teardown completed. `samp_crt/samp_wrt` rose by `0x20` between the
active-kernel snapshot and the first post-stop snapshot, then stayed constant
through another no-channel kernel. These counters show activity associated
with the channel-enabled run, but they do not establish that a trace was
written to usable GM memory: base address, length, and `wptr` all read zero.

This isolates a narrower fact than the earlier empty-channel probes:
`prof_drv_start` **does enable** the target AIV perfmon instance at kernel
kickstart on this EVB, while this direct-channel-only probe has no configured
perfmon destination. The older issue #905 experiment explicitly programmed
a self-managed GM buffer through AICPU before kernel launch. At this point,
that buffer programming had not been attempted on this EVB; the follow-up
single-core test is below. The result also does not identify
why a previously used channel fails on a later `prof_drv_start`.

## Single-core AICPU-managed GM buffer, 2026-10-23

The follow-up probe used a new AICPU SO (`l0_selfbuf_aicpu.cpp`) and host
launcher (`l0_selfbuf_host.cpp`) against current main. It configured **one**
monitored AIV1 subcore: physical cluster 27 / core ID 91 / fresh BIU channel
25. The host allocated one 1 MiB GM buffer and filled it with byte `0xA5`.
AICPU read and validated the target core's idle registers, then set the
buffer's 48-bit VA, length `0x100000`, write pointer 0, glitch filter 0,
cleared sample counters, and set `global_en=1`. It left `en=0` for the
firmware's task kickstart. No ACL profiling session or `PROF_INSTR` switch
was used. A separate read-only AICPU bootstrap check passed before channel
use (`task_20261023_162731_305940`).

| Point | `en` / `gen` | length | base | `wptr` / `crt` / `wrt` |
| --- | --- | ---: | --- | --- |
| AICPU before setup | 0 / 0 | 0 | 0 | 0 / 0 / 0 |
| AICPU after setup | 0 / 1 | `0x100000` | `0x1200_00020000` | 0 / 0 / 0 |
| Core 91 inside 72-block AIV kernel | **1 / 1** | `0x100000` | `0x1200_00020000` | `0x20` / `0x20` / `0x20` |
| AICPU before teardown | 1 / 1 | `0x100000` | `0x1200_00020000` | `0x40` / `0x40` / `0x40` |
| AICPU after teardown | 0 / 0 | 0 | 0 | `0x40` / `0x40` / `0x40` |

`prof_drv_start(25)=0`; all 72/72 AIV blocks completed and core 91 ran. The
GM buffer had exactly **64 changed bytes** and retained `0xA5` elsewhere.
Its first 64 bytes were (also archived as
[`channel25_core91_one_task_from_log.bin`](../testdata/l0_onboard/channel25_core91_one_task_from_log.bin)):

```text
e19737e0806000e1e34f00e2000000e3
4a0901f0e6020a0004000a50030020f0
020001f0c90c0b0004000b50030021f0
020001f0900000f0030030f0020010f0
```

Decoded as little-endian 4-byte L0 chunks using the old
`l0_perf_profiling.h` layout, the first four chunks form a `START_STAMP`
with `block_id=55` (the 72-block task's slot on core 91). The next chunks
include region 10 on scalar (`0x000a02e6`) and MTE3 (`0x500a0004`), and
region 11 on scalar (`0x000b0cc9`) and MTE3 (`0x500b0004`). Those are the two
`mark_stamp<PIPE_MTE3, 10/11>` calls in the probe kernel, so the buffer
contains this task's L0 events. `prof_channel_read(25)` was still 0 bytes:
the channel served as the enable trigger while the hardware wrote to the
self-managed GM destination. The full run was
`task_20261023_162749_306289` and exited 0.

Teardown disabled both gates and cleared base/length **before** freeing GM.
Readback immediately after teardown and a fresh read-only AICPU check
(`task_20261023_162858_308013`) both showed `en/gen/base/len=0`. The write
pointer and sample counters remained at `0x40` despite a write of 0 to the
pointer during teardown. This is residual counter state, not a live DMA
destination; its effect on later channel reopening has not been tested.
All runs used the device-0 lock and bounded task time. No server/device reset
or unrelated register write was performed.

This reproduces issue #905's essential **single-core** self-managed data
path on this EVB. It does not yet validate a multi-core collector, per-task
draining, wrap handling, or timing alignment for a production L0 swimlane.

## Two-task onboard capture and task-boundary decode, 2026-10-23

The next run exercised a complete capture/export path on current main, with
the same host-to-AICPU dispatcher used by the normal runtime. A host launcher
(`tools/l0_trace_host.cpp`) started fresh channel 27 for cluster 35 / AIV0
core 106. AICPU configured one 1 MiB self-managed GM buffer. Two separate
72-block AIV kernels (`tools/l0_trace_aiv.cpp`) ran sequentially with distinct
region pairs (10/11 and 20/21). After each kernel synchronized, AICPU read the
hardware write pointer. The host used those byte boundaries to assign decoded
L0 chunks to task 1 or task 2; region IDs only cross-check the assignment.

| Point | Result |
| --- | --- |
| Preflight | `en=1`, `gen=0`, base/length 0; local `en` quiesced before configuration |
| Setup | `prof_drv_start(27)=0`; global enable 1; buffer base `0x1200_00031000`, length 1 MiB |
| Task 1 | 72/72 AIV blocks valid; target core 106 ran; write pointer 64 |
| Task 2 | 72/72 AIV blocks valid; same core ran; write pointer 128 |
| Teardown | `en/gen/base/length=0` before GM free; `prof_stop(27)=0` |
| Export | 128 raw bytes; eight decoded pipe events; each task has scalar and MTE3 start/end stamps |

The raw trace is archived as
[`channel27_core106_two_tasks.bin`](../testdata/l0_onboard/channel27_core106_two_tasks.bin)
with an independent hex representation and SHA-256 in the
[archive README](../testdata/l0_onboard/README.md). Its working copy is
`build/l0_probe/l0_trace_channel27.bin`; the decoded file is
`build/l0_probe/l0_trace_channel27.json`. The JSON retains core ID, block ID,
absolute AICore cycle, pipe, region, task, and raw byte offset. Task 1 owns
bytes `[0,64)` and task 2 `[64,128)`. Its four MTE3 cycles are
`89166369202354`, `89166369205445`, `89166369550876`, and
`89166369552092`, in the expected two task pairs. The locked run
`task_20261023_164958_325607` exited 0. The decoder can be rerun without
touching the device:

```bash
build/l0_probe/l0_trace_host --decode build/l0_probe/l0_trace_channel27.bin 64 build/l0_probe/l0_trace_channel27
```

An earlier version of the standalone launcher failed at its AICPU read-only
preflight with `507018`; CANN's device error reported `open so failed`. The
launcher had used full-file FNV to name the inner SO while the repository's
dispatcher names it using the ELF Build-ID. Matching the existing
`elf_build_id_64` helper resolved that failure. Those failed attempts did not
start a BIU channel or program a GM destination. The diagnostic is about the
standalone launcher, not about a driver-side perfmon read failure.

This run proves sequential task association from write-pointer boundaries on
one monitored AIV core. The task IDs in this probe are launcher IDs, not yet
the runtime's real scheduler task IDs. A normal ST integration still needs
per-core buffer ownership, real FIN/task ID windows, rollover handling, and
export through the existing diagnostics lifecycle. No device/server reset was
used.

## Full 108-subcore onboard capture, 2026-10-26

The macro-gated ordinary ST integration first captured core 107 from a 72-AIV
task on channel 28 (288 raw bytes, four region 10/11 pipe events). The next
manual ST used one 36-block MIX task. Its 108 distinct output slots passed
golden, confirming that every AIC, AIV0, and AIV1 subcore executed.

The initial multi-core sequence started a fresh channel before assigning GM
buffers. Channel 24 then refused core 0 configuration (`status=-2`). Channels
26 and 23 showed the same 73 physical subcores at `gen=1,len=0,base=0`
immediately after start; core 0 remained so through twenty 100 ms reads.
Only the other 35 subcores were configured, and all 35 yielded parseable raw
data. Those 35 comprise AIV0/AIV1 in clusters 24–35 and AIC in clusters
25–35. The two runs copied 10,016 and 19,744 `wptr` prefix bytes respectively,
of which 9,536 and 9,728 bytes were new records, with
118 expected region 10/11 events each. The 73 were left untouched, and only
configured cores were disarmed. These results establish the observed start
order effect; they do not identify why the driver sets `gen=1` on that subset.

For channel 20, AICPU preflight found all 108 `gen/base/len` idle, then
configured each core's own 1 MiB GM slice **before** `prof_drv_start(20)`.
After start, AICPU read back all 108 `gen/base/len` values and confirmed the
same addresses and lengths. The MIX task passed, and after its run fence AICPU
disarmed and confirmed all 108 before the host copied any GM data or freed
buffers. The locked task `task_20261026_093634_716644` exited 0 in 6.53 s;
there was no reset or restart.

| Result | Value |
| --- | ---: |
| ST execution | 36 AIC + 72 AIV output slots passed |
| Configured cores with nonempty raw data | 108/108 |
| Sum of per-core `wptr` prefixes copied | 48,544 bytes |
| Initialization `A5` prefixes / new records | 19,744 / 28,800 bytes |
| Decoded pipe events | 360 |
| AIC | 36 cores, two scalar region 10/11 events each |
| AIV | 72 cores, two scalar and two MTE3 region 10/11 events each |
| Region/block mismatches | 0 |
| Raw versus independent offline decoder mismatches | 0/108 |

The [full-core survey](../testdata/l0_onboard/channel20_all_physical_20261026/l0_survey.json)
and each core's `.bin`/`.json` are archived with `SHA256SUMS`; the two partial
runs and the single-core ST sample are in the same
[testdata directory](../testdata/l0_onboard/README.md).

An offline byte audit found 35 files with a leading run of the allocation's
`0xA5` sentinel. For each of those cores, the run length exactly equals the
same core's final write pointer in the preceding channel-23 capture (35/35);
channel 23 versus channel 26 also matches 35/35. The three older single-core
samples match this pattern too. Thus `wptr` behaved cumulatively across
sessions on this EVB despite the control op's attempt to write zero; the
precise register/firmware cause remains unknown. The 108 new spans all start
with a complete START_STAMP, have no internal `0xA5` gaps, and independently
decode to the stored 360 expected events. The [per-core window audit](../testdata/l0_onboard/channel20_all_physical_20261026/l0_window_audit.json)
records inferred offsets. The current implementation now records the
post-start `wptr` as the start of the run's data window, but this correction
has not yet had a fresh-channel onboard test.

This verifies full-width **raw perfmon capture and basic pipe decoding** on this EVB. It
does not yet establish runtime task-ID attribution, wrap handling, precise
clock alignment, viewer rendering, or channel reuse after stop. The current
driver has previously rejected repeat starts, so channels 20/23/24/26 should
be treated as consumed for further testing.
