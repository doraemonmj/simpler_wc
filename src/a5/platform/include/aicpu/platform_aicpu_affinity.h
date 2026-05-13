#pragma once
#include <cstdint>

// Returns true if this thread should call aicpu_execute().
// Returns false if this thread should exit (dropped).
// logical_count: desired active threads (from runtime.sche_cpu_num)
// total_launched: actual threads launched (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)
bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched);

// Returns the deterministic thread index assigned by the affinity gate.
// ALLOWED_CPUS[0..N-2] → indices 0..N-2 (sche), ALLOWED_CPUS[N-1] → index N-1 (orch).
// Only valid after platform_aicpu_affinity_gate() returned true on this thread.
int32_t platform_aicpu_affinity_thread_idx();
