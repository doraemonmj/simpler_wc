/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#include "aicpu/platform_aicpu_affinity.h"

#include <atomic>
#include <cstdint>
#ifdef __linux__
#include <sched.h>
#endif

#include "common/unified_log.h"

static constexpr int32_t MAX_GATE_THREADS = 16;

// HT-enabled topology: only keep threads on these logical CPUs.
// The LAST entry is assigned as orchestrator; all others are schedulers.
static constexpr int32_t ALLOWED_CPUS[] = {6,7,13,14,4 };
static constexpr int32_t ALLOWED_CPU_COUNT = sizeof(ALLOWED_CPUS) / sizeof(ALLOWED_CPUS[0]);

static std::atomic<int32_t> s_reported{0};
static std::atomic<int32_t> s_cpu_written{0};
static std::atomic<int32_t> s_gate_init{0};
static std::atomic<int32_t> s_gate_ready{0};

static int32_t s_thread_cpu[MAX_GATE_THREADS];
static bool s_thread_survive[MAX_GATE_THREADS];
// Deterministic executor index: ALLOWED_CPUS position maps to thread_idx.
// ALLOWED_CPUS[0..N-2] → 0..N-2 (sche), ALLOWED_CPUS[N-1] → N-1 (orch).
static int32_t s_thread_exec_idx[MAX_GATE_THREADS];

static thread_local int32_t tl_exec_idx = -1;

bool platform_aicpu_affinity_gate(int32_t /*logical_count*/, int32_t total_launched) {
    if (ALLOWED_CPU_COUNT >= total_launched) {
        tl_exec_idx = -1;
        return true;
    }

    // Assign thread index
    int32_t idx = s_reported.fetch_add(1, std::memory_order_acq_rel);

    // Report CPU
#if defined(__aarch64__)
    int32_t cpu = sched_getcpu();
#elif defined(__x86_64__)
    int32_t cpu = sched_getcpu();
#else
    int32_t cpu = -1;
#endif

    LOG_INFO_V0("AICPU affinity gate: thread idx=%d sched_getcpu=%d", idx, cpu);

    if (idx < MAX_GATE_THREADS) {
        s_thread_cpu[idx] = cpu;
    }
    s_cpu_written.fetch_add(1, std::memory_order_release);

    // Barrier: wait until all threads have written their CPU
    while (s_cpu_written.load(std::memory_order_acquire) < total_launched) {}

    // CAS winner does allowlist classification
    int32_t expected = 0;
    if (s_gate_init.compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        for (int32_t i = 0; i < total_launched; ++i) {
            s_thread_survive[i] = false;
            s_thread_exec_idx[i] = -1;
        }

        int32_t allowed_cnt = 0;
        for (int32_t tid = 0; tid < total_launched; ++tid) {
            int32_t c = s_thread_cpu[tid];
            if (c < 0) continue;
            for (int32_t a = 0; a < ALLOWED_CPU_COUNT; ++a) {
                if (c == ALLOWED_CPUS[a]) {
                    s_thread_survive[tid] = true;
                    s_thread_exec_idx[tid] = a;
                    allowed_cnt++;
                    break;
                }
            }
        }

        LOG_INFO_V0(
            "AICPU affinity gate: allowed_cnt=%d total_launched=%d orch_cpu=%d",
            allowed_cnt, total_launched, ALLOWED_CPUS[ALLOWED_CPU_COUNT - 1]
        );

        s_gate_ready.store(1, std::memory_order_release);
    }

    // Wait for classification to complete
    while (s_gate_ready.load(std::memory_order_acquire) == 0) {}

    bool survive = (idx < total_launched) ? s_thread_survive[idx] : false;
    tl_exec_idx = (idx < total_launched) ? s_thread_exec_idx[idx] : -1;

    // Last thread resets state for next invocation
    static std::atomic<int32_t> s_cleanup{0};
    int32_t cleanup_idx = s_cleanup.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
        s_reported.store(0, std::memory_order_release);
        s_cpu_written.store(0, std::memory_order_release);
        s_gate_init.store(0, std::memory_order_release);
        s_gate_ready.store(0, std::memory_order_release);
        s_cleanup.store(0, std::memory_order_release);
    }

    if (!survive) {
        LOG_INFO_V0("AICPU affinity gate: thread idx=%d cpu=%d DROPPED", idx, cpu);
    } else {
        LOG_INFO_V0("AICPU affinity gate: thread idx=%d cpu=%d exec_idx=%d %s",
                 idx, cpu, tl_exec_idx,
                 tl_exec_idx == ALLOWED_CPU_COUNT - 1 ? "ACTIVE(orch)" : "ACTIVE(sche)");
    }

    return survive;
}

int32_t platform_aicpu_affinity_thread_idx() {
    return tl_exec_idx;
}
