#include <cstdint>
#include <cstdio>
#include <atomic>
#include <sched.h>
#include "device_log.h"
#include "graph.h"
#include "kernel_args.h"

// Static atomic counter for thread indexing
static std::atomic<int> threadId_(0);

// Forward declaration of execute function (defined in execute.cpp)
extern int execute(Graph& g, Handshake* hank, int num_aicore, int threadId);

/**
 * Handshake AICore - Initialize and synchronize with AICore kernels
 *
 * This function performs the initial handshake protocol with all AICore instances:
 * 1. Set aicpu_ready flag for each core
 * 2. Wait for each core to respond with aicore_done signal
 *
 * This ensures all cores are running and ready to receive tasks before
 * graph execution begins.
 *
 * @param arg Pointer to KernelArgs structure containing handshake buffers
 * @return 0 on success
 */
int HankAiCore(void *arg) {
    auto kargs = (KernelArgs *)arg;
    uint64_t num_aicore = kargs->block_dim * 3;

    // Phase 1: Signal all cores that AICPU is ready
    for (uint64_t i = 0; i < num_aicore; i++) {
        Handshake* hank = &kargs->graphArgs->workers[i];
        DEV_INFO("AICPU: hank addr = 0x%lx", (uint64_t)hank);
        hank->aicpu_ready = 1;
    }

    // Phase 2: Wait for all cores to acknowledge (busy-wait polling)
    for (uint64_t i = 0; i < num_aicore; i++) {
        Handshake* hank = &kargs->graphArgs->workers[i];
        // Busy-wait until AICore signals ready (aicore_done != 0)
        while (hank->aicore_done == 0) {
            // Polling loop - no sleep to minimize latency
        };
        DEV_INFO("success hank->aicore_done = %u", (uint64_t)hank->aicore_done);
    }
    return 0;
}

/**
 * Shutdown AICore - Send quit signal to all AICore kernels
 *
 * Sets the control flag to 1 for all cores, signaling them to exit
 * their execution loops and terminate gracefully.
 *
 * @param arg Pointer to KernelArgs structure containing handshake buffers
 * @return 0 on success
 */
int ShutdownAiCore(void *arg) {
    auto kargs = (KernelArgs *)arg;
    uint64_t num_aicore = kargs->block_dim * 3;
    for (uint64_t i = 0; i < num_aicore; i++) {
        Handshake* hank = &kargs->graphArgs->workers[i];
        hank->control = 1;  // Set quit signal
    }
    return 0;
}

extern "C" __attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    return 0;
}

/**
 * AICPU kernel initialization entry point
 *
 * This function is called once during kernel initialization by the CANN runtime.
 * It initializes logging and validates kernel arguments.
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to KernelArgs structure
 * @return 0 on success, -1 on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *arg) {
    InitLogSwitch();
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    DEV_INFO("%s", "Graph Executor Init: Initializing AICPU kernel");
    return 0;
}

/**
 * AICPU kernel main execution entry point
 *
 * This is the main entry point for the AICPU graph executor kernel.
 * It orchestrates the complete task graph execution:
 * 1. Handshake with all AICore instances
 * 2. Execute task graph using polling-based dispatch
 * 3. Shutdown all AICore instances
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to KernelArgs structure containing:
 *            - deviceArgs: device-specific arguments
 *            - block_dim: number of blocks (1 block = 1 AIC + 2 AIV)
 *            - graphArgs: task graph to execute (includes handshake buffers in workers[])
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }
    DEV_INFO("%s", "Graph Executor: Starting AICPU kernel execution");

    int threadId = threadId_++;

    auto kargs = (KernelArgs *)arg;

    // Step 1: Handshake with all AICore instances
    auto rc = HankAiCore(arg);
    if (rc != 0) {
        return rc;
    }

    // Step 2: Execute task graph if provided
    if (kargs->graphArgs != nullptr) {
        Graph* g = kargs->graphArgs;
        Handshake* hank = kargs->graphArgs->workers;
        int num_aicore = kargs->block_dim * 3;
        DEV_INFO("Graph has %d tasks", g->get_task_count());
        int completed = execute(*g, hank, num_aicore, threadId);
        DEV_INFO("Executed %d tasks from graph", completed);
    }

    // Step 3: Shutdown all AICore instances
    rc = ShutdownAiCore(arg);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("%s", "Graph Executor: Kernel execution completed successfully");
    return 0;
}
