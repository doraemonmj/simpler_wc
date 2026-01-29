#include <cstdint>

#include "aicore.h"
#include "runtime.h"

constexpr uint32_t AICORE_COREID_MASK = 0x0FFF;
/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

/**
 * Task execution wrapper - dispatches tasks using function pointers
 *
 * This function demonstrates the runtime function pointer dispatch pattern.
 * Following the production system flow:
 * - functionBinAddr points to compiled kernel code in device GM memory
 * - The address is cast to a function pointer: UnifiedKernelFunc kernel =
 * (UnifiedKernelFunc)functionBinAddr
 * - The kernel is invoked: kernel(task->args)
 *
 * This is the KEY difference from compile-time linking:
 * - OLD: extern "C" declarations, resolved at link time
 * - NEW: functionBinAddr from GM memory, cast at runtime
 *
 * With unified kernel signature, no switch statement is needed.
 * All kernels unpack their own arguments from the args array.
 *
 * @param task Pointer to task in global memory (null during initialization)
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task* task) {
    // Null task pointer indicates no work assigned (initialization state)
    if (task == nullptr) {
        return;
    }

    // Check for valid functionBinAddr
    if (task->functionBinAddr == 0) {
        // Invalid address - skip execution
        return;
    }

    // Cast functionBinAddr to unified function pointer and invoke
    // All kernels have signature: void kernel(__gm__ int64_t* args)
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)task->functionBinAddr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));
}

__aicore__ __attribute__((weak)) void AicoreExecute(__gm__ Runtime* runtime, int blockIdx, int coreType) {
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[blockIdx]);

    uint32_t physicalCoreId = static_cast<uint32_t>(get_coreid()) & AICORE_COREID_MASK;
    my_hank->aicore_done = physicalCoreId + 1;
    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    volatile uint32_t task_id = 0, lastTaskId = 0;
    // Phase 3: Main execution loop - poll for tasks until quit signal

    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

#ifdef ENABLE_REGISTER_FEATURE
        // Read task_id from register
        __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(task_id));
        // Check for quit signal (AICORE_TASK_STOP = 0x7FFFFFF0)
        if (task_id == AICORE_TASK_STOP) {
            break;  // Exit kernel
        }

        if (task_id != 0 && task_id != lastTaskId) {
            set_cond(1);      
            __gm__ Task* task_ptr = &(runtime->tasks[task_id - 1]);
            execute_task(task_ptr);
            // Mark task as complete (task_status: 0=idle, 1=busy)
            lastTaskId = task_id;
            set_cond(0);
        }
#else

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0 means valid Task* pointer)
        if (my_hank->task != 0) {
            __gm__ Task* task_ptr = reinterpret_cast<__gm__ Task*>(my_hank->task);
            execute_task(task_ptr);
            // Mark task as complete (task_status: 0=idle, 1=busy)
            my_hank->task_status = 0;
        }
#endif
    }
}