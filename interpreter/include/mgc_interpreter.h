#ifndef MGC_INTERPRETER_H
#define MGC_INTERPRETER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "mgc_fp16_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Memory Callback Function Types (For QEMU, flat memory, or testbenches)
// =============================================================================
typedef uint32_t (*mgc_mem_read_fn)(void *user_ctx, uint32_t byte_addr);
typedef void (*mgc_mem_write_fn)(void *user_ctx, uint32_t byte_addr, uint32_t data_word);

// =============================================================================
// Architectural GPU Core State Structure
// =============================================================================
typedef struct {
    // Scalar Registers: r0 through r63
    uint32_t regs[64];

    // Condition Registers: cr0 through cr7 (1 bit each, packed into uint8_t)
    uint8_t  cr;

    // 4-Lane Vector Register File: v0 through v7 (8 vectors * 4 lanes * 32-bit)
    uint32_t vregs[8][4];

    // Program Counter
    uint32_t pc;
    uint32_t pc_prev; // PC of preceding cycle (for relative branch calculation)

    // Immediate latch (for atomic 2-step immediate loading)
    uint16_t imm16_l;

    // Upper register bank prefix latch (one-shot, clears after 1 instruction)
    // Bits: [2] = srcA_hi, [1] = srcB_hi, [0] = dst_hi
    uint8_t  pending_reg_bank;

    // Execution Status
    bool     stop_core;
    bool     irq;
    bool     trace_enabled;

    // Latency & Statistics
    uint64_t instruction_count;
    uint64_t cycle_count;

    // Memory Callback Hooks
    mgc_mem_read_fn  mem_read;
    mgc_mem_write_fn mem_write;
    void            *user_ctx;

} mgc_gpu_state_t;

// =============================================================================
// Public API Functions
// =============================================================================

/**
 * @brief Initializes the GPU state structure with memory callbacks.
 */
void mgc_init(mgc_gpu_state_t *state,
              mgc_mem_read_fn read_cb,
              mgc_mem_write_fn write_cb,
              void *user_ctx);

/**
 * @brief Resets all registers, PC, and status flags to power-on default.
 */
void mgc_reset(mgc_gpu_state_t *state);

/**
 * @brief Steps the interpreter by one single instruction.
 * @return true if execution should continue, false if stopped or error.
 */
bool mgc_step(mgc_gpu_state_t *state);

/**
 * @brief Runs instructions until stop_core is executed or max_instructions is reached.
 * @return Total number of instructions executed.
 */
uint64_t mgc_run(mgc_gpu_state_t *state, uint64_t max_instructions);

/**
 * @brief Dumps current architectural state (PC, registers, CR) to stream.
 */
void mgc_dump_state(const mgc_gpu_state_t *state, FILE *out);

/**
 * @brief Helper: Convert IEEE-754 FP32 float to IEEE-754 binary16 (FP16).
 */
uint16_t mgc_float_to_fp16(float f);

/**
 * @brief Helper: Convert IEEE-754 binary16 (FP16) to IEEE-754 FP32 float.
 */
float mgc_fp16_to_float(uint16_t h);

/**
 * @brief Helper: Calculate square root of IEEE-754 FP16 value (Channel 7 Sub-op 2).
 */
uint16_t mgc_fp16_sqrt(uint16_t a);

/**
 * @brief Helper: Calculate reciprocal square root of IEEE-754 FP16 value (Channel 7 Sub-op 3).
 */
uint16_t mgc_fp16_rsqrt(uint16_t a);

#ifdef __cplusplus
}
#endif

#endif // MGC_INTERPRETER_H
