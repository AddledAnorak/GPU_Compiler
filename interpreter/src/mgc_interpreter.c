#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mgc_interpreter.h"

// Forward declarations from mgc_fp16_math.c
uint16_t mgc_fp16_add(uint16_t a, uint16_t b);
uint16_t mgc_fp16_sub(uint16_t a, uint16_t b);
uint16_t mgc_fp16_mul(uint16_t a, uint16_t b);
uint16_t mgc_fp16_div(uint16_t a, uint16_t b);
uint16_t mgc_fp16_min(uint16_t a, uint16_t b);
uint16_t mgc_fp16_max(uint16_t a, uint16_t b);
uint16_t mgc_fp16_round(uint16_t a, uint8_t sel);
bool     mgc_fp16_compare(uint16_t a, uint16_t b, uint8_t op);

// =============================================================================
// State Lifecycle
// =============================================================================

void mgc_init(mgc_gpu_state_t *state,
              mgc_mem_read_fn read_cb,
              mgc_mem_write_fn write_cb,
              void *user_ctx) {
    if (!state) return;
    memset(state, 0, sizeof(mgc_gpu_state_t));
    state->mem_read = read_cb;
    state->mem_write = write_cb;
    state->user_ctx = user_ctx;
    mgc_reset(state);
}

void mgc_reset(mgc_gpu_state_t *state) {
    if (!state) return;
    memset(state->regs, 0, sizeof(state->regs));
    state->cr = 0;
    memset(state->vregs, 0, sizeof(state->vregs));
    state->pc = 0;
    state->pc_prev = 0;
    state->imm16_l = 0;
    state->pending_reg_bank = 0;
    state->stop_core = false;
    state->irq = false;
    state->instruction_count = 0;
    state->cycle_count = 0;
}

// =============================================================================
// Helper: Register Index Resolution with One-Shot Regbank Prefix
// =============================================================================
static inline uint8_t resolve_reg(uint8_t base_5bit, uint8_t hi_bit) {
    return (uint8_t)((hi_bit ? 32 : 0) | (base_5bit & 0x1F));
}

// =============================================================================
// Core Execution Step
// =============================================================================

bool mgc_step(mgc_gpu_state_t *state) {
    if (!state || state->stop_core) return false;

    // 1. Fetch instruction word (PC is word address, byte address is PC * 4)
    uint32_t inst = 0;
    if (state->mem_read) {
        inst = state->mem_read(state->user_ctx, state->pc * 4);
    }

    uint32_t current_pc = state->pc;
    state->pc_prev = current_pc;
    state->pc = current_pc + 1; // Default next sequential PC
    state->instruction_count++;
    state->cycle_count++;

    // 2. Decode condition execution
    bool cond_en  = MGC_GET_COND_EN(inst);
    bool cond_inv = MGC_GET_COND_INV(inst);
    uint8_t cr_idx = MGC_GET_COND_CR(inst);
    bool cr_val   = (state->cr >> cr_idx) & 0x01;

    if (cond_en && ((cond_inv ^ cr_val) == 0)) {
        // Condition evaluated to false: skip instruction execution
        if (state->trace_enabled) {
            printf("[TRACE] Cycle %llu | PC %08X | (SKIPPED by CR%d)\n",
                   (unsigned long long)state->cycle_count, current_pc, cr_idx);
        }
        state->pending_reg_bank = 0; // Clear any pending regbank prefix
        return true;
    }

    // 3. Decode instruction components
    uint8_t opcode   = MGC_GET_OPCODE(inst);
    uint8_t subop    = MGC_GET_SUBOP(inst);
    bool    srcA_sel = MGC_GET_SRCA_SEL(inst);
    uint8_t srcA_5   = MGC_GET_SRCA(inst);
    uint8_t srcB_5   = MGC_GET_SRCB(inst);
    uint8_t dst_5    = MGC_GET_DST(inst);

    // Apply high register bank prefix bits: [2]=srcA, [1]=srcB, [0]=dst
    uint8_t regA = resolve_reg(srcA_5, (state->pending_reg_bank >> 2) & 1);
    uint8_t regB = resolve_reg(srcB_5, (state->pending_reg_bank >> 1) & 1);
    uint8_t regD = resolve_reg(dst_5,  (state->pending_reg_bank >> 0) & 1);

    // One-shot regbank prefix clears after instruction consumes it
    bool is_regbank_prefix = (opcode == MGC_OP_NOP) && (subop == 1);
    if (!is_regbank_prefix) {
        state->pending_reg_bank = 0;
    }

    // Resolve Source A operand
    uint32_t valA = 0;
    if (srcA_sel) {
        valA = state->regs[regA];
    } else {
        valA = MGC_K_BANK[srcA_5 & 0x07]; // Zero-extended in upper 16 bits
    }
    uint32_t valB = state->regs[regB];

    uint16_t valA_fp16 = (uint16_t)(valA & 0xFFFF);
    uint16_t valB_fp16 = (uint16_t)(valB & 0xFFFF);

    // 4. Execute Opcode Channel
    switch (opcode) {
        case MGC_OP_NOP: // Channel 0
            if (subop == 1) { // regbank prefix
                // instruction[20] = srcA_hi, instruction[9] = srcB_hi, instruction[4] = dst_hi
                uint8_t a_hi = (inst >> 20) & 1;
                uint8_t b_hi = (inst >> 9)  & 1;
                uint8_t d_hi = (inst >> 4)  & 1;
                state->pending_reg_bank = (a_hi << 2) | (b_hi << 1) | d_hi;
            }
            break;

        case MGC_OP_PRELOAD_IMM: { // Channel 1: ploadu_l / movi_u16
            uint16_t imm16 = MGC_GET_IMM16(inst);
            bool is_write  = MGC_GET_IMM16_FLAG(inst);
            if (!is_write) {
                state->imm16_l = imm16;
            } else {
                state->regs[regD] = ((uint32_t)imm16 << 16) | state->imm16_l;
            }
            break;
        }

        case MGC_OP_COPY_SIGN: { // Channel 2: copy, fabs16, fneg16, fnabs16
            uint8_t mode = (inst >> 10) & 0x03;
            switch (mode) {
                case 0: state->regs[regD] = valA; break; // generic copy
                case 1: state->regs[regD] = valA_fp16 & 0x7FFF; break; // fabs16
                case 2: state->regs[regD] = valA_fp16 ^ 0x8000; break; // fneg16
                case 3: state->regs[regD] = valA_fp16 | 0x8000; break; // fnabs16
            }
            break;
        }

        case MGC_OP_FMULP2: { // Channel 3: fmulp2_16 (x * 2^p)
            uint16_t sign = valA_fp16 & 0x8000;
            uint16_t exp = (valA_fp16 >> 10) & 0x1F;
            uint16_t frac = valA_fp16 & 0x03FF;
            int8_t p = (int8_t)((inst >> 5) & 0x1F);
            if (p & 0x10) p |= (int8_t)0xE0; // Sign-extend 5-bit signed integer [-16..+15]
            uint16_t res = 0;
            if (exp == 0x1F) { // Inf or NaN
                res = valA_fp16;
            } else if (exp == 0) { // Zero or subnormal
                res = sign;
            } else {
                int exp_adj = (int)exp + (int)p;
                if (exp_adj >= 31) {
                    res = sign | 0x7C00; // Infinity
                } else if (exp_adj <= 0) {
                    res = sign; // Underflow flush to zero
                } else {
                    res = sign | ((exp_adj & 0x1F) << 10) | frac;
                }
            }
            state->regs[regD] = (uint32_t)res;
            break;
        }

        case MGC_OP_FADD_FSUB: { // Channel 4: fadd16 / fsub16
            bool is_sub = (inst >> 10) & 1;
            uint16_t res = is_sub ? mgc_fp16_sub(valA_fp16, valB_fp16)
                                  : mgc_fp16_add(valA_fp16, valB_fp16);
            state->regs[regD] = (uint32_t)res;
            break;
        }

        case MGC_OP_FMUL: { // Channel 5: fmul16
            uint16_t res = mgc_fp16_mul(valA_fp16, valB_fp16);
            state->regs[regD] = (uint32_t)res;
            break;
        }

        case MGC_OP_FDIV: { // Channel 6: fdiv16
            uint16_t res = mgc_fp16_div(valA_fp16, valB_fp16);
            state->regs[regD] = (uint32_t)res;
            break;
        }

        case MGC_OP_FMIN_FMAX: { // Channel 7: FP16 Sub-Op Execution (fmin16, fmax16, fsqrt16, frsqrt16)
            uint16_t res = 0;
            switch (subop) {
                case 0: res = mgc_fp16_min(valA_fp16, valB_fp16); break;   // fmin16
                case 1: res = mgc_fp16_max(valA_fp16, valB_fp16); break;   // fmax16
                case 2: res = mgc_fp16_sqrt(valA_fp16); break;             // fsqrt16 (ISA v1.1)
                case 3: res = mgc_fp16_rsqrt(valA_fp16); break;            // frsqrt16 (ISA v1.1)
                default: res = mgc_fp16_min(valA_fp16, valB_fp16); break;
            }
            state->regs[regD] = (uint32_t)res;
            break;
        }

        case MGC_OP_ROUND_TRUNC: { // Channel 8: trunc16 / round16 / ceil16 / floor16
            uint16_t res = mgc_fp16_round(valA_fp16, (inst >> 10) & 0x03);
            state->regs[regD] = (uint32_t)res;
            break;
        }

        case MGC_OP_F16_TO_U8: { // Channel 9: f16tou8 (RGBA Color Packing)
            uint8_t byte_idx = (inst >> 10) & 0x03;
            uint32_t bg = valB; // background 32-bit register (srcB)
            float f = mgc_fp16_to_float(valA_fp16);
            uint8_t u8_val = 0;
            if (f <= 0.0f) {
                u8_val = 0;
            } else if (f >= 255.0f) {
                u8_val = 255;
            } else {
                u8_val = (uint8_t)((int32_t)f);
            }
            uint32_t mask = 0xFFU << (byte_idx * 8);
            state->regs[regD] = (bg & ~mask) | ((uint32_t)u8_val << (byte_idx * 8));
            break;
        }

        case MGC_OP_F16_TO_U16: { // Channel 10: f16tou16 (Halfword Packing)
            uint8_t hw_idx = (inst >> 10) & 0x01;
            uint32_t bg = valB;
            float f = mgc_fp16_to_float(valA_fp16);
            uint16_t u16_val = 0;
            if (f <= 0.0f) {
                u16_val = 0;
            } else if (f >= 65535.0f) {
                u16_val = 65535;
            } else {
                u16_val = (uint16_t)((int32_t)f);
            }
            if (hw_idx == 0) {
                state->regs[regD] = (bg & 0xFFFF0000U) | (uint32_t)u16_val;
            } else {
                state->regs[regD] = (bg & 0x0000FFFFU) | ((uint32_t)u16_val << 16);
            }
            break;
        }

        case MGC_OP_F16_TO_I32: { // Channel 11: f16toi32 (Unsigned / Clamped 32-bit int)
            float f = mgc_fp16_to_float(valA_fp16);
            state->regs[regD] = (f <= 0.0f) ? 0 : (uint32_t)((int64_t)f);
            break;
        }

        case MGC_OP_U8_TO_F16: { // Channel 12: u8tof16 (Vertex Attribute Byte Unpack)
            uint8_t byte_idx = (inst >> 10) & 0x03;
            uint8_t u8_val = (uint8_t)((valB >> (byte_idx * 8)) & 0xFF);
            uint16_t res = mgc_float_to_fp16((float)u8_val);
            state->regs[regD] = (uint32_t)res;
            break;
        }

        case MGC_OP_U16_TO_F16: { // Channel 13: u16tof16 (Vertex Attribute Halfword Unpack)
            uint8_t hw_idx = (inst >> 10) & 0x01;
            uint16_t u16_val = (uint16_t)((valB >> (hw_idx * 16)) & 0xFFFF);
            uint16_t res = mgc_float_to_fp16((float)u16_val);
            state->regs[regD] = (uint32_t)res;
            break;
        }

        case MGC_OP_I32_TO_F16: { // Channel 14: i32tof16 (32-bit int to FP16)
            uint32_t u = valB;
            state->regs[regD] = (uint32_t)mgc_float_to_fp16((float)u);
            break;
        }

        case MGC_OP_FCOMP: { // Channel 15: fcomp to CR
            uint8_t cmp_code = (inst >> 10) & 0x07;
            uint8_t cr_target = inst & 0x07;
            bool cmp_res = mgc_fp16_compare(valA_fp16, valB_fp16, cmp_code);
            if (cmp_res) {
                state->cr |= (1 << cr_target);
            } else {
                state->cr &= ~(1 << cr_target);
            }
            break;
        }

        case MGC_OP_INT_VEC: { // Channel 16: integer add, and, istep4, vector helpers
            switch (subop) {
                case 0: state->regs[regD] = valA + valB; break; // add
                case 1: state->regs[regD] = valA & valB; break; // and
                case 2: state->regs[regD] = valA + (valB << 2); break; // istep4
                case 3: { // vset4 rA, rB, vD
                    uint8_t vD = inst & 0x07;
                    state->vregs[vD][0] = valA;
                    state->vregs[vD][1] = valA + valB;
                    state->vregs[vD][2] = valA + (valB * 2);
                    state->vregs[vD][3] = valA + (valB * 3);
                    break;
                }
                case 4: { // vbcast rA, vD
                    uint8_t vD = inst & 0x07;
                    state->vregs[vD][0] = valA;
                    state->vregs[vD][1] = valA;
                    state->vregs[vD][2] = valA;
                    state->vregs[vD][3] = valA;
                    break;
                }
                case 5: { // vadd_i32 vA, vB, vD
                    uint8_t vA = (inst >> 16) & 0x07;
                    uint8_t vB = (inst >> 5)  & 0x07;
                    uint8_t vD = inst & 0x07;
                    for (int l = 0; l < 4; l++) {
                        state->vregs[vD][l] = state->vregs[vA][l] + state->vregs[vB][l];
                    }
                    break;
                }
                default: break;
            }
            break;
        }

        case MGC_OP_LOAD: { // Channel 19: load rAddr, rD
            if (state->mem_read) {
                state->regs[regD] = state->mem_read(state->user_ctx, valB);
            }
            break;
        }

        case MGC_OP_STORE: { // Channel 20: store rSrc, rAddr
            if (state->mem_write) {
                state->mem_write(state->user_ctx, valB, valA);
            }
            break;
        }

        case MGC_OP_JUMP_RCALL: { // Channel 21: jump (18-bit signed PC-relative offset)
            int32_t offset = (int32_t)(((inst >> 14) & 0xFF) << 10) | (inst & 0x3FF);
            // Sign-extend 18-bit offset
            if (offset & (1 << 17)) {
                offset |= 0xFFFC0000;
            }
            state->pc = (uint32_t)((int32_t)state->pc_prev + offset);
            break;
        }

        case MGC_OP_GOTO_CALL: { // Channel 22: goto (register-indirect)
            state->pc = valB;
            break;
        }

        case MGC_OP_RET: { // Channel 23: ret (pop return address from stack)
            if (state->mem_read) {
                state->pc = state->mem_read(state->user_ctx, valB);
            }
            break;
        }

        case MGC_OP_STOP_CORE: { // Channel 24: stop_core
            state->stop_core = true;
            state->irq = true; // Hardware IRQ fix verification
            break;
        }

        case MGC_OP_ICMP: { // Channel 26: signed integer compare to CR
            int32_t ia = (int32_t)valA;
            int32_t ib = (int32_t)valB;
            uint8_t cr_dest = inst & 0x07;
            bool cmp = false;
            switch (subop) {
                case 0: cmp = (ia <  ib); break; // icmp_lt
                case 1: cmp = (ia <= ib); break; // icmp_le
                case 2: cmp = (ia == ib); break; // icmp_eq
                case 3: cmp = (ia >= ib); break; // icmp_ge
                case 4: cmp = (ia >  ib); break; // icmp_gt
                case 5: cmp = (ia != ib); break; // icmp_ne
                default: break;
            }
            if (cmp) state->cr |= (1 << cr_dest);
            else     state->cr &= ~(1 << cr_dest);
            break;
        }

        default:
            // Unimplemented / pass-through opcode
            break;
    }

    if (state->trace_enabled) {
        printf("[TRACE] Cycle %llu | PC %08X | Opcode %2d | CR=%02X | r%d=%08X\n",
               (unsigned long long)state->cycle_count, current_pc, opcode, state->cr, regD, state->regs[regD]);
    }

    return !state->stop_core;
}

uint64_t mgc_run(mgc_gpu_state_t *state, uint64_t max_instructions) {
    if (!state) return 0;
    uint64_t executed = 0;
    while (!state->stop_core && (max_instructions == 0 || executed < max_instructions)) {
        if (!mgc_step(state)) break;
        executed++;
    }
    return executed;
}

void mgc_dump_state(const mgc_gpu_state_t *state, FILE *out) {
    if (!state || !out) return;
    fprintf(out, "=== MGC-FP16 Architectural State Dump ===\n");
    fprintf(out, "PC: %08X | Stop: %d | IRQ: %d | CR: %02X | Cycles: %llu\n",
            state->pc, state->stop_core, state->irq, state->cr, (unsigned long long)state->cycle_count);
    fprintf(out, "--- Scalar Registers (r0 - r31) ---\n");
    for (int i = 0; i < 32; i += 4) {
        fprintf(out, "r%02d: %08X  r%02d: %08X  r%02d: %08X  r%02d: %08X\n",
                i, state->regs[i], i+1, state->regs[i+1],
                i+2, state->regs[i+2], i+3, state->regs[i+3]);
    }
    fprintf(out, "--- Upper Registers (r32 - r63) ---\n");
    for (int i = 32; i < 64; i += 4) {
        fprintf(out, "r%02d: %08X  r%02d: %08X  r%02d: %08X  r%02d: %08X\n",
                i, state->regs[i], i+1, state->regs[i+1],
                i+2, state->regs[i+2], i+3, state->regs[i+3]);
    }
    fprintf(out, "========================================\n");
}
