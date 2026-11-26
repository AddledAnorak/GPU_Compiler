#ifndef MGC_FP16_ISA_H
#define MGC_FP16_ISA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MGC-FP16-v1 Opcode Channels (instruction[31:27])
// =============================================================================
#define MGC_OP_NOP                 0
#define MGC_OP_PRELOAD_IMM         1
#define MGC_OP_COPY_SIGN           2
#define MGC_OP_FMULP2              3
#define MGC_OP_FADD_FSUB           4
#define MGC_OP_FMUL                5
#define MGC_OP_FDIV                6
#define MGC_OP_FMIN_FMAX           7
#define MGC_OP_ROUND_TRUNC         8
#define MGC_OP_F16_TO_U8           9
#define MGC_OP_F16_TO_U16          10
#define MGC_OP_F16_TO_I32          11
#define MGC_OP_U8_TO_F16           12
#define MGC_OP_U16_TO_F16          13
#define MGC_OP_I32_TO_F16          14
#define MGC_OP_FCOMP               15
#define MGC_OP_INT_VEC             16
#define MGC_OP_LOAD_ADDR           17
#define MGC_OP_STORE_ADDR          18
#define MGC_OP_LOAD                19
#define MGC_OP_STORE               20
#define MGC_OP_JUMP_RCALL          21
#define MGC_OP_GOTO_CALL           22
#define MGC_OP_RET                 23
#define MGC_OP_STOP_CORE           24
#define MGC_OP_V4CR_STORE          25
#define MGC_OP_ICMP                26
#define MGC_OP_V4CMP_GE_ZERO       27
#define MGC_OP_V4CMP_AND           28
#define MGC_OP_V4MASK_TO_CR        29
#define MGC_OP_V4CMP_STEP          30
#define MGC_OP_V4CMP_STEP_AND      31

// =============================================================================
// Instruction Field Extraction Macros
// =============================================================================
#define MGC_GET_OPCODE(inst)       (((inst) >> 27) & 0x1F)
#define MGC_GET_COND_EN(inst)      (((inst) >> 26) & 0x01)
#define MGC_GET_COND_INV(inst)     (((inst) >> 25) & 0x01)
#define MGC_GET_COND_CR(inst)      (((inst) >> 22) & 0x07)
#define MGC_GET_SRCA_SEL(inst)     (((inst) >> 21) & 0x01)
#define MGC_GET_SRCA(inst)         (((inst) >> 16) & 0x1F)
#define MGC_GET_SUBOP(inst)        (((inst) >> 10) & 0x3F)
#define MGC_GET_SRCB(inst)         (((inst) >> 5)  & 0x1F)
#define MGC_GET_DST(inst)          (((inst) >> 0)  & 0x1F)

// Immediate & Branch Field Extraction
#define MGC_GET_IMM16(inst)        (((inst) >> 6)  & 0xFFFF)
#define MGC_GET_IMM16_FLAG(inst)   (((inst) >> 5)  & 0x01) // 1=commit write, 0=preload low

// =============================================================================
// FP16 Constant Bank (Indexed by srcA[2:0] when srcA_sel == 0)
// =============================================================================
static const uint16_t MGC_K_BANK[8] = {
    0xFC00, // k[0]: -Infinity
    0xBC00, // k[1]: -1.0
    0x0000, // k[2]:  0.0
    0x3C00, // k[3]:  1.0
    0x3800, // k[4]:  0.5
    0x4248, // k[5]:  Pi (~3.1416)
    0x7C00, // k[6]: +Infinity
    0x7E00  // k[7]: Quiet NaN
};

#ifdef __cplusplus
}
#endif

#endif // MGC_FP16_ISA_H
