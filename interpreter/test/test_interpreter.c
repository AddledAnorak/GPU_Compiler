#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mgc_interpreter.h"

// 64KB simulated flat memory for test
#define SIM_MEM_SIZE (64 * 1024)
static uint8_t sim_memory[SIM_MEM_SIZE];

static uint32_t mem_read_cb(void *user_ctx, uint32_t byte_addr) {
    (void)user_ctx;
    if (byte_addr + 4 <= SIM_MEM_SIZE) {
        return *(uint32_t*)(&sim_memory[byte_addr]);
    }
    return 0;
}

static void mem_write_cb(void *user_ctx, uint32_t byte_addr, uint32_t data_word) {
    (void)user_ctx;
    if (byte_addr + 4 <= SIM_MEM_SIZE) {
        *(uint32_t*)(&sim_memory[byte_addr]) = data_word;
    }
}

static void write_instruction(uint32_t word_idx, uint32_t inst) {
    uint32_t byte_addr = word_idx * 4;
    assert(byte_addr + 4 <= SIM_MEM_SIZE);
    *(uint32_t*)(&sim_memory[byte_addr]) = inst;
}

int main(void) {
    printf("==================================================================\n");
    printf("  RUNNING MGC-FP16 PORTABLE C ISA INTERPRETER VERIFICATION SUITE  \n");
    printf("==================================================================\n");

    mgc_gpu_state_t gpu;
    mgc_init(&gpu, mem_read_cb, mem_write_cb, NULL);
    gpu.trace_enabled = true;

    // Build directed test program in memory:
    // Inst 0: Preload low 16 bits of immediate: ploadu_l 10
    // Opcode=1, imm16=10, flag=0 -> (1 << 27) | (10 << 6) | (0 << 5)
    write_instruction(0, (1 << 27) | (10 << 6) | (0 << 5));

    // Inst 1: movi_u16 0, r1 (Writes {0, 10} = 10 to r1)
    // Opcode=1, imm16=0, flag=1, dst=1 -> (1 << 27) | (0 << 6) | (1 << 5) | (1)
    write_instruction(1, (1 << 27) | (0 << 6) | (1 << 5) | (1));

    // Inst 2: Preload low 16 bits: ploadu_l 25
    write_instruction(2, (1 << 27) | (25 << 6) | (0 << 5));

    // Inst 3: movi_u16 0, r2 (Writes {0, 25} = 25 to r2)
    write_instruction(3, (1 << 27) | (0 << 6) | (1 << 5) | (2));

    // Inst 4: Integer add r1, r2, r3 -> r3 = 10 + 25 = 35
    // Opcode=16, srcA_sel=1, srcA=1, subop=0, srcB=2, dst=3
    write_instruction(4, (16 << 27) | (1 << 21) | (1 << 16) | (0 << 10) | (2 << 5) | (3));

    // Inst 5: regbank prefix: dst_hi=1 (Target r32 + 3 = r35)
    // Opcode=0, subop=1, dst_hi=inst[4] -> (0 << 27) | (1 << 10) | (1 << 4)
    write_instruction(5, (0 << 27) | (1 << 10) | (1 << 4));

    // Inst 6: Copy r3 into r35 (using upper register bank prefix from Inst 5)
    // Opcode=2, srcA_sel=1, srcA=3, mode=0, dst=3 (maps to physical r35)
    write_instruction(6, (2 << 27) | (1 << 21) | (3 << 16) | (0 << 10) | (3));

    // Inst 7: icmp_ge r2, r1, cr0 (25 >= 10 -> cr0 should be 1)
    // Opcode=26, srcA_sel=1, srcA=2, subop=3, srcB=1, cr_dest=0
    write_instruction(7, (26 << 27) | (1 << 21) | (2 << 16) | (3 << 10) | (1 << 5) | (0));

    // Inst 8: stop_core (c0000000)
    write_instruction(8, 0xC0000000);

    // Run interpreter
    printf("[SIM] Launching execution...\n");
    uint64_t total_ops = mgc_run(&gpu, 100);
    printf("[SIM] Execution halted after %llu instructions.\n\n", (unsigned long long)total_ops);

    mgc_dump_state(&gpu, stdout);

    // Assertions for correctness
    printf("------------------------------------------------------------------\n");
    printf("[CHECK] Verifying architectural state...\n");
    assert(gpu.regs[1] == 10);
    printf("  [PASS] r1 = %u (Expected 10)\n", gpu.regs[1]);

    assert(gpu.regs[2] == 25);
    printf("  [PASS] r2 = %u (Expected 25)\n", gpu.regs[2]);

    assert(gpu.regs[3] == 35);
    printf("  [PASS] r3 = %u (Expected 35)\n", gpu.regs[3]);

    assert(gpu.regs[35] == 35);
    printf("  [PASS] r35 (Upper register via regbank) = %u (Expected 35)\n", gpu.regs[35]);

    assert((gpu.cr & 0x01) == 0x01);
    printf("  [PASS] cr0 = 1 (Expected 1 for 25 >= 10)\n");

    assert(gpu.stop_core == true);
    printf("  [PASS] stop_core asserted.\n");

    assert(gpu.irq == true);
    printf("  [PASS] Hardware IRQ asserted on stop_core.\n");

    printf("==================================================================\n");
    printf(">>> ALL PORTABLE C ISA INTERPRETER TESTS PASSED SUCCESSFULLY! <<<\n");
    printf("==================================================================\n");

    return 0;
}
