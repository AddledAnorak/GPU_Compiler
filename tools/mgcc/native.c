#include <stdint.h>
#include <stdlib.h>
#include "mgc_interpreter.h"

struct mgcc_buffer {
    uint32_t address;
    uint32_t size;
    unsigned char *data;
};

struct mgcc_execution {
    mgc_gpu_state_t state;
    const uint32_t *program;
    uint32_t words;
    const struct mgcc_buffer *buffers;
    uint32_t count;
    uint64_t budget;
    int fault;
    uint32_t fault_address;
};

static unsigned char *locate(struct mgcc_execution *execution, uint32_t address) {
    if (!(address & 3u)) {
        for (uint32_t index = 0; index < execution->count; index++) {
            const struct mgcc_buffer *buffer = &execution->buffers[index];
            if (address >= buffer->address &&
                (uint64_t)address + 4 <= (uint64_t)buffer->address + buffer->size)
                return buffer->data + (address - buffer->address);
        }
    }
    execution->fault = -1;
    execution->fault_address = address;
    return NULL;
}

static uint32_t read_word(void *context, uint32_t address) {
    struct mgcc_execution *execution = context;
    if (!(address & 3u) && address / 4 < execution->words)
        return execution->program[address / 4];
    unsigned char *data = locate(execution, address);
    if (!data)
        return 0;
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static void write_word(void *context, uint32_t address, uint32_t value) {
    struct mgcc_execution *execution = context;
    if (address / 4 < execution->words) {
        execution->fault = -1;
        execution->fault_address = address;
        return;
    }
    unsigned char *data = locate(execution, address);
    if (!data)
        return;
    data[0] = value;
    data[1] = value >> 8;
    data[2] = value >> 16;
    data[3] = value >> 24;
}

void *mgcc_create(const uint32_t *program, uint32_t words,
                  const struct mgcc_buffer *buffers, uint32_t count,
                  uint32_t entry, uint64_t budget) {
    if (!program || !words || words > 65536u || !budget || entry >= words || count > 4096u)
        return NULL;
    struct mgcc_execution *execution = calloc(1, sizeof(*execution));
    if (!execution)
        return NULL;
    execution->program = program;
    execution->words = words;
    execution->buffers = buffers;
    execution->count = count;
    execution->budget = budget;
    mgc_init(&execution->state, read_word, write_word, execution);
    execution->state.pc = entry;
    return execution;
}

int mgcc_poll(void *handle, uint32_t quantum) {
    struct mgcc_execution *execution = handle;
    if (!execution || !quantum || quantum > 4096u)
        return -5;
    for (uint32_t iteration = 0; iteration < quantum; iteration++) {
        if (execution->fault)
            return execution->fault;
        if (execution->state.stop_core)
            return execution->state.irq ? 1 : -5;
        if (execution->state.pc >= execution->words)
            return -2;
        if (execution->state.instruction_count >= execution->budget)
            return -3;
        uint32_t word = execution->program[execution->state.pc];
        unsigned channel = word >> 27;
        bool enabled = MGC_GET_COND_EN(word);
        bool predicate = (execution->state.cr >> MGC_GET_COND_CR(word)) & 1u;
        bool active = !enabled || (predicate ^ MGC_GET_COND_INV(word));
        if (active && (channel == 9 || channel == 10 || channel == 11)) {
            unsigned source = MGC_GET_SRCA(word) | ((execution->state.pending_reg_bank & 4u) ? 32u : 0u);
            uint16_t half = MGC_GET_SRCA_SEL(word) ? execution->state.regs[source] : MGC_K_BANK[source & 7u];
            bool exceptional = (half & 0x7c00u) == 0x7c00u;
            if (exceptional && (channel == 11 || (half & 0x3ffu)))
                return -4;
        }
        mgc_step(&execution->state);
        if (execution->fault)
            return execution->fault;
        if (execution->state.stop_core)
            return execution->state.irq ? 1 : -5;
    }
    return 0;
}

uint64_t mgcc_steps(void *handle) {
    return ((struct mgcc_execution *)handle)->state.instruction_count;
}

uint32_t mgcc_register(void *handle, uint32_t index) {
    return index < 64 ? ((struct mgcc_execution *)handle)->state.regs[index] : 0;
}

uint32_t mgcc_fault_address(void *handle) {
    return ((struct mgcc_execution *)handle)->fault_address;
}

void mgcc_destroy(void *handle) {
    free(handle);
}
