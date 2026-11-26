#include <assert.h>
#include <stdio.h>
#include "tools/mgcc/native.c"

static int run(const uint32_t *program, uint32_t words, uint64_t budget,
               struct mgcc_buffer *buffer) {
    void *execution = mgcc_create(program, words, buffer, 1, 0, budget);
    assert(execution);
    int result;
    do {
        result = mgcc_poll(execution, 16);
    } while (!result);
    mgcc_destroy(execution);
    return result;
}

int main(int count, char **arguments) {
    assert(count == 2);
    FILE *file = fopen(arguments[1], "rb");
    assert(file);
    assert(!fseek(file, 0, SEEK_END));
    long bytes = ftell(file);
    assert(bytes > 0 && bytes <= 65536 * 4 && !(bytes & 3));
    assert(!fseek(file, 0, SEEK_SET));
    uint32_t *program = malloc(bytes);
    assert(program && fread(program, 1, bytes, file) == (size_t)bytes);
    fclose(file);
    unsigned char output[16] = {0};
    struct mgcc_buffer buffer = {0x400000u, sizeof(output), output};
    assert(run(program, bytes / 4, 1000, &buffer) == 1);
    assert(output[0] == 0 && output[1] == 0x3c && output[12] == 0 && output[13] == 0x3c);
    free(program);
    uint32_t stop[] = {24u << 27};
    uint32_t loop[] = {21u << 27};
    uint32_t escape[] = {0};
    uint32_t bad_store[] = {20u << 27, 24u << 27};
    uint32_t bad_load[] = {(1u << 27) | (1u << 6), (1u << 27) | 32u | 1u,
                           (19u << 27) | (1u << 5), 24u << 27};
    assert(run(stop, 1, 1, &buffer) == 1);
    assert(run(loop, 1, 10, &buffer) == -3);
    assert(run(escape, 1, 10, &buffer) == -2);
    assert(run(bad_store, 2, 10, &buffer) == -1);
    assert(run(bad_load, 4, 10, &buffer) == -1);
    puts("five guard checks and fresh flat GLSL: pass");
    return 0;
}
