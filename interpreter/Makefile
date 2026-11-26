# =============================================================================
# Makefile for Portable C ISA Interpreter (MGC-FP16-v1)
# Works on Linux, macOS, and WSL
# =============================================================================

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -Iinclude -std=c99
LDFLAGS ?= -lm

SRC = src/mgc_interpreter.c src/mgc_fp16_math.c
OBJ = $(SRC:.c=.o)
TEST_SRC = test/test_interpreter.c
TEST_BIN = test/test_interpreter

.PHONY: all clean test lib

all: $(TEST_BIN)

lib: libmgc_interpreter.a

libmgc_interpreter.a: $(OBJ)
	ar rcs $@ $^

$(TEST_BIN): $(OBJ) $(TEST_SRC)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f src/*.o test/*.o $(TEST_BIN) libmgc_interpreter.a test_interpreter.exe
