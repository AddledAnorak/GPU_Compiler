"""Flat address-space model for mgcsim.

No memory map is documented anywhere in FP16_ISA_V1.pdf (see
docs/isa/OPEN_QUESTIONS.md item 8). Every base address below is a
placeholder chosen only so the emulator has *a* consistent address space
to run against -- change these constants (only these constants) once the
real memory map is known.
"""

from __future__ import annotations

DATA_BASE = 0x0000_0000
DATA_SIZE = 0x0010_0000  # 1 MiB

FRAMEBUFFER_BASE = 0x0010_0000
FRAMEBUFFER_WIDTH = 256
FRAMEBUFFER_HEIGHT = 256
FRAMEBUFFER_STRIDE = FRAMEBUFFER_WIDTH  # words per row; pixel format TBD (item 8)
FRAMEBUFFER_SIZE = FRAMEBUFFER_STRIDE * FRAMEBUFFER_HEIGHT

STACK_BASE = 0x0020_0000
STACK_SIZE = 0x0001_0000


class MemoryError_(Exception):
    pass


class Memory:
    """Word-addressed (4-byte granularity assumed), sparse, sized regions."""

    def __init__(self):
        self._data: dict[int, int] = {}
        self._framebuffer: dict[int, int] = {}
        self._stack: dict[int, int] = {}

    def _region_for(self, addr: int):
        if DATA_BASE <= addr < DATA_BASE + DATA_SIZE:
            return self._data, addr - DATA_BASE
        if FRAMEBUFFER_BASE <= addr < FRAMEBUFFER_BASE + FRAMEBUFFER_SIZE:
            return self._framebuffer, addr - FRAMEBUFFER_BASE
        if STACK_BASE <= addr < STACK_BASE + STACK_SIZE:
            return self._stack, addr - STACK_BASE
        raise MemoryError_(f"address {addr:#010x} is outside all modeled regions")

    def load(self, addr: int) -> int:
        region, off = self._region_for(addr)
        return region.get(off, 0)

    def store(self, addr: int, value: int) -> None:
        region, off = self._region_for(addr)
        region[off] = value & 0xFFFFFFFF

    def framebuffer_addr(self, x: int, y: int) -> int:
        return FRAMEBUFFER_BASE + y * FRAMEBUFFER_STRIDE + x

    def dump_framebuffer(self) -> dict[int, int]:
        return dict(self._framebuffer)
