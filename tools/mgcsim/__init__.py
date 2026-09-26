"""mgcsim: functional emulator for MGC-FP16-v1.

With no board and no RTL available (see docs/isa/OPEN_QUESTIONS.md), this
is the only way to test the assembler output and, later, the compiler
backend end to end. It models the 64x32-bit register file, CR0-CR7, the
FP16 constant bank, a flat memory space, and -- critically -- the latency
contract from Table 10, raising on scheduling violations instead of
silently tolerating them.
"""

from .cpu import CPU, LatencyViolation, MgcSimError

__all__ = ["CPU", "LatencyViolation", "MgcSimError"]
