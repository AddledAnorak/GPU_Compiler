"""Evidence-backed encoding for Aakash’s recovered GPU examples."""

from .encoding import (
    HARDWARE_FORMAT, HardwareEncodingError, Instruction, assemble_hardware,
    decode_word, disassemble_hardware, load_mem, parse_source,
)
from .cpu import ByteMemory, HardwareCPU, MemoryAccessError, StepLimitExceeded

__all__ = [
    "HARDWARE_FORMAT", "HardwareEncodingError", "Instruction",
    "assemble_hardware", "decode_word", "disassemble_hardware", "load_mem",
    "parse_source",
    "ByteMemory", "HardwareCPU", "MemoryAccessError", "StepLimitExceeded",
]
