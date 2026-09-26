"""mgcasm: table-driven assembler/disassembler for MGC-FP16-v1.

Everything this package knows about the instruction set comes from
docs/isa/isa.yaml. See docs/isa/OPEN_QUESTIONS.md and ENCODING_GUESSES.md
for what is genuine vendor spec vs. provisional guess.
"""

from .assembler import Assembler, AsmError, assemble
from .disassembler import Disassembler, DisasmError, disassemble
from .isa import ISA

__all__ = [
    "ISA",
    "Assembler",
    "AsmError",
    "assemble",
    "Disassembler",
    "DisasmError",
    "disassemble",
]
