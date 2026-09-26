"""End-to-end check that the mgc Gallium backend's output loop closes:
NIR -> MGC assembly text -> mgcasm -> mgcsim -> checked framebuffer value.

The .asm fixtures here are not hand-written -- they are the literal
MGC_ASM_DUMP output captured by running the actual mgc backend
(mesa/src/gallium/drivers/mgc/) against real inputs:

  tests/fixtures/00_arith_fs.asm  -- shaders/00_arith/{vert,frag}.glsl
                                      compiled through real GLSL -> Mesa
                                      -> NIR -> mgc_compiler.c, via
                                      tools/mgc_harness.c. GLSL constant-
                                      folded (2*3+0.5)/2 to a single 3.25
                                      literal before the backend ever saw
                                      it (see docs/glsl-to-nir/
                                      dictionary.md's 05_loop finding for
                                      why that's expected, not a bug).

  tests/fixtures/01_flat_color_fs.asm -- shaders/01_flat_color/frag.glsl
                                      through the same real pipeline,
                                      paired with shaders/00_arith/
                                      vert.glsl (01_flat_color's own
                                      vertex shader needs load_input,
                                      which is out of scope -- see below).

  tests/fixtures/arith_chain.asm  -- mesa/src/gallium/drivers/mgc/
                                      mgc_test_arith.c, a standalone
                                      NIR-builder-based unit test that
                                      bypasses GLSL entirely so constant
                                      folding can't collapse the chain --
                                      this is what actually exercises
                                      multi-instruction fmul16/fadd16/
                                      fdiv16 with wait_* insertion and
                                      register materialization/operand-
                                      swap for the constant bank. Same
                                      (2*3+0.5)/2 = 3.25 computation as
                                      00_arith, so the two fixtures are a
                                      cross-check of each other.

Regenerate a fixture (requires a local Mesa build with the mgc driver --
see the Weeks 1-4 plan, Phase 0):
    EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=mgc \\
    MESA_SHADER_CACHE_DISABLE=1 MGC_ASM_DUMP=<dir> \\
    LD_LIBRARY_PATH="mesa/build/src/egl:mesa/build/src/mapi/es2api:mesa/build/src/gallium/targets/dri" \\
        tools/mgc_harness shaders/<name>/vert.glsl shaders/<name>/frag.glsl

Note on 01_flat_color's own vertex shader: every vertex shader in the
shaders/ corpus except 00_arith's reads a_position via load_input, which
the Weeks 1-4 backend doesn't support (no attribute-fetch model exists in
the ISA yet -- Week 5-8). mgc_compiler.c treats vertex-stage store_output
as a documented no-op for exactly this reason (see its comment), so any
of those vertex shaders still fails to *link* on load_input specifically,
which is why 01_flat_color's fragment shader is paired with 00_arith's
vertex shader here rather than its own.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import numpy as np

from mgcasm import ISA, assemble, disassemble
from mgcasm.fp16 import float_to_fp16_bits
from mgcsim import CPU

isa = ISA()
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _run_fixture(name: str) -> CPU:
    src = (FIXTURES / name).read_text()
    words = assemble(src, isa)

    # (a) mgcasm accepts the backend's output
    lines = disassemble(words, isa)
    words2 = assemble("\n".join(lines), isa)
    assert words == words2, "round-trip mismatch"  # (b) round-trips

    cpu = CPU(isa=isa, max_cycles=1000)
    cpu.run(words)  # (c) no LatencyViolation -- validates the wait_* policy
    assert cpu.halted
    return cpu


def test_flat_color_fragment_shader_roundtrip():
    cpu = _run_fixture("01_flat_color_fs.asm")
    fb = cpu.mem
    assert fb.load(0x100000) & 0xFFFF == float_to_fp16_bits(1.0)  # (d) R
    assert fb.load(0x100001) & 0xFFFF == float_to_fp16_bits(0.0)  # G
    assert fb.load(0x100002) & 0xFFFF == float_to_fp16_bits(0.0)  # B
    assert fb.load(0x100003) & 0xFFFF == float_to_fp16_bits(1.0)  # A


def test_arith_fragment_shader_through_glsl():
    # Constant-folded by GLSL before reaching the backend -- see the
    # module docstring. Still real evidence the GLSL->NIR->mgc path
    # closes; test_arith_chain_bypassing_glsl below is what actually
    # exercises the multi-instruction path.
    cpu = _run_fixture("00_arith_fs.asm")
    expected = float_to_fp16_bits(np.float16(3.25))
    fb = cpu.mem
    assert fb.load(0x100000) & 0xFFFF == expected  # R
    assert fb.load(0x100001) & 0xFFFF == expected  # G
    assert fb.load(0x100002) & 0xFFFF == expected  # B
    assert fb.load(0x100003) & 0xFFFF == float_to_fp16_bits(1.0)  # A


def test_arith_chain_bypassing_glsl():
    src = (FIXTURES / "arith_chain.asm").read_text()
    # Confirms the actual instruction shape, not just the numeric result:
    # a real fmul16/fadd16/fdiv16 chain with wait_* between each op, and
    # the commutative-operand-swap that put the constant-bank #4 (0.5)
    # into fadd16's srcA slot rather than materializing a register for it.
    assert "fmul16" in src and "wait_fmul" in src
    assert "fadd16 #4," in src and "wait_fadd" in src
    assert "fdiv16" in src and "wait_fdiv" in src

    cpu = _run_fixture("arith_chain.asm")
    expected = float_to_fp16_bits(np.float16(3.25))
    assert cpu.mem.load(0x100000) & 0xFFFF == expected
