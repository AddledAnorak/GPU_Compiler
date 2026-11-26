import json
from pathlib import Path

import pytest

from tools.mgcemu import AssemblyError, Emulator, Memory, assemble, decode_word
from tools.mgcemu.__main__ import main, read_mem

EXAMPLES = Path(__file__).resolve().parents[1] / "hardware-examples/aakash-2026-09-28"


@pytest.mark.parametrize("name", ["bucky_12frame_dualbuf_clean", "utah_12frame_dualbuf_shader"])
@pytest.mark.usefixtures("local_hardware_examples")
def test_assemble_unchanged_paired_samples(name):
    source = (EXAMPLES / f"{name}.asm").read_text()
    assert assemble(source, sample_source=True) == read_mem(EXAMPLES / f"{name}.program.mem")


@pytest.mark.usefixtures("local_hardware_examples")
def test_assemble_gears_preserves_source_binary_disagreement():
    source = assemble((EXAMPLES / "mesa_gears_hd.asm").read_text(), sample_source=True)
    binary = read_mem(EXAMPLES / "mesa_gears_hd.program.mem")
    assert (len(source), len(binary)) == (597, 609)
    assert sum(left != right for left, right in zip(source, binary)) == 154


def test_branch_offsets_include_prefixes_and_macro_words():
    words = assemble("entry: movi_u16 1, r32\n[!cr1] jump_next entry\njump entry")
    assert words[:3] == [0x08000040, 0x00000410, 0x08000020]
    relative_next, relative_current = map(decode_word, words[3:])
    assert (relative_next.offset, relative_current.offset) == (-4, -4)
    assert (relative_next.condition, relative_next.invert) == (1, True)
    assert (relative_next.form.pc_bias, relative_current.form.pc_bias) == (1, 0)
    assert decode_word(assemble("jump -131072")[0]).offset == -131072
    assert decode_word(assemble("jump_next 131071")[0]).offset == 131071


def test_assembled_program_can_mix_scalar_and_sample_operations():
    words = assemble("""
        movi_u16 0x100, r32
        movi_u16 4, r35
        vset4 r32, r35, v0
        movi_i32 -1, r36
        movi_u16 1, r37
        vset4 r36, r37, v1
        vcmp_ge_zero_i32_mask v1, r34
        movi_f16 0.5, r32
        fadd16 #3, r32, r33
        wait_fadd
        jump test
        movi_u16 99, r33
    test: icmp_eq r33, r33, cr1
        [cr1] jump_next pixel
        movi_u16 99, r33
    pixel: vstore_mask_r r33, v0, r34
        stop_core
    """)
    memory = Memory()
    memory.map(0x100, 16)
    emulator = Emulator(words, memory=memory, allow=("sample-next-pc", "sample-vector"))
    assert emulator.run(max_steps=len(words)).reason == "stop_core"
    assert emulator.irq and emulator.registers[33:35] == [0x3E00, 0b1110]
    assert [memory.read(address) for address in range(0x100, 0x110, 4)] == [0, 0x3E00, 0x3E00, 0x3E00]


def test_pdf_wait_macros_are_explicit_nops():
    cases = {"wait_load": 4, "wait_fadd": 6, "wait_fcomp": 3, "wait_fmul": 8,
             "wait_fdiv": 8, "wait_conv": 3, "wait_short": 3, "wait 2": 2}
    for instruction, slots in cases.items():
        assert assemble(instruction) == [0] * slots
    assert assemble("stop_core") == [0xC0000000]


def test_immediate_aliases_register_banks_and_c_literal_quantization():
    assert assemble("ploadu_l 0x1234\nloadu_h 0xabcd, r5") == [0x08048D00, 0x082AF365]
    assert assemble("regbank 1, 1, 1\nfsub16 r1, r2, r3") == assemble("fsub16 r33, r34, r35")
    assert assemble("u32tof16 r33, r34") == assemble("i32tof16 r33, r34")
    emulator = Emulator(assemble("movi_f16 1.00146484375, r2\nstop_core"))
    emulator.run(max_steps=3)
    assert emulator.registers[2] == 0x3C01


@pytest.mark.parametrize(("source", "error"), [
    ("load r64, r1", "r register"),
    ("vfadd16 v0, v1, v8", "v register"),
    ("fadd16 #8, r2, r3", "constant-bank index"),
    ("icmp_eq r1, r2, cr8", "cr register"),
    ("fadd16 r1, r2", "expects 3 operand"),
    ("[cr8] nop", "expected a predicate"),
    ("jump nowhere", "undefined label"),
    ("jump end\nend:", "has no instruction"),
    ("loop: nop\nloop: nop", "duplicate label"),
    ("jump 131072", "branch displacement"),
    ("movi_f16 inf, r1", "finite half"),
    ("movi_f16 65536, r1", "finite half"),
    ("call r1", "named in the ISA PDF"),
    (".include \"arbitrary.inc\"", "includes are not read"),
    ("not_an_op", "unknown instruction"),
])
def test_assembly_reports_source_line(source, error):
    with pytest.raises(AssemblyError, match=f"line [23]: .*{error}"):
        assemble("// header\n" + source)


def test_sample_source_mode_does_not_read_arbitrary_includes():
    with pytest.raises(AssemblyError, match="includes are not read"):
        assemble('.include "anything.inc"\nnop', sample_source=True)
    assert assemble("jump 0") == [0xA8000000]
    assert assemble("jump 0", sample_source=True) == [0xA8001C00]


def test_cli_assembles_and_runs_same_words_in_either_format(tmp_path, capsys):
    source = tmp_path / "demo.asm"
    binary = tmp_path / "demo.program.mem"
    source.write_text("movi_u16 5, r1\nstop_core\n")
    assert main(["assemble", str(source), "-o", str(binary)]) == 0
    assert binary.read_text() == "08000140\n08000021\nc0000000\n"
    assert main([str(source), "--max-steps", "3"]) == 0
    direct = json.loads(capsys.readouterr().out)
    assert main([str(binary), "--max-steps", "3"]) == 0
    assert direct == json.loads(capsys.readouterr().out)
    assert direct["irq"] and direct["registers"][1] == 5
    assert main(["assemble", str(source), "-o", str(binary)]) == 1
    assert binary.read_text() == "08000140\n08000021\nc0000000\n"
    invalid = tmp_path / "invalid.program.mem"
    source.write_text("call r1\n")
    assert main(["assemble", str(source), "-o", str(invalid)]) == 1
    assert "line 1: call:" in capsys.readouterr().err
    assert not invalid.exists()
