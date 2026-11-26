from pathlib import Path

import pytest

from tools.mgcasm.fp16 import float_to_fp16_bits
from tools.mgchw import (
    ByteMemory, HardwareCPU, HardwareEncodingError, MemoryAccessError,
    StepLimitExceeded, assemble_hardware, decode_word, disassemble_hardware,
    load_mem, parse_source,
)
from tools.mgchw.cli import DEFAULT_EXAMPLES, framebuffer_png, verify_examples
from tools.mgchw.cpu import UnsupportedHardwareSemantics


@pytest.mark.parametrize("scene", ["bucky_12frame_dualbuf_clean", "utah_12frame_dualbuf_shader"])
@pytest.mark.usefixtures("local_hardware_examples")
def test_supplied_source_reproduces_real_binary(scene):
    assert assemble_hardware((DEFAULT_EXAMPLES / f"{scene}.asm").read_text()) == load_mem(
        DEFAULT_EXAMPLES / f"{scene}.program.mem")


@pytest.mark.parametrize("path", sorted(DEFAULT_EXAMPLES.glob("*.program.mem")), ids=lambda path: path.stem)
def test_every_supplied_binary_roundtrips_without_provisional_encoding(path):
    words = load_mem(path)
    assert assemble_hardware("\n".join(disassemble_hardware(words))) == words


@pytest.mark.usefixtures("local_hardware_examples")
def test_gears_mismatch_is_visible_not_silently_repaired():
    report = verify_examples()
    gears = next(item for item in report["source_pairs"] if item["file"] == "mesa_gears_hd.asm")
    assert gears["source_words"] == 597
    assert gears["binary_words"] == 609
    assert gears["overlapping_word_mismatches"] == 154
    assert not gears["matches"]
    assert len(report["files"]) == 13


@pytest.mark.parametrize("offset", [-131072, -5000, -1, 0, 1000, 131071])
def test_split_signed_branch_encoding(offset):
    assert decode_word(assemble_hardware(f"jump {offset}")[0]).branch_offset == offset


@pytest.mark.parametrize("source", [
    "add r32, r1, r2", "regbank 1", "unknown r0", "movi_u16 -1, r1",
    "movi_i32 4294967296, r1", "vadd_i32 v8, v0, v0", "jump 131072",
    '.include "untrusted.inc"', "duplicate: nop\nduplicate: nop",
])
def test_unverified_or_invalid_hardware_assembly_is_rejected(source):
    with pytest.raises(HardwareEncodingError):
        assemble_hardware(source)


def test_immediate_integer_and_predicate_execution():
    cpu = HardwareCPU()
    cpu.run(assemble_hardware("""
        movi_i32 -1, r1
        movi_u16 5, r2
        add r1, r2, r3
        icmp_gt r3, r2, cr0
        [cr0] movi_u16 99, r4
        [!cr0] movi_u16 7, r4
    """))
    assert cpu.halted and cpu.registers[3] == 4 and cpu.registers[4] == 7


def test_byte_addressed_four_lane_masks_and_stores():
    memory = ByteMemory()
    memory.map_region(0, 16)
    cpu = HardwareCPU(memory=memory)
    cpu.run(assemble_hardware("""
        movi_i32 0, r0
        movi_u16 4, r1
        vset4 r0, r1, v0
        movi_i32 -1, r2
        movi_u16 1, r3
        vset4 r2, r3, v2
        vcmp_ge_zero_i32_mask v2, r4
        movi_u16 4660, r5
        vstore_mask_r r5, v0, r4
    """))
    assert [memory.read32(address) for address in range(0, 16, 4)] == [0, 4660, 4660, 4660]
    assert cpu.registers[4] == 14


def test_memory_alignment_bounds_and_read_only_regions():
    memory = ByteMemory()
    memory.map_region(0, 8)
    memory.map_region(16, 4, writable=False)
    for address in [1, 8, -4]:
        with pytest.raises(MemoryAccessError):
            memory.read32(address)
    with pytest.raises(MemoryAccessError):
        memory.write32(16, 1)
    with pytest.raises(MemoryAccessError):
        memory.map_region(4, 8)


def test_execution_budget_stops_an_animation_loop():
    with pytest.raises(StepLimitExceeded):
        HardwareCPU().run(assemble_hardware("forever: jump forever\nnop\nnop"), max_steps=10)


def test_host_only_skip_cycles_are_also_bounded():
    with pytest.raises(StepLimitExceeded):
        HardwareCPU().run(assemble_hardware("nop\nnop"), max_steps=10, skip_pcs={0: 1, 1: 0})


def test_approximate_rgb565_semantics_require_explicit_opt_in():
    words = assemble_hardware("movi_f16 1.0, r0\nvbcast r0, v0\nvrgb565_ceramic v0, v1")
    with pytest.raises(UnsupportedHardwareSemantics):
        HardwareCPU().run(words)
    cpu = HardwareCPU(allow_approximate_fp16=True)
    cpu.run(words)
    assert cpu.vectors[1] == [65535] * 4


def test_fp16_add_and_constant_bank():
    cpu = HardwareCPU()
    cpu.run(assemble_hardware("movi_f16 0.5, r0\nfadd16 #3, r0, r1\nwait_fadd"))
    assert cpu.registers[1] == float_to_fp16_bits(1.5)


def test_hex_image_validation_and_png(tmp_path: Path):
    path = tmp_path / "bad.mem"
    path.write_text("100000000\n")
    with pytest.raises(HardwareEncodingError):
        load_mem(path)
    memory = ByteMemory()
    memory.map_region(0, 4)
    memory.write32(0, 0xF800)
    assert framebuffer_png(memory, 0, 1, 1).startswith(b"\x89PNG\r\n\x1a\n")


def test_source_labels_include_real_macro_widths():
    source = "start: movi_i32 1, r1\nload r1, r2\nwait_load\nend: nop"
    assert parse_source(source)[1] == {"start": 0, "end": 7}


def test_compiler_artifact_comments_do_not_change_hardware_words():
    assert assemble_hardware("; compiler metadata\nmovi_i32 1, r0 ; literal\n") == assemble_hardware("movi_i32 1, r0")
