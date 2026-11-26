"""Fresh scalar-add compilation and functional execution against IEEE binary16."""

import json
import math
import os
import struct
from pathlib import Path

import pytest

import tools.mgchw as hardware_module
from tools.host_render import ShaderExecutor, pack_constants
from tools.mgcasm import assemble
from tools.mgcdriver import (
    Driver,
    HardwareSimulationBackend,
    Program,
    ProgramFormat,
    ProgramFormatError,
    SimulationBackend,
)
from tools.mgchw import (
    ByteMemory,
    HardwareCPU,
    assemble_hardware,
    decode_word,
    disassemble_hardware,
    load_mem,
    parse_source,
)
from tools.run_shader import ROOT, compile_glsl, execute_assembly, execute_shader_pair
from tools.shader_layout import ShaderLayout, link_varyings


HARDWARE = ProgramFormat.HARDWARE
LEGACY = ProgramFormat.LEGACY
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
UTAH = ROOT / "hardware-examples/aakash-2026-09-28/utah_12frame_dualbuf_shader"
GOLDEN_ADDS = (
    (340, 0x20200003, (0, 0, 3)),
    (347, 0x20230063, (3, 3, 3)),
    (422, 0x20220002, (2, 0, 2)),
    (432, 0x20240024, (4, 1, 4)),
    (453, 0x20220062, (2, 3, 2)),
)
ADD_CASES = (
    pytest.param(0x3C00, 0x3800, 0x3E00, id="positive"),
    pytest.param(0xBC00, 0xB800, 0xBE00, id="negative"),
    pytest.param(0x3A00, 0xB400, 0x3800, id="mixed-sign"),
    pytest.param(0x3C00, 0xBC00, 0x0000, id="exact-cancellation"),
    pytest.param(0x0000, 0x0000, 0x0000, id="positive-zeros"),
    pytest.param(0x8000, 0x8000, 0x8000, id="negative-zeros"),
    pytest.param(0x8000, 0x0000, 0x0000, id="opposite-zeros"),
    pytest.param(0x3C00, 0x8000, 0x3C00, id="negative-zero-identity"),
    pytest.param(0x0001, 0x0001, 0x0002, id="positive-subnormals"),
    pytest.param(0x8001, 0x8001, 0x8002, id="negative-subnormals"),
    pytest.param(0x03FF, 0x0001, 0x0400, id="subnormal-to-normal"),
    pytest.param(0x0400, 0x8001, 0x03FF, id="normal-to-subnormal"),
    pytest.param(0x0001, 0x8001, 0x0000, id="subnormal-cancellation"),
    pytest.param(0x3C00, 0x1000, 0x3C00, id="halfway-even-down"),
    pytest.param(0x3C01, 0x1000, 0x3C02, id="halfway-odd-up"),
    pytest.param(0xBC00, 0x9000, 0xBC00, id="negative-halfway-even"),
    pytest.param(0xBC01, 0x9000, 0xBC02, id="negative-halfway-odd"),
    pytest.param(0x3C00, 0x1001, 0x3C01, id="above-halfway"),
    pytest.param(0x4000, 0x1000, 0x4000, id="below-halfway"),
    pytest.param(0x7BFF, 0x0001, 0x7BFF, id="maximum-finite-plus-tiny"),
    pytest.param(0x7BFF, 0x7BFF, 0x7C00, id="positive-result-overflow"),
    pytest.param(0xFBFF, 0xFBFF, 0xFC00, id="negative-result-overflow"),
)
CONSTANTS = (1.0, 1.0, -0.5, 0.3)
TINY_CONSTANTS = (2.0**-24, 2.0**-14, 0.125, -1.0)
VERTEX_SOURCE = """#version 100
attribute vec4 a_position;
attribute vec4 a_color;
varying vec4 v_operands;
void main() {
    gl_Position = a_position;
    v_operands = a_color;
}
"""
UNIFORM_VERTEX_SOURCE = """#version 100
attribute vec4 a_position;
attribute vec4 a_color;
uniform vec2 u_shift;
varying vec3 v_operands;
void main() {
    gl_Position = vec4(a_position.x + u_shift.x,
                       a_position.y + u_shift.y, a_position.z, a_position.w);
    v_operands = vec3(a_color.x + u_shift.x, a_color.y, a_color.z);
}
"""
FRAGMENT_BODIES = {
    "register": """gl_FragColor = vec4(u_operands.x + u_operands.y,
                                     u_operands.z, u_operands.w, 1.0);""",
    "constants": """gl_FragColor = vec4(u_operands.x + 1.0, 1.0 + u_operands.y,
                                      u_operands.z + -0.5, 0.3 + u_operands.w);""",
    "tiny_constants": """gl_FragColor = vec4(u_operands.x + 0.000000059604644775390625,
        0.00006103515625 + u_operands.y, u_operands.z + 0.125, -1.0 + u_operands.w);""",
    "chained": """float first = u_operands.x + u_operands.y;
    float second = first + u_operands.z;
    gl_FragColor = vec4(first, second, second + u_operands.w, 1.0);""",
    "uniforms": """gl_FragColor = vec4(u_bias.x + u_bias.y, v_operands);""",
}
ADD_COUNTS = {
    "register": {"vs": 0, "fs": 1},
    "constants": {"vs": 0, "fs": 4},
    "tiny_constants": {"vs": 0, "fs": 4},
    "chained": {"vs": 0, "fs": 3},
    "uniforms": {"vs": 3, "fs": 1},
}


def half_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def half_value(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def ieee_half_add(left, right):
    total = half_value(left) + half_value(right)
    try:
        return half_bits(total)
    except OverflowError:
        return half_bits(math.copysign(math.inf, total))


def fragment_source(body, *, uniforms=False):
    declarations = ("varying vec3 v_operands;\nuniform vec2 u_bias;\n" if uniforms else
                    "varying vec4 v_operands;\nuniform vec4 u_operands;\n")
    return (
        "#version 100\nprecision mediump float;\n"
        + declarations
        + "void main() {\n"
        + body
        + "\n}\n"
    )


def write_shader_pair(directory, vertex, fragment):
    directory.mkdir(parents=True)
    (directory / "vert.glsl").write_text(vertex)
    (directory / "frag.glsl").write_text(fragment)


def compile_pair(directory, output, target):
    assembly = compile_glsl(directory, BUILD, output, target=target)
    layouts = {
        stage: ShaderLayout(json.loads((output / f"{stage}.layout.json").read_text()))
        for stage in ("vs", "fs")
    }
    return {"assembly": assembly, "layouts": layouts, "directory": directory}


def operand_payload(layout, components):
    entries = sorted(layout.entries("uniforms"), key=lambda entry: entry["word_offset"])
    assert len(entries) == len(components) == 4
    assert all(entry["type"] == "float" for entry in entries)
    return {
        layout.base("uniform") + entry["word_offset"] * layout.stride: components[index]
        for index, entry in enumerate(entries)
    }


def vertex_payload(layout, position, operands):
    locations = sorted({entry["location"] for entry in layout.entries("inputs")})
    assert len(locations) == 2
    attributes = dict(zip(locations, (
        tuple(half_bits(value) for value in position),
        tuple(half_bits(value) for value in operands),
    )))
    return {
        layout.base("input") + entry["word_offset"] * layout.stride:
        attributes[entry["location"]][entry["component"]]
        for entry in layout.entries("inputs")
    }


def packed_uniforms(layout, values):
    entries = sorted(layout.entries("uniforms"), key=lambda entry: entry["word_offset"])
    assert len(entries) == len(values)
    assert all(entry["type"] == "float" and entry["bit_size"] == 32 for entry in entries)
    contents = bytearray((entries[-1]["word_offset"] + 1) * 4)
    expected = {}
    for entry, value in zip(entries, values):
        struct.pack_into("<f", contents, entry["word_offset"] * 4, value)
        expected[entry["word_offset"]] = half_bits(value)
    payload = pack_constants(layout, [contents.hex()])
    assert payload == expected
    return payload


def assert_hardware_execution(result, expected):
    assert result["program_format"] == HARDWARE.value
    assert result["shader_abi_version"] == 2
    assert result["fence"] == "completed"
    assert result["host_steps"] == result["program_words"]
    assert not result["fpga_execution"] and not result["rtl_timing_validated"]
    assert result["outputs"]["fragment"]["raw_words"] == list(expected) + [0] * 12


@pytest.fixture(scope="module")
def compiler_build():
    if not (BUILD / "build.ninja").is_file() or not (ROOT / "tools/mgc_harness").is_file():
        pytest.skip("configured Mesa and freshly built GLES harness required")
    return BUILD


@pytest.fixture(scope="module")
def hardware_add_glsl(tmp_path_factory, compiler_build):
    root = tmp_path_factory.mktemp("hardware-add-glsl")
    artifacts = {}
    for name, body in FRAGMENT_BODIES.items():
        directory = root / name
        vertex = UNIFORM_VERTEX_SOURCE if name == "uniforms" else VERTEX_SOURCE
        write_shader_pair(directory, vertex, fragment_source(body, uniforms=name == "uniforms"))
        artifacts[name] = compile_pair(directory, root / f"{name}-hardware", HARDWARE)
    return artifacts


@pytest.fixture(scope="module")
def legacy_add_glsl(tmp_path_factory, hardware_add_glsl):
    output = tmp_path_factory.mktemp("legacy-add-glsl")
    return {
        name: compile_pair(hardware_add_glsl[name]["directory"], output / name, LEGACY)
        for name in ("register", "constants")
    }


@pytest.fixture
def managed_hardware_cpus(monkeypatch):
    instances = []

    def observed_cpu(*args, **kwargs):
        cpu = HardwareCPU(*args, **kwargs)
        instances.append(cpu)
        return cpu

    monkeypatch.setattr(hardware_module, "HardwareCPU", observed_cpu)
    return instances


@pytest.fixture(scope="module")
def utah_evidence(local_hardware_examples):
    source = UTAH.with_suffix(".asm").read_text()
    image = load_mem(UTAH.with_suffix(".program.mem"))
    instructions, _ = parse_source(source)
    return source, image, instructions


@pytest.mark.parametrize("pc,word,registers", GOLDEN_ADDS, ids=lambda value: str(value))
def test_imported_utah_add_word_encodes_register_fields(utah_evidence, pc, word, registers):
    _, image, instructions = utah_evidence
    assert image[pc] == word
    operands = tuple(f"r{register}" for register in registers)
    source_instruction = next(instruction for instruction in instructions if instruction.pc == pc)
    assert source_instruction.mnemonic == "fadd16"
    assert source_instruction.operands == operands
    decoded = decode_word(image[pc])
    assert decoded.mnemonic == "fadd16"
    assert decoded.source_a_register is True
    assert (decoded.source_a, decoded.source_b, decoded.destination) == registers
    assert decoded.condition is None and not decoded.invert
    assert bool(word & (1 << 21))
    assert ((word >> 16) & 31, (word >> 5) & 31, word & 31) == registers
    assert word == 0x20200000 | (registers[0] << 16) | (registers[1] << 5) | registers[2]
    assembly = "fadd16 " + ", ".join(operands)
    assert disassemble_hardware([word]) == [assembly]
    assert assemble_hardware(assembly + "\nwait_fadd") == [word] + [0] * 6
    assert assemble_hardware("\n".join(disassemble_hardware(image[pc:pc + 7]))) == image[pc:pc + 7]


def test_all_sixty_imported_utah_scalar_adds_and_explicit_six_word_waits(utah_evidence):
    source, image, instructions = utah_evidence
    adds = [(index, instruction) for index, instruction in enumerate(instructions)
            if instruction.mnemonic == "fadd16"]
    assert len(adds) == 60
    assert sum(decode_word(word).mnemonic == "fadd16" for word in image) == 60
    assert assemble_hardware(source) == image
    for index, instruction in adds:
        assert all(operand.startswith("r") for operand in instruction.operands)
        assert decode_word(image[instruction.pc]).source_a_register is True
    waited_adds = [(index, instruction) for index, instruction in adds
                   if instructions[index + 1].mnemonic == "wait_fadd"]
    assert len(waited_adds) == 24
    for index, instruction in waited_adds:
        assert instructions[index + 1].mnemonic == "wait_fadd"
        assert instructions[index + 1].pc == instruction.pc + 1
        assert instructions[index + 2].pc == instruction.pc + 7
        assert image[instruction.pc + 1:instruction.pc + 7] == [0] * 6


@pytest.mark.parametrize("name", FRAGMENT_BODIES)
def test_fresh_glsl_emits_only_verified_scalar_add_words(hardware_add_glsl, name):
    artifact = hardware_add_glsl[name]
    for stage in ("vs", "fs"):
        assembly = artifact["assembly"][stage]
        layout = artifact["layouts"][stage]
        assert assembly.startswith("; mgc fp16-hardware-v1 backend, ABI=2,")
        assert layout.target == HARDWARE and layout.stride == 4
        assert all(entry["type"] == "float" for entry in layout.entries("outputs"))
        words = assemble_hardware(assembly)
        decoded = [decode_word(word) for word in words]
        assert {instruction.mnemonic for instruction in decoded} <= {
            "preload_low", "load_high", "copy", "load", "store", "nop", "fadd16",
        }
        assert sum(instruction.mnemonic == "fadd16" for instruction in decoded) == ADD_COUNTS[name][stage]
        assert assemble_hardware("\n".join(disassemble_hardware(words))) == words
        instructions, _ = parse_source(assembly)
        for index, instruction in enumerate(instructions):
            if instruction.mnemonic != "fadd16":
                continue
            registers = tuple(int(operand[1:]) for operand in instruction.operands)
            assert instruction.operands == tuple(f"r{register}" for register in registers)
            actual = decoded[instruction.pc]
            assert actual.source_a_register is True
            assert (actual.source_a, actual.source_b, actual.destination) == registers
            assert words[instruction.pc] == 0x20200000 | (registers[0] << 16) | (registers[1] << 5) | registers[2]
            assert instructions[index + 1].mnemonic == "wait_fadd"
            assert instructions[index + 1].pc == instruction.pc + 1
            assert instructions[index + 2].pc == instruction.pc + 7
            assert words[instruction.pc + 1:instruction.pc + 7] == [0] * 6


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("left,right,known_expected", ADD_CASES)
def test_runtime_register_add_matches_independent_binary16(
    hardware_add_glsl, managed_hardware_cpus, left, right, known_expected,
):
    artifact = hardware_add_glsl["register"]
    expected = ieee_half_add(left, right)
    assert expected == known_expected
    operands = (left, right, half_bits(0.25), half_bits(-0.5))
    result = execute_assembly(
        artifact["assembly"]["fs"], raw_values=operand_payload(artifact["layouts"]["fs"], operands),
        target=HARDWARE,
    )
    assert_hardware_execution(result, (expected, *operands[2:], half_bits(1.0)))
    assert len(managed_hardware_cpus) == 1
    cpu = managed_hardware_cpus[0]
    assert isinstance(cpu, HardwareCPU) and isinstance(cpu.memory, ByteMemory)
    assert not cpu.allow_approximate_fp16
    assert cpu.instruction_counts["fadd16"] == 1
    assert cpu.halted and cpu.pc == result["program_words"]


@pytest.mark.parametrize("name,constants", [("constants", CONSTANTS), ("tiny_constants", TINY_CONSTANTS)])
def test_add_literals_are_materialized_registers_not_constant_bank(hardware_add_glsl, name, constants):
    instructions, _ = parse_source(hardware_add_glsl[name]["assembly"]["fs"])
    adds = [instruction for instruction in instructions if instruction.mnemonic == "fadd16"]
    assert len(adds) == 4
    assert all(all(operand.startswith("r") for operand in instruction.operands) for instruction in adds)
    for constant in constants:
        materializations = [
            instruction for instruction in instructions
            if instruction.mnemonic == "movi_i32" and int(instruction.operands[0], 0) == half_bits(constant)
        ]
        assert materializations
        assert any(
            load.pc < add.pc and load.operands[1] in add.operands[:2]
            for load in materializations for add in adds
        )


@pytest.mark.parametrize("name,constants", [("constants", CONSTANTS), ("tiny_constants", TINY_CONSTANTS)])
@pytest.mark.parametrize("values", [
    pytest.param((0.0, 0.0, 0.0, 0.0), id="zero"),
    pytest.param((0.25, -0.75, 1.25, -0.5), id="mixed-sign"),
    pytest.param((-1.0, -1.0, 0.5, -0.3), id="cancellation"),
    pytest.param((2.0**-24, -(2.0**-24), 2.0**-11, -(2.0**-12)), id="subnormal-rounding"),
    pytest.param((65504.0, -65504.0, -65504.0, 65504.0), id="maximum-finite"),
])
def test_runtime_plus_materialized_constants_matches_binary16(hardware_add_glsl, name, constants, values):
    artifact = hardware_add_glsl[name]
    result = execute_assembly(
        artifact["assembly"]["fs"], values=operand_payload(artifact["layouts"]["fs"], values), target=HARDWARE,
    )
    expected = [ieee_half_add(half_bits(value), half_bits(constant)) for value, constant in zip(values, constants)]
    assert_hardware_execution(result, expected)


@pytest.mark.parametrize("operands", [
    pytest.param((0x3C00, 0x1000, 0x1000, 0x0000), id="round-at-each-add"),
    pytest.param((0x3C01, 0x1000, 0x1000, 0x1000), id="odd-then-even-ties"),
    pytest.param((0x7BFF, 0xFBFF, 0x0001, 0x0001), id="large-cancellation-then-subnormal"),
    pytest.param((0x0400, 0x8001, 0x8001, 0x0000), id="subnormal-chain"),
    pytest.param((0xBC00, 0x9000, 0x9000, 0x8000), id="negative-rounding"),
    pytest.param((0x3800, 0xB400, 0xBC00, 0x3C00), id="mixed-sign-chain"),
])
def test_chained_sums_round_each_intermediate_to_binary16(hardware_add_glsl, operands):
    artifact = hardware_add_glsl["chained"]
    first = ieee_half_add(operands[0], operands[1])
    second = ieee_half_add(first, operands[2])
    third = ieee_half_add(second, operands[3])
    if operands == (0x3C00, 0x1000, 0x1000, 0x0000):
        assert second != half_bits(sum(half_value(value) for value in operands[:3]))
    result = execute_assembly(
        artifact["assembly"]["fs"], raw_values=operand_payload(artifact["layouts"]["fs"], operands),
        target=HARDWARE,
    )
    assert_hardware_execution(result, (first, second, third, half_bits(1.0)))


@pytest.mark.parametrize("position,operands,shift,bias", [
    pytest.param((0.25, -0.5, 0.75, 1.0), (0.5, -0.25, 0.125, 1.0),
                 (0.125, -0.25), (0.25, -0.5), id="mixed-sign"),
    pytest.param((0.0, -0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0),
                 (0.0, -0.0), (0.0, 0.0), id="zero"),
    pytest.param((1.0, 2.0**-14, 0.0, 1.0), (1.0, 2.0**-11, 2.0**-24, 0.5),
                 (2.0**-11, -(2.0**-24)), (2.0**-11, 2.0**-24), id="rounding-subnormal"),
])
def test_reflected_pair_adds_in_both_stages_with_typed_uniforms(
    hardware_add_glsl, managed_hardware_cpus, position, operands, shift, bias,
):
    artifact = hardware_add_glsl["uniforms"]
    vertex_layout, fragment_layout = (artifact["layouts"][stage] for stage in ("vs", "fs"))
    mapping = link_varyings(vertex_layout, fragment_layout)
    assert len(mapping) == 3
    vertex_uniforms = packed_uniforms(vertex_layout, shift)
    fragment_uniforms = packed_uniforms(fragment_layout, bias)
    vertex_values = vertex_payload(vertex_layout, position, operands)
    vertex_values.update({vertex_layout.base("uniform") + offset * 4: value
                          for offset, value in vertex_uniforms.items()})
    result = execute_shader_pair(
        artifact["assembly"], varying_map={source: destination for destination, source in mapping.items()},
        vertex_raw_values=vertex_values,
        fragment_raw_values={fragment_layout.base("uniform") + offset * 4: value
                             for offset, value in fragment_uniforms.items()},
        target=HARDWARE,
    )
    position_bits = [half_bits(value) for value in position]
    operand_bits = [half_bits(value) for value in operands]
    shift_bits = [half_bits(value) for value in shift]
    bias_bits = [half_bits(value) for value in bias]
    position_bits[0] = ieee_half_add(position_bits[0], shift_bits[0])
    position_bits[1] = ieee_half_add(position_bits[1], shift_bits[1])
    operand_bits[0] = ieee_half_add(operand_bits[0], shift_bits[0])
    expected_fragment = (
        ieee_half_add(bias_bits[0], bias_bits[1]),
        *operand_bits[:3],
    )
    for entry in vertex_layout.entries("outputs"):
        expected = position_bits if entry["location"] == 0 else operand_bits
        assert result["outputs"]["vertex"]["raw_words"][entry["word_offset"]] == expected[entry["component"]]
    assert_hardware_execution(result["stages"]["fs"], expected_fragment)
    assert result["fence"] == "completed" and result["program_format"] == HARDWARE.value
    assert not result["interpolation"] and not result["rasterization"]
    assert len(managed_hardware_cpus) == 2
    assert [cpu.instruction_counts["fadd16"] for cpu in managed_hardware_cpus] == [3, 1]


def test_shader_executor_repeated_add_invocations_use_real_managed_hardware_cpu(
    hardware_add_glsl, managed_hardware_cpus,
):
    artifact = hardware_add_glsl["uniforms"]
    layout = artifact["layouts"]["fs"]
    bias = (0.125, -(2.0**-24))
    uniforms = packed_uniforms(layout, bias)
    executor = ShaderExecutor(artifact["assembly"]["fs"], layout, uniforms)
    try:
        assert isinstance(executor.driver.backend, HardwareSimulationBackend)
        assert executor.program.format == HARDWARE
        cases = ((1.0, 2.0**-11, 2.0**-14, 0.5), (-1.0, 0.25, 0.0, -0.0),
                 (0.0, 0.0, 2.0**-24, 1.0))
        uniform_cases = ((1.0, 2.0**-11), (-1.0, 0.25), (2.0**-24, 2.0**-24))
        for values, uniform_values in zip(cases, uniform_cases):
            operands = tuple(half_bits(value) for value in values)
            inputs = {entry["word_offset"]: operands[entry["component"]] for entry in layout.entries("inputs")}
            for offset, value in packed_uniforms(layout, uniform_values).items():
                executor.buffers["uniform"].upload(struct.pack("<I", value), offset=offset * 4)
            actual = executor.invoke(inputs)
            expected = (
                ieee_half_add(half_bits(uniform_values[0]), half_bits(uniform_values[1])),
                *operands[:3],
            )
            assert actual == {entry["word_offset"]: expected[entry["component"]]
                              for entry in layout.entries("outputs")}
            for entry in layout.entries("outputs"):
                stored = executor.buffers["fragment"].download(offset=entry["word_offset"] * 4, size=4)
                assert struct.unpack("<I", stored)[0] == actual[entry["word_offset"]]
        assert executor.invocations == 3
        assert executor.steps == len(executor.program.words) * 3
        assert len(managed_hardware_cpus) == 3
        assert len({id(cpu) for cpu in managed_hardware_cpus}) == 3
        assert all(cpu.halted and cpu.instruction_counts["fadd16"] == 1 for cpu in managed_hardware_cpus)
    finally:
        executor.close()


@pytest.mark.parametrize("name", ["register", "constants"])
def test_add_hardware_and_legacy_formats_execute_only_under_their_own_abi(
    hardware_add_glsl, legacy_add_glsl, name,
):
    operands = (0.25, -0.5, 0.75, 0.125)
    expected = (
        (ieee_half_add(half_bits(operands[0]), half_bits(operands[1])),
         half_bits(operands[2]), half_bits(operands[3]), half_bits(1.0))
        if name == "register" else
        tuple(ieee_half_add(half_bits(value), half_bits(constant)) for value, constant in zip(operands, CONSTANTS))
    )
    for target, artifacts, other in ((HARDWARE, hardware_add_glsl, legacy_add_glsl),
                                     (LEGACY, legacy_add_glsl, hardware_add_glsl)):
        artifact = artifacts[name]
        layout = artifact["layouts"]["fs"]
        assembly = artifact["assembly"]["fs"]
        assert layout.target == target
        result = execute_assembly(assembly, values=operand_payload(layout, operands), target=target)
        assert result["program_format"] == target.value
        assert result["shader_abi_version"] == (2 if target == HARDWARE else 1)
        assert result["outputs"]["fragment"]["raw_words"][:4] == list(expected)
        with pytest.raises(ValueError, match="never execute|explicit compiler target/ABI"):
            execute_assembly(assembly, target=other[name]["layouts"]["fs"].target)
        with pytest.raises(ValueError, match="provenance"):
            ShaderExecutor(assembly, other[name]["layouts"]["fs"])
    if name == "constants":
        assert "#3" in legacy_add_glsl[name]["assembly"]["fs"]
        assert "#" not in hardware_add_glsl[name]["assembly"]["fs"]


@pytest.mark.parametrize("target,backend_type", [(HARDWARE, SimulationBackend), (LEGACY, HardwareSimulationBackend)])
def test_fresh_add_program_cannot_be_submitted_to_the_other_format_backend(
    hardware_add_glsl, legacy_add_glsl, target, backend_type,
):
    artifact = (hardware_add_glsl if target == HARDWARE else legacy_add_glsl)["register"]
    assembly = artifact["assembly"]["fs"]
    words = assemble_hardware(assembly) if target == HARDWARE else assemble(assembly)
    program = Program.from_words(words, format=target)
    with Driver(backend_type(memory_size=5 << 20)) as driver:
        with pytest.raises(ProgramFormatError, match="format|legacy|hardware"):
            driver.submit(program)


@pytest.mark.parametrize("target,ambient", [(HARDWARE, LEGACY), (LEGACY, HARDWARE)])
def test_fresh_add_explicit_target_overrides_ambient_format(
    hardware_add_glsl, tmp_path, monkeypatch, target, ambient,
):
    monkeypatch.setenv("MGC_COMPILE_TARGET", ambient.value)
    artifact = compile_pair(hardware_add_glsl["register"]["directory"], tmp_path, target)
    assert all(layout.target == target for layout in artifact["layouts"].values())
    profile = HARDWARE.value if target == HARDWARE else "legacy-simulator"
    version = 2 if target == HARDWARE else 1
    assert all(assembly.startswith(f"; mgc {profile} backend, ABI={version},")
               for assembly in artifact["assembly"].values())


UNSUPPORTED_EXPRESSIONS = (
    pytest.param("{left} * {right}", id="fmul"),
    pytest.param("{left} / {right}", id="fdiv"),
    pytest.param("-{left}", id="fneg"),
    pytest.param("{left} - {right}", id="fsub"),
    pytest.param("abs({left})", id="fabs"),
    pytest.param("min({left}, {right})", id="fmin"),
    pytest.param("float(int({left}))", id="integer-conversion"),
)


@pytest.mark.parametrize("stage", ["vs", "fs"])
@pytest.mark.parametrize("expression", UNSUPPORTED_EXPRESSIONS)
def test_fresh_unsupported_glsl_operation_fails_compiler_without_assembly_artifacts(
    compiler_build, tmp_path, stage, expression,
):
    if stage == "vs":
        unsupported = expression.format(left="a_position.z", right="a_position.w")
        vertex = VERTEX_SOURCE.replace(
            "gl_Position = a_position;",
            "gl_Position = vec4(a_position.x + a_position.y, " + unsupported + ", a_position.z, a_position.w);",
        )
        fragment = fragment_source(FRAGMENT_BODIES["register"])
    else:
        unsupported = expression.format(left="u_operands.z", right="u_operands.w")
        vertex = VERTEX_SOURCE
        fragment = fragment_source(
            "gl_FragColor = vec4(u_operands.x + u_operands.y, " + unsupported + ", u_operands.z, 1.0);",
        )
    directory, output = tmp_path / "source", tmp_path / "output"
    write_shader_pair(directory, vertex, fragment)
    with pytest.raises(ValueError, match="GLSL/MGC compilation failed"):
        compile_glsl(directory, compiler_build, output, target=HARDWARE)
    log = (output / "compile.log").read_text()
    assert "hardware target" in log and "unsupported NIR instruction" in log
    assert "HardwareEncodingError" not in log
    assert not list(output.rglob("*.asm"))
    assert not list(output.rglob("*.program.mem"))


@pytest.mark.parametrize("literal", ["65520.0", "1e10"])
def test_fresh_add_rejects_nonrepresentable_finite_literal_without_artifacts(
    compiler_build, tmp_path, literal,
):
    directory, output = tmp_path / "source", tmp_path / "output"
    write_shader_pair(
        directory, VERTEX_SOURCE,
        fragment_source(f"gl_FragColor = vec4(u_operands.x + {literal}, u_operands.y, u_operands.z, 1.0);"),
    )
    with pytest.raises(ValueError, match="GLSL/MGC compilation failed"):
        compile_glsl(directory, compiler_build, output, target=HARDWARE)
    assert "finite representable FP16 literals" in (output / "compile.log").read_text()
    assert not list(output.rglob("*.asm"))
    assert not list(output.rglob("*.program.mem"))
