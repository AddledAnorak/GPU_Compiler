"""Fresh NIR mixed-axis lowering, independent of managed texture consumers."""

import json
import os
import re
import subprocess

import numpy as np
import pytest

from test_backend_runtime import backend_binary
from tools.mgcasm import assemble, disassemble
from tools.mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from tools.mgcsim import CPU
from tools.mgcsim.cpu import MgcSimError


WRAPS = ("clamp-to-edge", "repeat", "mirrored-repeat")
MIXED_OPTIONS = {7: "host-texture-mixed", 8: "host-texture-linear-mixed"}
LEGACY_OPTIONS = (None, "host-texture", "host-texture-linear", "host-texture-repeat",
                  "host-texture-linear-repeat", "host-texture-mirrored-repeat",
                  "host-texture-linear-mirrored-repeat")
INPUT = 0x10000
UNIFORM = 0x20000
FRAMEBUFFER = 0x100000
TEXTURE_BASE = UNIFORM + 15 * 0x1000
CONFIGURATIONS = [(version, wrap_s, wrap_t) for version in MIXED_OPTIONS
                  for wrap_s in WRAPS for wrap_t in WRAPS]


def compile_texture(binary, option=None, *, mode="texture-rgba", count=4, wraps=None,
                    extra_environment=None, success=True):
    environment = dict(os.environ)
    for variable in ("MGC_TEST_WRAP_S", "MGC_TEST_WRAP_T", "MGC_TEST_SOURCE_LOD", "MGC_NIR_DUMP"):
        environment.pop(variable, None)
    environment["MGC_TEST_LAYOUT"] = "1"
    if wraps is not None:
        environment.update(MGC_TEST_WRAP_S=wraps[0], MGC_TEST_WRAP_T=wraps[1])
    environment.update(extra_environment or {})
    command = [str(binary), mode, str(count)]
    if option is not None:
        command.append(option)
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    if not success:
        assert result.returncode == 1, result.stderr
        assert not result.stdout
        assert "error exposed partial assembly" not in result.stderr
        return result.stderr
    assert result.returncode == 0, result.stderr
    layout = json.loads(next(line.removeprefix("layout=") for line in result.stderr.splitlines()
                             if line.startswith("layout=")))
    statistics = {name: int(value) for name, value in re.findall(r"(\w+)=(\d+)", result.stderr)
                  if name != "version"}
    words = assemble(result.stdout)
    assert assemble("\n".join(disassemble(words))) == words
    return {"assembly": result.stdout, "words": words, "layout": layout, "stats": statistics}


@pytest.fixture(scope="module", params=CONFIGURATIONS,
                ids=[f"v{version}-{wrap_s}-{wrap_t}" for version, wrap_s, wrap_t in CONFIGURATIONS])
def mixed_program(backend_binary, request):
    version, wrap_s, wrap_t = request.param
    program = compile_texture(backend_binary, MIXED_OPTIONS[version], wraps=(wrap_s, wrap_t))
    program.update(version=version, wrap_s=wrap_s, wrap_t=wrap_t)
    return program


def texture_texels(width, height):
    channels = bytes(channel % 256 for row in range(height) for column in range(width)
                     for channel in (column * 29 + row * 53 + 11,
                                     column * 71 + row * 17 + 37,
                                     column * 13 + row * 97 + 19,
                                     column * 7 + row * 11 + 151))
    return (np.frombuffer(channels, dtype=np.uint8).astype(np.float64) / 255).astype(np.float16).reshape(
        height, width, 4)


def fp16_axis(coordinate, size, wrap, linear):
    coordinate = np.float16(coordinate)
    if wrap == "mirrored-repeat":
        period = np.float16(np.floor(np.float16(coordinate * np.float16(0.5))))
        coordinate = np.float16(coordinate - np.float16(period * np.float16(2)))
        coordinate = np.float16(min(coordinate, np.float16(np.float16(2) - coordinate)))
    if wrap == "repeat":
        coordinate = np.float16(coordinate - np.float16(np.floor(coordinate)))
    else:
        coordinate = np.float16(min(max(coordinate, np.float16(0)), np.float16(1)))
    scaled = np.float16(coordinate * np.float16(size))
    if not linear:
        texel = int(np.floor(scaled))
        return (texel % size if wrap == "repeat" else min(texel, size - 1),), None
    position = np.float16(scaled - np.float16(0.5))
    lower = int(np.floor(position))
    weight = np.float16(position - np.float16(lower))
    if wrap == "repeat":
        return (lower % size, (lower + 1) % size), weight
    return (min(max(lower, 0), size - 1), min(max(lower + 1, 0), size - 1)), weight


def fp16_sample(texels, coordinates, wraps, linear):
    horizontal, weight_s = fp16_axis(coordinates[0], texels.shape[1], wraps[0], linear)
    vertical, weight_t = fp16_axis(coordinates[1], texels.shape[0], wraps[1], linear)
    if not linear:
        return texels[vertical[0], horizontal[0]]
    rows = []
    for row in vertical:
        lower, upper = (texels[row, column] for column in horizontal)
        difference = (upper - lower).astype(np.float16)
        rows.append((lower + (difference * weight_s).astype(np.float16)).astype(np.float16))
    difference = (rows[1] - rows[0]).astype(np.float16)
    return (rows[0] + (difference * weight_t).astype(np.float16)).astype(np.float16)


def sample_cpu(program, width, height, coordinates, *, max_cycles=100000, texels=None, payload=None):
    texels = texture_texels(width, height) if texels is None else texels
    cpu = CPU(max_cycles=max_cycles)
    cpu.mem.store(INPUT, float_to_fp16_bits(coordinates[0]))
    cpu.mem.store(INPUT + 1, float_to_fp16_bits(coordinates[1]))
    if payload is None:
        cpu.mem.store(TEXTURE_BASE, float_to_fp16_bits(width))
        cpu.mem.store(TEXTURE_BASE + 1, float_to_fp16_bits(height))
        cpu.mem.store(TEXTURE_BASE + 2, 0)
        cpu.mem.store(TEXTURE_BASE + 3, 0)
        for index, channel in enumerate(texels.flat):
            cpu.mem.store(TEXTURE_BASE + 4 + index, float_to_fp16_bits(float(channel)))
    else:
        for address, value in payload.items():
            cpu.mem.store(UNIFORM + address, value)
    reads, writes = [], []
    original_load, original_store = cpu.mem.load, cpu.mem.store

    def checked_load(address):
        assert address in (INPUT, INPUT + 1) or TEXTURE_BASE <= address < TEXTURE_BASE + 4 + texels.size
        reads.append(address)
        return original_load(address)

    def checked_store(address, value):
        assert FRAMEBUFFER <= address < FRAMEBUFFER + 4
        writes.append(address)
        original_store(address, value)

    cpu.mem.load, cpu.mem.store = checked_load, checked_store
    return cpu, texels, reads, writes


def test_fresh_layout_and_allocation_contract(mixed_program):
    layout = mixed_program["layout"]
    assert layout["version"] == 2 and layout["shader_abi_version"] == 1
    assert layout["program_format"] == "legacy-mgcsim-v1"
    assert layout["stage"] == "fragment" and layout["address_units"] == "words"
    assert layout["supported"] is True and layout["errors"] == []
    assert layout["bases"] == {"input": INPUT, "uniform": UNIFORM, "vertex": 0x30000,
                               "fragment": FRAMEBUFFER}
    assert layout["host_texture"] == {
        "version": mixed_program["version"], "texture": 0, "sampler": 0, "storage_block": 15,
        "max_width": 16, "max_height": 16,
        "filter": "linear" if mixed_program["version"] == 8 else "nearest",
        "wrap_s": mixed_program["wrap_s"], "wrap_t": mixed_program["wrap_t"],
    }
    assert list(layout["host_texture"]) == ["version", "texture", "sampler", "storage_block",
                                            "max_width", "max_height", "filter", "wrap_s", "wrap_t"]
    assert "wrap" not in layout["host_texture"]
    assert [entry["word_offset"] for entry in layout["inputs"]] == [0, 1]
    assert [entry["word_offset"] for entry in layout["outputs"]] == [0, 1, 2, 3]
    assert layout["uniforms"] == []
    assert set(layout) == {"version", "program_format", "shader_abi_version", "stage",
                           "address_units", "bases", "host_texture", "inputs", "outputs", "uniforms",
                           "uses_discard", "supported", "errors"}
    statistics = mixed_program["stats"]
    assert statistics["abi"] == 1
    assert 0 < statistics["registers"] <= 32 and 0 <= statistics["predicates"] <= 8
    assert 20 < statistics["instructions"] < 2000
    assert statistics["waits"] <= statistics["immediate_waits"]


@pytest.mark.parametrize("width,height", ((1, 1), (1, 16), (16, 1), (3, 5), (5, 3), (16, 16)))
@pytest.mark.parametrize("coordinates", ((0, 0), (1, 1), (-1, -3), (2, 4), (3, 5),
                                         (-0.125, 1.875), (1.875, -0.125), (0.5, 0.5),
                                         (0.37, 0.61), (-1.37, 2.61), (-65504, 65504),
                                         (-2 ** -14, -2 ** -24), (2 ** -24, 1 - 2 ** -11),
                                         (1 + 2 ** -10, 1 - 2 ** -11)))
def test_all_axis_pairs_match_fp16_and_stay_in_uploaded_storage(mixed_program, width, height,
                                                              coordinates):
    cpu, texels, reads, writes = sample_cpu(mixed_program, width, height, coordinates)
    cpu.run(mixed_program["words"])
    assert cpu.halted and cpu.cycles < cpu.max_cycles
    assert sorted(writes) == list(range(FRAMEBUFFER, FRAMEBUFFER + 4))
    actual_bits = [cpu.mem._framebuffer[index] for index in range(4)]
    expected = fp16_sample(texels, coordinates,
                          (mixed_program["wrap_s"], mixed_program["wrap_t"]),
                          mixed_program["version"] == 8)
    if mixed_program["version"] == 7:
        assert actual_bits == [float_to_fp16_bits(float(channel)) for channel in expected]
    else:
        np.testing.assert_allclose([fp16_bits_to_float(channel) for channel in actual_bits],
                                   expected.astype(np.float64), rtol=0, atol=2 ** -10)
    assert INPUT in reads and INPUT + 1 in reads
    assert TEXTURE_BASE in reads and TEXTURE_BASE + 1 in reads
    assert len([address for address in reads if address >= TEXTURE_BASE + 4]) == (
        16 if mixed_program["version"] == 8 else 4)


def test_step_budget_does_not_commit_partial_output(mixed_program):
    cpu, _, _, writes = sample_cpu(mixed_program, 3, 5, (-0.125, 1.875), max_cycles=1)
    with pytest.raises(MgcSimError, match="exceeded max_cycles=1"):
        cpu.run(mixed_program["words"])
    assert not cpu.halted and writes == [] and cpu.mem.dump_framebuffer() == {}


@pytest.mark.parametrize("version", range(1, 7))
def test_equal_axes_preserve_legacy_assembly_and_metadata_verbatim(backend_binary, version):
    wrap = WRAPS[(version - 1) // 2]
    legacy = compile_texture(backend_binary, LEGACY_OPTIONS[version], wraps=(WRAPS[0], WRAPS[0]))
    mixed = compile_texture(backend_binary, MIXED_OPTIONS[8 if version % 2 == 0 else 7],
                            wraps=(wrap, wrap))
    assert mixed["assembly"] == legacy["assembly"]
    assert mixed["stats"] == legacy["stats"]
    assert legacy["layout"]["host_texture"] == {
        "version": version, "texture": 0, "sampler": 0, "storage_block": 15,
        "max_width": 16, "max_height": 16,
        "filter": "linear" if version % 2 == 0 else "nearest", "wrap": wrap,
    }
    assert list(legacy["layout"]["host_texture"]) == ["version", "texture", "sampler", "storage_block",
                                                      "max_width", "max_height", "filter", "wrap"]
    assert {key: value for key, value in mixed["layout"].items() if key != "host_texture"} == {
        key: value for key, value in legacy["layout"].items() if key != "host_texture"}


def test_exported_helpers_and_zero_defaults(backend_binary):
    result = subprocess.run([str(backend_binary), "texture-helpers"], capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and result.stderr == ""


@pytest.mark.parametrize("version", (7, 8))
def test_mixed_versions_default_both_axes_to_clamp(backend_binary, version):
    program = compile_texture(backend_binary, MIXED_OPTIONS[version])
    assert program["layout"]["host_texture"]["wrap_s"] == WRAPS[0]
    assert program["layout"]["host_texture"]["wrap_t"] == WRAPS[0]


@pytest.mark.parametrize("version", range(7))
@pytest.mark.parametrize("variable", ("MGC_TEST_WRAP_S", "MGC_TEST_WRAP_T"))
@pytest.mark.parametrize("wrap", WRAPS[1:])
def test_legacy_and_default_options_reject_nonzero_axis_state(backend_binary, version, variable, wrap):
    diagnostic = compile_texture(backend_binary, LEGACY_OPTIONS[version], mode="copy",
                                 extra_environment={variable: wrap}, success=False)
    assert "host texture wrap axes require mixed version 7 or 8" in diagnostic


@pytest.mark.parametrize("version", (7, 8))
@pytest.mark.parametrize("axis", ("s", "t"))
@pytest.mark.parametrize("value", (3, 99, -1, 0xFFFFFFFF))
def test_compiler_rejects_unknown_wrap_enums_without_assembly(backend_binary, version, axis, value):
    option = MIXED_OPTIONS[version] + "-invalid-" + axis
    diagnostic = compile_texture(backend_binary, option, count=value, success=False)
    assert f"host texture wrap_{axis} is unsupported" in diagnostic


@pytest.mark.parametrize("variable", ("MGC_TEST_WRAP_S", "MGC_TEST_WRAP_T"))
@pytest.mark.parametrize("value", ("", "0", "1", "2", "clamp", "Repeat", " repeat", "repeat ",
                                   "mirrored_repeat", "clamp-to-border", "CLAMP_TO_EDGE", "nan"))
def test_helper_wrap_configuration_is_strict(backend_binary, variable, value):
    diagnostic = compile_texture(backend_binary, MIXED_OPTIONS[7],
                                 extra_environment={variable: value}, success=False)
    assert f"{variable} must be clamp-to-edge, repeat, or mirrored-repeat" in diagnostic


@pytest.mark.parametrize("option", (None, "hardware"))
def test_unconfigured_texture_abi_rejects_sampling(backend_binary, option):
    diagnostic = compile_texture(backend_binary, option, success=False)
    assert "texture sampling is not supported" in diagnostic


@pytest.mark.parametrize("option", ["hardware-" + option for option in LEGACY_OPTIONS[1:]] +
                         ["hardware-" + option for option in MIXED_OPTIONS.values()])
def test_hardware_rejects_every_host_texture_version(backend_binary, option):
    diagnostic = compile_texture(backend_binary, option, success=False)
    assert "host texture version is unsupported for this target" in diagnostic


def test_hardware_default_rejects_nonzero_new_axis_state(backend_binary):
    diagnostic = compile_texture(backend_binary, "hardware", mode="copy",
                                 extra_environment={"MGC_TEST_WRAP_T": "repeat"}, success=False)
    assert "host texture wrap axes require mixed version 7 or 8" in diagnostic


def test_unknown_version_is_now_eleven_and_rejects_without_texture(backend_binary):
    diagnostic = compile_texture(backend_binary, "host-texture-unknown", mode="copy", success=False)
    assert "host texture version is unsupported for this target" in diagnostic


@pytest.mark.parametrize("version", (7, 8))
def test_register_allocation_failure_never_exposes_partial_assembly(backend_binary, version):
    diagnostic = compile_texture(backend_binary, MIXED_OPTIONS[version], mode="copy-register-pressure",
                                 wraps=("repeat", "mirrored-repeat"), success=False)
    assert "register pressure exceeds 32 registers" in diagnostic
