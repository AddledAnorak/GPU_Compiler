"""Versioned bilinear inputs and fresh bounded legacy compiler execution."""

import copy
import re
import struct
import subprocess

import numpy as np
import pytest

from test_backend_runtime import backend_binary
from tools.host_render import FragmentStatus, ShaderExecutor, pack_float
from tools.host_texture import (HOST_TEXTURE_LAYOUT, HOST_TEXTURE_LINEAR_LAYOUT,
                               LINEAR_SAMPLER_STATE, SAMPLER_STATE, TEXTURE_BASE,
                               host_texture_contract, pack_texture)
from tools.mgcasm import assemble, disassemble
from tools.mgcdriver import ExecutionTimeoutError, ProgramFormat
from tools.shader_layout import ShaderLayout


def layout_source(version=2):
    return {"version": 2, "supported": True, "errors": [],
            "program_format": ProgramFormat.LEGACY.value, "shader_abi_version": 1,
            "address_units": "words", "stage": "fragment",
            "bases": {"input": 0x10000, "uniform": 0x20000,
                      "vertex": 0x30000, "fragment": 0x100000},
            "inputs": [{"word_offset": component, "component": component, "location": 0, "interpolation": "smooth",
                        "type": "float", "bit_size": 32} for component in range(2)],
            "outputs": [{"word_offset": 0, "component": 0, "location": 0, "interpolation": "smooth",
                         "type": "float", "bit_size": 32}],
            "uniforms": [], "host_texture": host_texture_contract(version)}


def texture_payload(width, height, version=2):
    contents = bytes((index * 37 + 19) % 256 for index in range(width * height * 4))
    return {"width": width, "height": height, "format": "R8G8B8A8_UNORM",
            "data_hex": contents.hex(),
            "sampler": dict(LINEAR_SAMPLER_STATE if version == 2 else SAMPLER_STATE)}


def fp16_bilinear(texture, horizontal, vertical):
    texels = np.frombuffer(bytes.fromhex(texture["data_hex"]), dtype=np.uint8)
    texels = (texels.astype(np.float64) / 255).astype(np.float16).reshape(
        texture["height"], texture["width"], 4)
    indices = []
    fractions = []
    for value, size in ((horizontal, texture["width"]), (vertical, texture["height"])):
        coordinate = np.clip(np.float16(value), np.float16(0), np.float16(1))
        position = np.float16(np.float16(coordinate * np.float16(size)) - np.float16(0.5))
        lower = np.float16(np.floor(position))
        fractions.append(np.float16(position - lower))
        indices.append((max(0, min(int(lower), size - 1)),
                        max(0, min(int(lower) + 1, size - 1))))
    lower = np.float16(texels[indices[1][0], indices[0][0]] + np.float16(
        np.float16(texels[indices[1][0], indices[0][1]] - texels[indices[1][0], indices[0][0]]) * fractions[0]))
    higher = np.float16(texels[indices[1][1], indices[0][0]] + np.float16(
        np.float16(texels[indices[1][1], indices[0][1]] - texels[indices[1][1], indices[0][0]]) * fractions[0]))
    return np.float16(lower + np.float16(np.float16(higher - lower) * fractions[1]))


@pytest.mark.parametrize("version", (1, 2))
@pytest.mark.parametrize("width,height", ((1, 1), (1, 16), (16, 1), (3, 5), (16, 16)))
def test_versioned_texture_payload_keeps_input_bytes_bounds_and_fp16_words(version, width, height):
    source = layout_source(version)
    texture = texture_payload(width, height, version)
    original = copy.deepcopy((source, texture))
    reflected = ShaderLayout(source)
    payload = pack_texture(reflected, [texture])
    assert (source, texture) == original
    assert set(payload) == set(range(TEXTURE_BASE, TEXTURE_BASE + 4 + width * height * 4))
    assert payload[TEXTURE_BASE] == pack_float(width)
    assert payload[TEXTURE_BASE + 1] == pack_float(height)
    assert payload[TEXTURE_BASE + 2] == payload[TEXTURE_BASE + 3] == 0
    for index, value in enumerate(bytes.fromhex(texture["data_hex"])):
        assert payload[TEXTURE_BASE + 4 + index] == pack_float(value / 255)
    assert max(payload) < TEXTURE_BASE + 4096


@pytest.mark.parametrize("value", (0, 11, -1, True, False, None, "2", 2.0, {}, []))
def test_texture_contract_rejects_unknown_and_noninteger_versions(value):
    with pytest.raises(ValueError, match="version"):
        host_texture_contract(value)


def test_contract_selection_does_not_mutate_shared_constants():
    first = host_texture_contract(1)
    second = host_texture_contract(2)
    first["version"] = 100
    second["filter"] = "unsupported"
    assert host_texture_contract(1) == HOST_TEXTURE_LAYOUT
    assert host_texture_contract(2) == HOST_TEXTURE_LINEAR_LAYOUT


@pytest.mark.parametrize("version", (1, 2))
def test_compiler_version_and_captured_filter_must_match(version):
    texture = texture_payload(2, 2, 3 - version)
    with pytest.raises(ValueError, match="sampler"):
        pack_texture(layout_source(version), [texture])


@pytest.mark.parametrize("field,value", (("version", 1), ("version", 3), ("version", True),
                                        ("filter", "nearest"), ("filter", "linear-mipmap-linear"),
                                        ("wrap", "repeat"), ("max_width", 17), ("texture", 1),
                                        ("sampler", 1), ("storage_block", 14)))
def test_malformed_linear_reflection_fails_before_shader_execution(field, value):
    source = layout_source()
    source["host_texture"][field] = value
    with pytest.raises(ValueError):
        ShaderLayout(source)
    with pytest.raises(ValueError):
        pack_texture(source, [texture_payload(2, 2)])


@pytest.mark.parametrize("field,value", (("min_filter", "nearest"), ("mag_filter", "nearest"),
                                        ("mip_filter", "linear"), ("wrap_s", "repeat"),
                                        ("wrap_t", "repeat"), ("normalized_coords", False)))
def test_linear_sampler_rejects_other_modes(field, value):
    texture = texture_payload(2, 2)
    texture["sampler"][field] = value
    with pytest.raises(ValueError, match="sampler"):
        pack_texture(layout_source(), [texture])


@pytest.fixture(scope="module")
def linear_program(backend_binary):
    result = subprocess.run([str(backend_binary), "texture", "4", "host-texture-linear"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("; mgc legacy-simulator backend, ABI=1, stage=MESA_SHADER_FRAGMENT")
    assert "hardware" not in result.stdout
    registers = int(re.search(r"registers=(\d+)", result.stderr).group(1))
    predicates = int(re.search(r"predicates=(\d+)", result.stderr).group(1))
    assert registers <= 32 and predicates <= 8
    words = assemble(result.stdout)
    assert assemble("\n".join(disassemble(words))) == words
    assert len(words) > 40
    return result.stdout


@pytest.mark.parametrize("width,height", ((1, 1), (1, 16), (16, 1), (3, 5), (16, 16)))
@pytest.mark.parametrize("horizontal,vertical", ((-10, -3), (0, 0), (1, 1), (4, 20),
                                                (0.125, 0.875), (0.5, 0.5),
                                                (0.03125, 0.96875), (0.37, 0.61)))
def test_fresh_compiler_uses_managed_fp16_bilinear_loads_with_bounded_edge_indices(
        linear_program, width, height, horizontal, vertical):
    texture = texture_payload(width, height)
    executor = ShaderExecutor(linear_program, ShaderLayout(layout_source()), texture=[texture])
    try:
        result = executor.invoke_fragment({0: pack_float(horizontal), 1: pack_float(vertical)})
        assert result.status is FragmentStatus.COMMITTED
        assert executor.invocations == 1 and executor.steps > 40
        actual = struct.unpack("<e", struct.pack("<H", result.outputs[0]))[0]
    finally:
        executor.close()
    expected = float(fp16_bilinear(texture, horizontal, vertical)[0])
    assert actual == pytest.approx(expected, abs=2 ** -10)
    if width == height == 1 or (horizontal <= 0 and vertical <= 0) or (horizontal >= 1 and vertical >= 1):
        assert result.outputs[0] == pack_float(expected)


def test_linear_managed_instruction_budget_is_failure_not_a_zero_color(linear_program):
    executor = ShaderExecutor(linear_program, ShaderLayout(layout_source()),
                              texture=[texture_payload(2, 2)], max_steps=1)
    try:
        with pytest.raises(ExecutionTimeoutError):
            executor.invoke_fragment({0: pack_float(0.5), 1: pack_float(0.5)})
        assert all(not buffer.in_use for buffer in executor.buffers.values())
    finally:
        executor.close()


@pytest.mark.parametrize("option,diagnostic", ((None, "texture sampling is not supported"),
                                              ("host-texture-unknown", "host texture version"),
                                              ("hardware-host-texture", "host texture version"),
                                              ("hardware-host-texture-linear", "host texture version")))
def test_default_unknown_and_both_hardware_texture_versions_fail_without_partial_assembly(
        backend_binary, option, diagnostic):
    command = [str(backend_binary), "texture", "4"]
    if option:
        command.append(option)
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and not result.stdout
    assert diagnostic in result.stderr and "error exposed partial assembly" not in result.stderr
