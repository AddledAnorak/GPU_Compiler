"""Fresh mirrored-repeat compiler execution and exact managed texture contracts."""

import copy
import json
import math
import os
import re
import struct
import subprocess

import numpy as np
import pytest

from test_backend_runtime import backend_binary
from test_linear_texture import layout_source, texture_payload
from tools.host_render import FragmentStatus, ShaderExecutor, pack_float
from tools.host_texture import (HOST_TEXTURE_LINEAR_MIRRORED_REPEAT_LAYOUT,
                               HOST_TEXTURE_MIRRORED_REPEAT_LAYOUT,
                               TEXTURE_BASE, host_texture_contract, pack_texture)
from tools.mgcasm import assemble, disassemble
from tools.mgcdriver import ExecutionTimeoutError, backends
from tools.shader_layout import ShaderLayout


def mirrored_texture(width, height, version):
    texture = texture_payload(width, height, 2 if version == 6 else 1)
    texture["sampler"].update(wrap_s="mirrored-repeat", wrap_t="mirrored-repeat")
    return texture


def fp16_mirrored_sample(texture, horizontal, vertical, linear):
    dimensions = texture["width"], texture["height"]
    neighbors = []
    weights = []
    for coordinate, size in zip((horizontal, vertical), dimensions):
        quantized = np.float16(coordinate)
        periods = np.float16(np.floor(np.float16(quantized * np.float16(0.5))))
        remainder = np.float16(quantized - np.float16(periods * np.float16(2)))
        folded = np.float16(min(remainder, np.float16(np.float16(2) - remainder)))
        normalized = np.clip(folded, np.float16(0), np.float16(1))
        scaled = np.float16(normalized * np.float16(size))
        if linear:
            position = np.float16(scaled + np.float16(-0.5))
            lower = math.floor(float(position))
            neighbors.append((min(max(lower, 0), size - 1), min(max(lower + 1, 0), size - 1)))
            weights.append(np.float16(position - np.float16(lower)))
        else:
            neighbors.append((min(math.floor(float(scaled)), size - 1),))
    channels = np.asarray(list(bytes.fromhex(texture["data_hex"])), dtype=np.float64)
    texels = (channels / 255).astype(np.float16).reshape(dimensions[1], dimensions[0], 4)
    if not linear:
        return texels[neighbors[1][0], neighbors[0][0]]
    sampled = []
    for vertical_index in neighbors[1]:
        lower, higher = (texels[vertical_index, index] for index in neighbors[0])
        sampled.append((lower + ((higher - lower).astype(np.float16) * weights[0]).astype(np.float16))
                       .astype(np.float16))
    return (sampled[0] + ((sampled[1] - sampled[0]).astype(np.float16) * weights[1]).astype(np.float16))


@pytest.mark.parametrize("version,expected", ((5, HOST_TEXTURE_MIRRORED_REPEAT_LAYOUT),
                                             (6, HOST_TEXTURE_LINEAR_MIRRORED_REPEAT_LAYOUT)))
def test_mirrored_contract_is_independent_and_preserves_existing_versions(version, expected):
    contract = host_texture_contract(version)
    assert contract == expected
    contract["wrap"] = "unsupported"
    assert host_texture_contract(version) == expected
    assert [host_texture_contract(previous)["wrap"] for previous in range(1, 5)] == [
        "clamp-to-edge", "clamp-to-edge", "repeat", "repeat"]


@pytest.mark.parametrize("version", (5, 6))
@pytest.mark.parametrize("width,height", ((1, 1), (1, 16), (16, 1), (3, 5), (5, 3), (16, 16)))
def test_mirrored_packing_preserves_exact_declared_input(version, width, height):
    texture = mirrored_texture(width, height, version)
    original = copy.deepcopy(texture)
    payload = pack_texture(ShaderLayout(layout_source(version)), [texture])
    assert texture == original
    assert set(payload) == set(range(TEXTURE_BASE, TEXTURE_BASE + 4 + width * height * 4))
    assert [payload[TEXTURE_BASE + index] for index in range(4)] == [pack_float(width), pack_float(height), 0, 0]
    assert [payload[TEXTURE_BASE + 4 + index] for index in range(width * height * 4)] == [
        pack_float(channel / 255) for channel in bytes.fromhex(texture["data_hex"])]


@pytest.mark.parametrize("version", (5, 6))
@pytest.mark.parametrize("field,value", (("wrap_s", "repeat"), ("wrap_t", "clamp-to-edge"),
                                       ("wrap_s", "mirrored-repeat "), ("normalized_coords", False),
                                       ("mip_filter", "linear"), ("min_filter", "unsupported"),
                                       ("mag_filter", "unsupported")))
def test_mirrored_sampler_mismatch_is_fail_closed(version, field, value):
    texture = mirrored_texture(3, 5, version)
    texture["sampler"][field] = value
    with pytest.raises(ValueError, match="sampler"):
        pack_texture(layout_source(version), [texture])


@pytest.mark.parametrize("version", (5, 6))
@pytest.mark.parametrize("field,value", (("wrap", "repeat"), ("wrap", "clamp-to-edge"),
                                       ("version", 11), ("version", True), ("filter", "unsupported"),
                                       ("max_width", 17), ("storage_block", 14)))
def test_mirrored_reflection_requires_exact_versioned_contract(version, field, value):
    source = layout_source(version)
    source["host_texture"][field] = value
    with pytest.raises(ValueError):
        ShaderLayout(source)
    with pytest.raises(ValueError):
        pack_texture(source, [mirrored_texture(2, 2, version)])


@pytest.fixture(scope="module", params=(5, 6))
def mirrored_program(backend_binary, request):
    version = request.param
    option = "host-texture-linear-mirrored-repeat" if version == 6 else "host-texture-mirrored-repeat"
    result = subprocess.run([str(backend_binary), "texture", "4", option], capture_output=True,
                            text=True, timeout=30, env={**os.environ, "MGC_TEST_LAYOUT": "1"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("; mgc legacy-simulator backend, ABI=1, stage=MESA_SHADER_FRAGMENT")
    assert int(re.search(r"registers=(\d+)", result.stderr).group(1)) <= 32
    assert int(re.search(r"predicates=(\d+)", result.stderr).group(1)) <= 8
    words = assemble(result.stdout)
    assert assemble("\n".join(disassemble(words))) == words
    source = json.loads(next(line.removeprefix("layout=") for line in result.stderr.splitlines()
                             if line.startswith("layout=")))
    layout = ShaderLayout(source)
    assert layout.host_texture == host_texture_contract(version)
    assert layout.target.value == "legacy-mgcsim-v1" and source["shader_abi_version"] == 1
    return version, result.stdout, layout


@pytest.mark.parametrize("width,height", ((1, 1), (1, 16), (16, 1), (3, 5), (5, 3), (16, 16)))
@pytest.mark.parametrize("horizontal,vertical", ((0, 0), (1, 1), (-1, -3), (2, 4), (3, 5),
                                               (-0.125, 1.875), (0.5, 0.5), (1.5, 2.5),
                                               (0.37, 0.61), (-1.37, 2.61), (-65504, 65504),
                                               (-2 ** -14, -2 ** -24), (2 ** -24, 1 - 2 ** -11)))
def test_fresh_mirror_shader_matches_fp16_and_bounds(mirrored_program, monkeypatch,
                                                   width, height, horizontal, vertical):
    version, assembly, layout = mirrored_program
    texture = mirrored_texture(width, height, version)
    uniform_base = layout.source["bases"]["uniform"]
    texture_start = uniform_base + TEXTURE_BASE
    texture_end = texture_start + 4 + width * height * 4
    loads = []
    original_load = backends._LegacyMemory.load

    def checked_load(memory, address):
        if uniform_base <= address < uniform_base + 65536:
            assert texture_start <= address < texture_end
            loads.append(address)
        return original_load(memory, address)

    monkeypatch.setattr(backends._LegacyMemory, "load", checked_load)
    executor = ShaderExecutor(assembly, layout, texture=[texture])
    try:
        result = executor.invoke_fragment({0: pack_float(horizontal), 1: pack_float(vertical)})
        assert result.status is FragmentStatus.COMMITTED
        actual = struct.unpack("<e", struct.pack("<H", result.outputs[0]))[0]
        expected = float(fp16_mirrored_sample(texture, horizontal, vertical, version == 6)[0])
        assert actual == pytest.approx(expected, abs=2 ** -10)
        assert executor.invocations == 1 and executor.steps > 20
        assert loads and texture_start in loads and texture_start + 1 in loads
    finally:
        executor.close()


def test_mirror_even_periods_and_reflections_match_but_odd_seams_differ(mirrored_program):
    version, assembly, layout = mirrored_program
    executor = ShaderExecutor(assembly, layout, texture=[mirrored_texture(3, 5, version)])
    try:
        equivalent = ((0.125, 0.625), (-0.125, -0.625), (2.125, 4.625), (1.875, 3.375))
        samples = [executor.invoke_fragment({0: pack_float(horizontal), 1: pack_float(vertical)}).outputs
                   for horizontal, vertical in equivalent]
        assert all(sample == samples[0] for sample in samples)
        edges = [executor.invoke_fragment({0: pack_float(seam), 1: pack_float(seam)}).outputs
                 for seam in (0, 1, 2, 3)]
        assert edges[0] == edges[2] and edges[1] == edges[3] and edges[0] != edges[1]
    finally:
        executor.close()


def test_mirror_budget_failure_is_not_output_commit(mirrored_program):
    version, assembly, layout = mirrored_program
    executor = ShaderExecutor(assembly, layout, texture=[mirrored_texture(2, 2, version)], max_steps=1)
    try:
        with pytest.raises(ExecutionTimeoutError):
            executor.invoke_fragment({0: pack_float(-0.5), 1: pack_float(1.5)})
        assert all(not buffer.in_use for buffer in executor.buffers.values())
    finally:
        executor.close()


@pytest.mark.parametrize("option", ("hardware-host-texture-mirrored-repeat",
                                    "hardware-host-texture-linear-mirrored-repeat", "host-texture-unknown"))
def test_mirrored_versions_reject_hardware_and_unknown_without_partial_program(backend_binary, option):
    result = subprocess.run([str(backend_binary), "texture", "4", option], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 1 and not result.stdout
    assert "host texture version" in result.stderr
