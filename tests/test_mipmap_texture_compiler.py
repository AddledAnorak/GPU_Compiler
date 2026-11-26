"""Explicit source-constant LOD -> fresh NIR -> selected uploaded mip execution."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from test_backend_runtime import backend_binary
from test_mixed_texture_compiler import (FRAMEBUFFER, MIXED_OPTIONS, TEXTURE_BASE, UNIFORM, WRAPS,
                                         compile_texture, fp16_sample, sample_cpu)
from tools.host_texture import SAMPLER_STATE, pack_texture
from tools.mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from tools.mgcsim.cpu import MgcSimError


MIPMAP_OPTIONS = {9: "host-texture-mipmap", 10: "host-texture-linear-mipmap"}
CONFIGURATIONS = [(version, lod, wrap_s, wrap_t) for version in MIPMAP_OPTIONS for lod in range(5)
                  for wrap_s in WRAPS for wrap_t in WRAPS]


def compile_mipmap(binary, version, lod, *, mode="texture-lod", wraps=None, success=True,
                   extra_environment=None):
    environment = {"MGC_TEST_SOURCE_LOD": str(lod), **(extra_environment or {})}
    return compile_texture(binary, MIPMAP_OPTIONS[version], mode=mode, count=0, wraps=wraps,
                           extra_environment=environment, success=success)


@pytest.fixture(scope="module", params=CONFIGURATIONS,
                ids=[f"v{version}-lod{lod}-{wrap_s}-{wrap_t}"
                     for version, lod, wrap_s, wrap_t in CONFIGURATIONS])
def mipmap_program(backend_binary, request):
    version, lod, wrap_s, wrap_t = request.param
    program = compile_mipmap(backend_binary, version, lod, wraps=(wrap_s, wrap_t))
    program.update(version=version, lod=lod, wrap_s=wrap_s, wrap_t=wrap_t)
    return program


def uploaded_chain(program, width, height):
    levels = []
    while True:
        level = len(levels)
        channels = bytes(channel % 256 for row in range(height) for column in range(width)
                         for channel in (level * 47 + column * 29 + row * 53 + 11,
                                         level * 31 + column * 71 + row * 17 + 37,
                                         level * 59 + column * 13 + row * 97 + 19,
                                         level * 23 + column * 7 + row * 11 + 151))
        levels.append({"level": level, "width": width, "height": height, "data_hex": channels.hex()})
        if width == height == 1:
            break
        width, height = max(width // 2, 1), max(height // 2, 1)
    sampler_filter = "linear" if program["version"] == 10 else "nearest"
    texture = {"width": levels[0]["width"], "height": levels[0]["height"],
               "format": "R8G8B8A8_UNORM", "data_hex": levels[0]["data_hex"],
               "sampler": {**SAMPLER_STATE, "min_filter": sampler_filter, "mag_filter": sampler_filter,
                           "mip_filter": "nearest", "wrap_s": program["wrap_s"],
                           "wrap_t": program["wrap_t"]},
               "mipmaps": levels[1:]}
    return texture, levels


def selected_storage(program, width, height):
    texture, levels = uploaded_chain(program, width, height)
    selected = levels[min(program["lod"], len(levels) - 1)]
    texels = (np.frombuffer(bytes.fromhex(selected["data_hex"]), dtype=np.uint8).astype(np.float64) /
              255).astype(np.float16).reshape(selected["height"], selected["width"], 4)
    original = copy.deepcopy(texture)
    payload = pack_texture(SimpleNamespace(source=program["layout"]), [texture])
    assert texture == original
    base = TEXTURE_BASE - UNIFORM
    assert sorted(payload) == list(range(base, base + 4 + texels.size))
    assert payload[base] == float_to_fp16_bits(selected["width"])
    assert payload[base + 1] == float_to_fp16_bits(selected["height"])
    assert payload[base + 2] == payload[base + 3] == 0
    assert [payload[base + 4 + index] for index in range(texels.size)] == [
        float_to_fp16_bits(float(channel)) for channel in texels.flat]
    return selected, texels, payload


def test_exact_source_lod_metadata_and_existing_abi(mipmap_program):
    layout = mipmap_program["layout"]
    assert layout["version"] == 2 and layout["shader_abi_version"] == 1
    assert layout["program_format"] == "legacy-mgcsim-v1" and layout["stage"] == "fragment"
    assert layout["supported"] is True and layout["errors"] == []
    assert layout["host_texture"] == {
        "version": mipmap_program["version"], "texture": 0, "sampler": 0, "storage_block": 15,
        "max_width": 16, "max_height": 16,
        "filter": "linear" if mipmap_program["version"] == 10 else "nearest",
        "wrap_s": mipmap_program["wrap_s"], "wrap_t": mipmap_program["wrap_t"],
        "lod": mipmap_program["lod"], "mip_filter": "nearest", "max_levels": 5,
    }
    assert list(layout["host_texture"]) == ["version", "texture", "sampler", "storage_block",
                                            "max_width", "max_height", "filter", "wrap_s", "wrap_t",
                                            "lod", "mip_filter", "max_levels"]
    assert mipmap_program["stats"]["host_texture_lod"] == mipmap_program["lod"]
    assert layout["uniforms"] == []
    assert [entry["word_offset"] for entry in layout["inputs"]] == [0, 1]
    assert [entry["word_offset"] for entry in layout["outputs"]] == [0, 1, 2, 3]
    assert 0 < mipmap_program["stats"]["registers"] <= 32
    assert 0 <= mipmap_program["stats"]["predicates"] <= 8


@pytest.mark.parametrize("width,height", ((1, 1), (1, 16), (16, 1), (3, 5), (5, 3), (16, 16)))
def test_all_filters_axes_lods_sample_only_selected_uploaded_level(mipmap_program, width, height):
    selected, texels, payload = selected_storage(mipmap_program, width, height)
    coordinates = ((0, 0), (1, 1), (-1, -3), (2, 4), (-0.125, 1.875), (1.875, -0.125),
                   (0.37, 0.61), (-2 ** -14, -2 ** -24))
    for coordinate in coordinates:
        cpu, _, reads, writes = sample_cpu(mipmap_program, selected["width"], selected["height"],
                                           coordinate, texels=texels, payload=payload)
        cpu.run(mipmap_program["words"])
        assert cpu.halted and cpu.cycles < cpu.max_cycles
        assert sorted(writes) == list(range(FRAMEBUFFER, FRAMEBUFFER + 4))
        expected = fp16_sample(texels, coordinate,
                              (mipmap_program["wrap_s"], mipmap_program["wrap_t"]),
                              mipmap_program["version"] == 10)
        actual_bits = [cpu.mem._framebuffer[index] for index in range(4)]
        if mipmap_program["version"] == 9:
            assert actual_bits == [float_to_fp16_bits(float(channel)) for channel in expected]
        else:
            np.testing.assert_allclose([fp16_bits_to_float(channel) for channel in actual_bits],
                                       expected.astype(np.float64), rtol=0, atol=2 ** -10)
        assert TEXTURE_BASE in reads and TEXTURE_BASE + 1 in reads
        assert len([address for address in reads if address >= TEXTURE_BASE + 4]) == (
            16 if mipmap_program["version"] == 10 else 4)


def test_checked_lod_disappears_without_new_sampling_instructions(mipmap_program, backend_binary):
    mixed = compile_texture(backend_binary, MIXED_OPTIONS[8 if mipmap_program["version"] == 10 else 7],
                            wraps=(mipmap_program["wrap_s"], mipmap_program["wrap_t"]))
    assert mipmap_program["assembly"] == mixed["assembly"]
    assert {key: value for key, value in mipmap_program["stats"].items() if key != "host_texture_lod"} == {
        key: value for key, value in mixed["stats"].items() if key != "host_texture_lod"}


def test_mipmap_step_budget_never_commits_partial_output(mipmap_program):
    selected, texels, payload = selected_storage(mipmap_program, 3, 5)
    cpu, _, _, writes = sample_cpu(mipmap_program, selected["width"], selected["height"], (-0.125, 1.875),
                                   texels=texels, payload=payload, max_cycles=1)
    with pytest.raises(MgcSimError, match="exceeded max_cycles=1"):
        cpu.run(mipmap_program["words"])
    assert not cpu.halted and writes == [] and cpu.mem.dump_framebuffer() == {}


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("lod", ("-1", "-0.000000059604644775390625", "0.5", "1.5", "4.000000476837158",
                                "1.0000001192092896", "5", "65504", "nan", "inf", "-inf"))
def test_nonfinite_fractional_negative_and_large_lods_reject_without_assembly(backend_binary, version,
                                                                          lod):
    diagnostic = compile_mipmap(backend_binary, version, lod, success=False)
    assert "host mipmap LOD must be a finite exact integer in 0..4" in diagnostic


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("mode", ("texture-lod-dynamic", "texture-lod-uniform", "texture-lod-vector",
                                 "texture-lod-16", "texture-lod-64"))
def test_only_scalar_32bit_source_constants_are_accepted(backend_binary, version, mode):
    diagnostic = compile_mipmap(backend_binary, version, 2, mode=mode, success=False)
    assert "host mipmap LOD must be a scalar float constant integer in 0..4" in diagnostic


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("mode", ("texture-rgba", "texture-grad", "texture-bias", "texture-lod-missing",
                                 "texture-lod-extra", "texture-lod-vertex"))
def test_mipmap_profile_never_relabels_implicit_bias_or_gradient_samples(backend_binary, version, mode):
    diagnostic = compile_mipmap(backend_binary, version, 2, mode=mode, success=False)
    assert "supports one direct 2D float fragment sample only" in diagnostic


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("lod", range(5))
def test_lod_and_coordinate_source_order_are_independent(backend_binary, version, lod):
    normal = compile_mipmap(backend_binary, version, lod, wraps=("repeat", "mirrored-repeat"))
    reversed_sources = compile_mipmap(backend_binary, version, lod, mode="texture-lod-reversed",
                                      wraps=("repeat", "mirrored-repeat"))
    assert reversed_sources == normal


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("lod", (0, 2, 4))
def test_environment_cannot_override_recorded_source_lod(backend_binary, version, lod):
    program = compile_mipmap(backend_binary, version, lod,
                             extra_environment={"MGC_HOST_TEXTURE_LOD": "1", "MGC_TEXTURE_LOD": "3"})
    assert program["stats"]["host_texture_lod"] == lod
    assert program["layout"]["host_texture"]["lod"] == lod


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("axis", ("s", "t"))
@pytest.mark.parametrize("value", (3, 99, -1))
def test_mipmap_profile_preserves_unknown_axis_enum_rejection(backend_binary, version, axis, value):
    diagnostic = compile_texture(backend_binary, MIPMAP_OPTIONS[version] + "-invalid-" + axis,
                                 mode="texture-lod", count=value,
                                 extra_environment={"MGC_TEST_SOURCE_LOD": "2"}, success=False)
    assert f"host texture wrap_{axis} is unsupported" in diagnostic


@pytest.mark.parametrize("option", (None, "hardware", "host-texture", "host-texture-linear",
                                   "host-texture-repeat", "host-texture-linear-repeat",
                                   "host-texture-mirrored-repeat", "host-texture-linear-mirrored-repeat",
                                   "host-texture-mixed", "host-texture-linear-mixed",
                                   "hardware-host-texture-mipmap", "hardware-host-texture-linear-mipmap"))
def test_old_default_and_hardware_profiles_fail_closed_for_explicit_lod(backend_binary, option):
    compile_texture(backend_binary, option, mode="texture-lod",
                    extra_environment={"MGC_TEST_SOURCE_LOD": "2"}, success=False)


@pytest.mark.parametrize("version", (9, 10))
def test_mipmap_profile_keeps_default_clamp_axes(backend_binary, version):
    program = compile_mipmap(backend_binary, version, 3)
    assert program["layout"]["host_texture"]["wrap_s"] == WRAPS[0]
    assert program["layout"]["host_texture"]["wrap_t"] == WRAPS[0]


def test_selected_storage_contract_explicitly_rejects_trilinear_state(mipmap_program):
    texture, _ = uploaded_chain(mipmap_program, 3, 5)
    texture["sampler"]["mip_filter"] = "linear"
    with pytest.raises(ValueError, match="sampler"):
        pack_texture(SimpleNamespace(source=mipmap_program["layout"]), [texture])
