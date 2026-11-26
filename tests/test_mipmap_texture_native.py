"""Explicit shader LOD contracts, independent mip oracles and fresh actual GL readback."""

import copy
from dataclasses import replace
from fractions import Fraction
import inspect
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from tools import mipmap_texture_acceptance as acceptance
from tools import run_render
from tools.host_texture import TEXTURE_BASE, host_texture_contract, pack_texture
from tools.linear_texture_acceptance import rational_sample as clamp_sample
from tools.mgcdriver import ProgramFormat
from tools.mirrored_texture_acceptance import rational_sample as mirrored_sample
from tools.repeat_texture_acceptance import rational_sample as repeat_sample
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ABI_BASES, ROOT
from tools.shader_layout import ShaderLayout


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
HARNESS = Path(os.environ.get("MGC_MIPMAP_TEST_HARNESS", ROOT / "tools/mgc_harness"))
FIXTURE = ROOT / "tests/fixtures/host-texture"
HELPERS = (capture_draw, softpipe_reference, native_host_readback)
FLAGS = ("MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_DATA", "MGC_HOST_TEXTURE_FILTER", "MGC_HOST_TEXTURE_WRAP",
         "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T", "MGC_HOST_CAPTURE_DIR", "MGC_ASM_DUMP",
         "MGC_FRAGMENT_STATUS", "MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST",
         "MGC_HOST_BLEND", "MGC_HOST_CLEAR_RGBA", "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT", "MGC_HOST_DRAW_REPEATS",
         "MGC_HOST_GLSL_PROFILE")


class InterceptedDraw(Exception):
    pass


class NonexactString(str):
    pass


def helper_options(helper, texture=None):
    options = {"texture": texture}
    if helper is capture_draw:
        options["host_texture"] = texture is not None
    if helper is native_host_readback:
        options["socket_path"] = Path("/tmp/mgc-mipmap-test.sock")
    return options


def artifact_fixture(case, levels):
    common = {"version": 1, "supported": True, "program_format": ProgramFormat.LEGACY.value,
              "shader_abi_version": 1, "address_units": "words", "bases": dict(ABI_BASES),
              "inputs": [], "outputs": [], "uniforms": []}
    vertex = {**common, "stage": "vertex"}
    fragment = {**common, "stage": "fragment", "version": 2, "host_texture": acceptance.texture_contract(case),
                "fragment_status": dict(acceptance.STATUS), "uses_discard": case.variant != "sample"}
    return {"schema": "mgc-host-draw-v2", "program_format": ProgramFormat.LEGACY.value, "shader_abi_version": 1,
            "shaders": {"vs": {"layout": vertex, "assembly": "mov r0, 0\n"},
                        "fs": {"layout": fragment, "assembly": "mgc_block_0:\nmov r0, 0\n"}},
            "textures": [acceptance.captured_texture(levels, case)]}


@pytest.mark.parametrize("width,height,dimensions", (
    (1, 1, ((1, 1),)), (3, 5, ((3, 5), (1, 2), (1, 1))),
    (5, 3, ((5, 3), (2, 1), (1, 1))), (16, 16, ((16, 16), (8, 8), (4, 4), (2, 2), (1, 1))),
    (1, 16, ((1, 16), (1, 8), (1, 4), (1, 2), (1, 1))),
    (16, 1, ((16, 1), (8, 1), (4, 1), (2, 1), (1, 1))),
))
def test_complete_floor_half_chain_and_exact_little_endian_file(width, height, dimensions):
    levels = acceptance.input_levels(width, height)
    originals = tuple(level.copy() for level in levels)
    contents = acceptance.encode_mipmaps(levels)
    assert acceptance.MIPMAP_HEADER.size == 20 and len(acceptance.MIPMAP_MAGIC) == 8
    assert contents[:20] == b"MGCMIP1\0" + struct.pack("<III", width, height, len(dimensions))
    assert acceptance.mipmap_dimensions(width, height) == dimensions
    assert contents[20:] == b"".join(level.tobytes() for level in levels)
    assert len(contents) == 20 + sum(level_width * level_height * 4 for level_width, level_height in dimensions)
    decoded = acceptance.decode_mipmaps(contents)
    assert all(np.array_equal(first, second) for first, second in zip(decoded, levels))
    assert all(not np.shares_memory(first, second) for first, second in zip(decoded, levels))
    assert all(np.array_equal(first, second) for first, second in zip(originals, levels))
    if width == height == 16:
        assert len(contents) == acceptance.MAX_MIPMAP_BYTES == 1384


@pytest.mark.parametrize("width,height", ((1, 1), (3, 5), (5, 3), (16, 16)))
def test_independent_patterns_are_not_downsampled_or_aliased(width, height):
    levels = acceptance.input_levels(width, height)
    assert all(level.dtype == np.uint8 and level.flags.c_contiguous for level in levels)
    assert all(not np.shares_memory(first, second) for first, second in itertools.combinations(levels, 2))
    assert len({tuple(int(channel) for channel in level[0, 0]) for level in levels}) == len(levels)
    if len(levels) > 1:
        for level, previous in zip(levels[1:], levels):
            assert not np.array_equal(level[0, 0], previous[:2, :2].mean(axis=(0, 1)).round().astype(np.uint8))


@pytest.mark.parametrize("dimensions", ((0, 1), (17, 1), (1, -1), (True, 1), (1, 1.0), ("3", 5)))
def test_dimensions_reject_out_of_bounds_and_nonexact_types(dimensions):
    with pytest.raises(ValueError):
        acceptance.input_levels(*dimensions)


@pytest.mark.parametrize("invalid", (b"", b"MGCMIP1\0", b"\0" * 20, b"\0" * 1385,
                                    bytearray(b"\0" * 24), "MGCMIP1"))
def test_mipmap_decoder_rejects_invalid_header_size_and_type(invalid):
    with pytest.raises(ValueError):
        acceptance.decode_mipmaps(invalid)


def malformed_mipmaps():
    valid = acceptance.encode_mipmaps(acceptance.input_levels(3, 5))
    return (
        b"MGCMIP2\0" + valid[8:], valid[:-1], valid + b"\0", valid[:20] + valid[20:80],
        b"MGCMIP1\0" + struct.pack("<III", 0, 5, 3) + valid[20:],
        b"MGCMIP1\0" + struct.pack("<III", 17, 5, 3) + valid[20:],
        b"MGCMIP1\0" + struct.pack("<III", 3, 5, 0) + valid[20:],
        b"MGCMIP1\0" + struct.pack("<III", 3, 5, 2) + valid[20:],
        b"MGCMIP1\0" + struct.pack("<III", 3, 5, 6) + valid[20:],
        struct.pack("<II", 3, 5) + valid[20:],
        b"MGCMIP1\0" + struct.pack("<III", 3, 5, 3) + bytes(4 * (15 + 6 + 1)),
    )


@pytest.mark.parametrize("contents", malformed_mipmaps())
def test_mipmap_decoder_rejects_incomplete_oversized_or_trailing_levels(contents):
    with pytest.raises(ValueError):
        acceptance.decode_mipmaps(contents)


def test_validate_entire_chain_before_encoding_or_selected_oracle():
    levels = acceptance.input_levels(16, 16)
    invalid_chains = ((), levels[:-1], levels + (levels[-1],), (levels[0], levels[2], *levels[2:]),
                      (levels[0], levels[1].astype(np.float32), *levels[2:]),
                      (levels[0], levels[1][:, ::-1], *levels[2:]),
                      (levels[0], levels[1][:, :, :3], *levels[2:]))
    for invalid in invalid_chains:
        with pytest.raises(ValueError):
            acceptance.encode_mipmaps(invalid)
        with pytest.raises(ValueError):
            acceptance.rational_sample(invalid, 0, 0, lod=0)


@pytest.mark.parametrize("lod", (0, 1, 2, 3, 4))
@pytest.mark.parametrize("width,height", ((1, 1), (3, 5), (5, 3), (16, 16)))
def test_explicit_integer_lod_selects_uploaded_level_and_clamps_only_to_last(width, height, lod):
    levels = acceptance.input_levels(width, height)
    selected = min(lod, len(levels) - 1)
    assert acceptance.selected_level(levels, lod) == selected
    assert acceptance.rational_sample(levels, 0, 0, lod=lod) == tuple(Fraction(int(channel)) for channel in levels[selected][0, 0])


@pytest.mark.parametrize("lod", (-1, 5, True, False, 1.0, Fraction(1), "1", None))
def test_rational_lod_rejects_fractional_implicit_unbounded_and_nonexact_types(lod):
    with pytest.raises(ValueError):
        acceptance.rational_sample(acceptance.input_levels(3, 5), 0, 0, lod=lod)


@pytest.mark.parametrize("texture_filter", ("nearest", "linear"))
@pytest.mark.parametrize("wrap,sampler", (("clamp-to-edge", clamp_sample), ("repeat", repeat_sample),
                                         ("mirrored-repeat", mirrored_sample)))
@pytest.mark.parametrize("lod", (0, 1, 2, 3, 4))
def test_fraction_axis_oracle_matches_retained_wrap_math_at_selected_level(texture_filter, wrap, sampler, lod):
    levels = acceptance.input_levels(16, 16)
    selected = levels[lod]
    for horizontal, vertical in ((Fraction(-17, 16), Fraction(33, 16)), (0, 1), (1, 0),
                                  (Fraction(-1, 1 << 24), Fraction(1, 2)),
                                  (Fraction(5, 4), Fraction(-7, 4))):
        assert acceptance.rational_sample(levels, horizontal, vertical, texture_filter, lod=lod,
                                           texture_wrap_s=wrap, texture_wrap_t=wrap) == sampler(
                                               selected, horizontal, vertical, texture_filter)


@pytest.mark.parametrize("texture_filter", ("nearest", "linear"))
@pytest.mark.parametrize("wrap_s,wrap_t", tuple(itertools.product(acceptance.WRAPS, repeat=2)))
def test_fraction_texel_centers_and_seams_have_independent_axes(texture_filter, wrap_s, wrap_t):
    levels = acceptance.input_levels(16, 16)
    texels = levels[3]
    assert acceptance.rational_sample(levels, Fraction(1, 4), Fraction(3, 4), texture_filter, lod=3,
                                      texture_wrap_s=wrap_s, texture_wrap_t=wrap_t) == tuple(Fraction(int(channel)) for channel in texels[1, 0])
    horizontal = Fraction(-1, 4) if texture_filter == "nearest" else 0
    expected_column = 1 if wrap_s == "repeat" else 0
    if texture_filter == "nearest":
        assert acceptance.rational_sample(levels, horizontal, Fraction(5, 4), texture_filter, lod=3,
                                          texture_wrap_s=wrap_s, texture_wrap_t=wrap_t) == tuple(
                                              Fraction(int(channel)) for channel in texels[0 if wrap_t == "repeat" else 1, expected_column])
    else:
        columns = ((0, Fraction(1, 2)), (1, Fraction(1, 2))) if wrap_s == "repeat" else ((0, Fraction(1)),)
        rows = ((0, Fraction(1, 2)), (1, Fraction(1, 2))) if wrap_t == "repeat" else ((1, Fraction(1)),)
        expected = tuple(sum(Fraction(int(texels[row, column, channel])) * row_weight * column_weight
                             for row, row_weight in rows for column, column_weight in columns) for channel in range(4))
        assert acceptance.rational_sample(levels, horizontal, 1, texture_filter, lod=3,
                                          texture_wrap_s=wrap_s, texture_wrap_t=wrap_t) == expected


def test_fraction_oracle_never_calls_packing_rendering_or_gl(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("independent mip oracle reached production packing, shader execution or GL")
    for name in ("pack_texture", "render_artifact", "capture_draw", "softpipe_reference", "native_host_readback"):
        monkeypatch.setattr(acceptance, name, forbidden)
    levels = acceptance.input_levels(16, 16)
    original = acceptance.encode_mipmaps(levels)
    image, coverage = acceptance.rational_oracle(levels, lod=2, phase="negative-seams", texture_filter="linear",
                                                texture_wrap_s="repeat", texture_wrap_t="mirrored-repeat")
    assert image.shape == (16, 16, 4) and image.dtype == np.uint8
    assert coverage.dtype == np.uint32 and np.all(coverage == 1)
    assert original == acceptance.encode_mipmaps(levels)


def test_selected_level_bilinear_uses_exact_four_tap_byte_unit_weights():
    levels = acceptance.input_levels(16, 16)
    levels[3][:] = [[[0, 255, 0, 0], [255, 0, 0, 0]], [[0, 0, 255, 127], [0, 0, 0, 255]]]
    channels = acceptance.rational_sample(levels, Fraction(1, 2), Fraction(1, 2), "linear", lod=3)
    assert channels == (Fraction(255, 4), Fraction(255, 4), Fraction(255, 4), Fraction(191, 2))
    assert tuple(round(channel) for channel in channels) == (64, 64, 64, 96)
    assert all(isinstance(channel, Fraction) for channel in channels)


@pytest.mark.parametrize("texture_filter", ("nearest", "linear"))
def test_transparent_selected_level_keeps_rgb_and_explicit_commit_coverage(texture_filter):
    levels = acceptance.input_levels(1, 1)
    image, coverage = acceptance.rational_oracle(levels, lod=4, texture_filter=texture_filter)
    assert np.all(image == [13, 127, 250, 0]) and np.all(coverage == 1)
    image, coverage = acceptance.rational_oracle(levels, lod=4, texture_filter=texture_filter, discard=True)
    assert np.all(image[:, :8] == 0) and np.all(image[:, 8:] == [13, 127, 250, 0])
    assert np.all(coverage[:, :8] == 0) and np.all(coverage[:, 8:] == 1)


@pytest.mark.parametrize("options", ({"discard": 1}, {"texture_filter": "LINEAR"}, {"texture_filter": True},
                                     {"texture_wrap_s": None}, {"texture_wrap_t": NonexactString("repeat")},
                                     {"phase": "implicit-derivatives"}))
def test_rational_oracle_rejects_unknown_or_noncanonical_options(options):
    with pytest.raises(ValueError):
        acceptance.rational_oracle(acceptance.input_levels(3, 5), **options)


@pytest.mark.parametrize("case", acceptance.CASES)
def test_selected_input_change_controls_expected_image_unselected_levels_do_not(case):
    levels = acceptance.input_levels(case.width, case.height)
    original = acceptance.encode_mipmaps(levels)
    options = {"lod": case.lod, "phase": case.phase, "texture_filter": case.texture_filter,
               "texture_wrap_s": case.texture_wrap_s, "texture_wrap_t": case.texture_wrap_t,
               "discard": case.variant != "sample"}
    image, coverage = acceptance.rational_oracle(levels, **options)
    selected = acceptance.selected_level(levels, case.lod)
    for level in range(len(levels)):
        changed = acceptance.changed_levels(levels, level)
        controlled, controlled_coverage = acceptance.rational_oracle(changed, **options)
        assert np.array_equal(coverage, controlled_coverage)
        if level == selected:
            assert acceptance.image_comparison(image, controlled, 0)["max_channel_error"] > 2
        else:
            assert np.array_equal(image, controlled)
        assert all(np.array_equal(first, second) == (index != level) for index, (first, second) in enumerate(zip(changed, levels)))
    assert acceptance.encode_mipmaps(levels) == original


def test_case_matrix_is_bounded_and_covers_filters_lods_asymmetry_completion_and_z16():
    assert len(acceptance.CASES) == 26 and len(acceptance.DEFAULT_CASES) == 10
    assert len({case.name for case in acceptance.CASES}) == 26
    assert acceptance.MAX_DRAW_STEPS == 2_000_000 and acceptance.MAX_CHANNEL_ERROR == 2
    for texture_filter in ("nearest", "linear"):
        cases = tuple(case for case in acceptance.CASES if case.texture_filter == texture_filter)
        assert {case.lod for case in cases} == set(range(5))
        assert {case.lod for case in acceptance.DEFAULT_CASES if case.texture_filter == texture_filter} == set(range(5))
        assert {(case.width, case.height) for case in cases} == {(1, 1), (3, 5), (5, 3), (16, 16)}
        assert {case.variant for case in cases} == {"sample", "early", "late", "runtime-loop"}
        assert {case.phase for case in cases} >= {"clamp", "negative-seams", "seams", "even-odd-seams"}
        assert any(case.depth_test and case.repeats == 2 for case in cases)


@pytest.mark.parametrize("case", acceptance.CASES)
def test_fresh_shader_requires_lod_extension_literal_lod_and_nonconstant_varying_uv(tmp_path, case):
    shader = acceptance._shader(case, tmp_path)
    fragment = (shader / "frag.glsl").read_text()
    vertex = (shader / "vert.glsl").read_text()
    assert fragment.startswith("#version 120\n#extension GL_ARB_shader_texture_lod : require\n")
    assert fragment.count(f"texture2DLod(u_texture, v_uv, {case.lod}.0)") == 1
    assert "texture2D(" not in fragment and "texture2DGrad" not in fragment
    assert "varying vec2 v_uv;" in fragment and "uniform sampler2D u_texture;" in fragment
    assert vertex.count("attribute vec4 a_position;") == 1
    assert "v_uv = (a_position.xy + vec2(1.0)) * 0.5;" in vertex
    assert ("discard;" in fragment) == (case.variant != "sample")


@pytest.mark.parametrize("texture_filter,version", (("nearest", 9), ("linear", 10)))
@pytest.mark.parametrize("lod", (0, 1, 2, 3, 4))
def test_exact_twelve_key_reflection_and_selected_input_ubo15_only(texture_filter, version, lod):
    case = replace(acceptance.CASES[0], texture_filter=texture_filter, lod=lod,
                   texture_wrap_s="repeat", texture_wrap_t="mirrored-repeat")
    levels = acceptance.input_levels(16, 16)
    artifact = artifact_fixture(case, levels)
    source = artifact["shaders"]["fs"]["layout"]
    contract = source["host_texture"]
    assert set(contract) == {"version", "texture", "sampler", "storage_block", "max_width", "max_height",
                             "filter", "wrap_s", "wrap_t", "lod", "mip_filter", "max_levels"}
    assert contract == host_texture_contract(version, wrap_s="repeat", wrap_t="mirrored-repeat", lod=lod)
    before = copy.deepcopy(artifact)
    acceptance.check_contract(artifact, levels, case)
    words = pack_texture(ShaderLayout(source), artifact["textures"])
    texels = levels[lod]
    scalars = np.array([texels.shape[1], texels.shape[0], 0, 0] +
                       [int(channel) / 255 for channel in texels.reshape(-1)], dtype=np.float16).view(np.uint16)
    assert words == {TEXTURE_BASE + index: int(value) for index, value in enumerate(scalars)}
    assert len(words) == 4 + texels.size and max(words) < TEXTURE_BASE + 4096
    assert artifact == before


@pytest.mark.parametrize("width,height", ((1, 1), (3, 5), (5, 3)))
def test_reflection_keeps_source_lod_when_uploaded_chain_clamps_to_last(width, height):
    case = replace(acceptance.CASES[0], width=width, height=height, lod=4)
    levels = acceptance.input_levels(width, height)
    artifact = artifact_fixture(case, levels)
    acceptance.check_contract(artifact, levels, case)
    assert artifact["shaders"]["fs"]["layout"]["host_texture"]["lod"] == 4
    words = pack_texture(artifact["shaders"]["fs"]["layout"], artifact["textures"])
    assert len(words) == 8
    assert words[TEXTURE_BASE] == words[TEXTURE_BASE + 1] == 0x3C00


@pytest.mark.parametrize("corruption", ("hex", "missing", "dimensions", "level_bool", "sampler"))
def test_packer_validates_unselected_levels_before_returning_base_level_words(corruption):
    case = acceptance.CASES[0]
    levels = acceptance.input_levels(16, 16)
    artifact = artifact_fixture(case, levels)
    textures = artifact["textures"]
    if corruption == "hex":
        textures[0]["mipmaps"][-1]["data_hex"] = "gg000000"
    elif corruption == "missing":
        textures[0]["mipmaps"].pop()
    elif corruption == "dimensions":
        textures[0]["mipmaps"][-1]["width"] = 2
    elif corruption == "level_bool":
        textures[0]["mipmaps"][0]["level"] = True
    else:
        textures[0]["sampler"]["mip_filter"] = "linear"
    with pytest.raises(ValueError):
        pack_texture(artifact["shaders"]["fs"]["layout"], textures)


@pytest.mark.parametrize("corruption", ("abi", "format", "layout_version", "lod_bool", "lod_wrong", "max_levels", "extra_metadata",
                                        "base_bytes", "level_bytes", "level_order", "level_dimensions", "extra_level_key", "mip_filter",
                                        "sampler_extra", "unselected_hex", "unselected_missing", "uniform_alias"))
def test_capture_rejects_metadata_abi_bytes_sampler_and_unselected_corruption(corruption):
    case = replace(acceptance.CASES[0], lod=0)
    levels = acceptance.input_levels(16, 16)
    artifact = artifact_fixture(case, levels)
    source = artifact["shaders"]["fs"]["layout"]
    texture = artifact["textures"][0]
    if corruption == "abi":
        artifact["shader_abi_version"] = True
    elif corruption == "format":
        texture["format"] = "B8G8R8A8_UNORM"
    elif corruption == "layout_version":
        source["version"] = 1
    elif corruption == "lod_bool":
        source["host_texture"]["lod"] = False
    elif corruption == "lod_wrong":
        source["host_texture"]["lod"] = 1
    elif corruption == "max_levels":
        source["host_texture"]["max_levels"] = 6
    elif corruption == "extra_metadata":
        source["host_texture"]["implicit_lod"] = True
    elif corruption == "base_bytes":
        texture["data_hex"] = "00" + texture["data_hex"][2:]
    elif corruption == "level_bytes":
        texture["mipmaps"][0]["data_hex"] = "00" + texture["mipmaps"][0]["data_hex"][2:]
    elif corruption == "level_order":
        texture["mipmaps"][0]["level"] = 2
    elif corruption == "level_dimensions":
        texture["mipmaps"][0]["width"] = 7
    elif corruption == "extra_level_key":
        texture["mipmaps"][0]["generated"] = True
    elif corruption == "mip_filter":
        texture["sampler"]["mip_filter"] = "linear"
    elif corruption == "sampler_extra":
        texture["sampler"]["lod_bias"] = 0
    elif corruption == "unselected_hex":
        texture["mipmaps"][-1]["data_hex"] = "not-hex!"
    elif corruption == "unselected_missing":
        texture["mipmaps"].pop()
    else:
        source["uniforms"] = [{"word_offset": TEXTURE_BASE, "type": "float", "bit_size": 32}]
    with pytest.raises(ValueError):
        acceptance.check_contract(artifact, levels, case)


@pytest.mark.parametrize("helper", HELPERS)
def test_mipmap_api_is_keyword_only_with_unchanged_legacy_defaults(helper):
    parameters = inspect.signature(helper).parameters
    assert parameters["mipmaps"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["mipmaps"].default is False
    assert parameters["texture_filter"].default == "nearest"
    assert parameters["texture_wrap"].default == "clamp-to-edge"
    for name in ("texture_wrap_s", "texture_wrap_t"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY and parameters[name].default is None


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("invalid", (0, 1, None, "true", [], {}, np.bool_(True)))
def test_api_rejects_nonboolean_mipmaps_before_gl(monkeypatch, tmp_path, helper, invalid):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid mipmap API reached GL"))
    with pytest.raises(ValueError):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper), mipmaps=invalid)


@pytest.mark.parametrize("helper", HELPERS)
def test_mipmaps_without_an_explicit_texture_fail_before_gl(monkeypatch, tmp_path, helper):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("missing mip texture reached GL"))
    with pytest.raises(ValueError):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper), mipmaps=True)


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("option,invalid", (("texture_filter", "trilinear"), ("texture_filter", NonexactString("linear")),
                                          ("texture_wrap_s", "mirror"), ("texture_wrap_t", "REPEAT"),
                                          ("texture_wrap_s", True), ("texture_wrap_t", NonexactString("repeat"))))
def test_api_rejects_unknown_noncanonical_filter_or_axis_before_gl(monkeypatch, tmp_path, helper, option, invalid):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid mip sampler reached GL"))
    texture = tmp_path / "texture.bin"
    texture.write_bytes(acceptance.encode_mipmaps(acceptance.input_levels(3, 5)))
    with pytest.raises(ValueError):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper, texture), mipmaps=True, **{option: invalid})


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("unknown", ({"lod": 1}, {"texture_lod": 2}, {"mip_filter": "linear"}))
def test_api_cannot_override_shader_lod_or_request_unknown_mip_modes(monkeypatch, tmp_path, helper, unknown):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("unknown mip option reached GL"))
    with pytest.raises(TypeError):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper), mipmaps=True, **unknown)


@pytest.mark.parametrize("texture_filter", ("nearest", "linear"))
def test_hardware_mipmaps_rejected_before_gl(monkeypatch, tmp_path, texture_filter):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("hardware mip texture reached GL"))
    texture = tmp_path / "texture.bin"
    texture.write_bytes(acceptance.encode_mipmaps(acceptance.input_levels(3, 5)))
    with pytest.raises(ValueError):
        capture_draw(FIXTURE, BUILD, tmp_path, target=ProgramFormat.HARDWARE, host_texture=True, texture=texture,
                     texture_filter=texture_filter, mipmaps=True)


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("texture_filter", ("nearest", "linear"))
@pytest.mark.parametrize("wrap_s,wrap_t", (("clamp-to-edge", "clamp-to-edge"), ("repeat", "mirrored-repeat"),
                                         ("mirrored-repeat", "clamp-to-edge")))
def test_api_emits_v9_v10_same_input_for_actual_mgc_and_softpipe(monkeypatch, tmp_path, helper, texture_filter, wrap_s, wrap_t):
    for name in FLAGS:
        monkeypatch.setenv(name, "inherited-invalid")
    texture = tmp_path / "texture.bin"
    contents = acceptance.encode_mipmaps(acceptance.input_levels(3, 5))
    texture.write_bytes(contents)
    def intercepted(command, *, env, **kwargs):
        assert env["GALLIUM_DRIVER"] == ("softpipe" if helper is softpipe_reference else "mgc")
        assert env["MGC_HOST_TEXTURE"] == ("10" if texture_filter == "linear" else "9")
        assert env["MGC_HOST_TEXTURE_DATA"] == str(texture)
        assert env["MGC_HOST_TEXTURE_FILTER"] == texture_filter
        assert env["MGC_HOST_GLSL_PROFILE"] == "desktop-120"
        assert env.get("MGC_HOST_TEXTURE_WRAP_S", env.get("MGC_HOST_TEXTURE_WRAP")) == wrap_s
        assert env.get("MGC_HOST_TEXTURE_WRAP_T", env.get("MGC_HOST_TEXTURE_WRAP")) == wrap_t
        assert all(env.get(name) != "inherited-invalid" for name in FLAGS)
        assert command[3] == {capture_draw: "--draw", softpipe_reference: "--reference", native_host_readback: "--host-execute"}[helper]
        assert texture.read_bytes() == contents
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper, texture), mipmaps=True,
               texture_filter=texture_filter, texture_wrap_s=wrap_s, texture_wrap_t=wrap_t)


@pytest.mark.parametrize("helper", HELPERS)
def test_default_nonmip_api_does_not_opt_into_new_versions(monkeypatch, tmp_path, helper):
    for name in FLAGS:
        monkeypatch.setenv(name, "inherited-invalid")
    def intercepted(command, *, env, **kwargs):
        assert "MGC_HOST_TEXTURE" not in env and "MGC_HOST_TEXTURE_DATA" not in env
        assert "MGC_HOST_GLSL_PROFILE" not in env
        assert "MGC_HOST_TEXTURE_WRAP_S" not in env and "MGC_HOST_TEXTURE_WRAP_T" not in env
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper))


@pytest.mark.parametrize("arguments", (("--mipmaps",), ("--target", "hardware", "--mipmaps"),
                                       ("--texture-filter", "trilinear"), ("--texture-wrap-s", "mirror"),
                                       ("--texture-lod", "2")))
def test_render_cli_malformed_default_hardware_and_unknowns_do_not_reach_gl(monkeypatch, tmp_path, arguments):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid render CLI reached GL"))
    argv = [str(FIXTURE), "--output", str(tmp_path), *arguments]
    try:
        result = run_render.main(argv)
    except SystemExit as error:
        assert error.code == 2
    else:
        assert result == 1


@pytest.mark.parametrize("cases", ((), (acceptance.CASES[0], acceptance.CASES[0]),
                                   (replace(acceptance.CASES[0], name="unregistered"),)))
def test_invalid_acceptance_selection_never_reaches_gl(monkeypatch, tmp_path, cases):
    monkeypatch.setattr(acceptance, "_run_case", lambda *args, **kwargs: pytest.fail("invalid selection reached GL"))
    with pytest.raises(ValueError, match="distinct named bounded"):
        acceptance.validate(tmp_path, BUILD, cases=cases)
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("mode", ("default", "explicit", "extended"))
def test_acceptance_cli_build_output_cases_and_default_subset(monkeypatch, tmp_path, capsys, mode):
    seen = {}
    def intercepted(output, build, *, cases):
        seen.update(output=output, build=build, cases=cases)
        return {"tested_cases": [case.name for case in cases], "maximum_channel_error": 0}
    monkeypatch.setattr(acceptance, "validate", intercepted)
    arguments = ["--build", str(BUILD), "--output", str(tmp_path)]
    cases = ((acceptance.CASES[2], acceptance.CASES[7]) if mode == "explicit" else
             acceptance.CASES if mode == "extended" else acceptance.DEFAULT_CASES)
    if mode == "explicit":
        arguments += ["--cases", *[case.name for case in cases]]
    elif mode == "extended":
        arguments += ["--extended"]
    assert acceptance.main(arguments) == 0
    assert seen == {"output": tmp_path, "build": BUILD, "cases": cases}
    printed = json.loads(capsys.readouterr().out)
    assert printed["tested_cases"] == [case.name for case in cases]
    assert printed["summary"] == str(tmp_path / "summary.json") and printed["fpga_execution"] is False


def test_unavailable_extension_is_flagged_without_fake_native_results(monkeypatch, tmp_path):
    def unavailable(directory, build, case):
        directory.mkdir(parents=True)
        (directory / "capture.log").write_text("GL_ARB_shader_texture_lod not advertised by actual GL\n")
        raise ValueError("fresh Gallium draw failed")
    monkeypatch.setattr(acceptance, "_run_case", unavailable)
    with pytest.raises(ValueError):
        acceptance.validate(tmp_path, BUILD, cases=(acceptance.CASES[0],))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "unavailable" and summary["extension_available"] is False
    assert summary["passed"] is False and summary["native_gl_readback"] is False
    assert summary["fpga_execution"] is False and summary["cases"] == []
    assert not list(tmp_path.rglob("*.rgba8"))


def test_acceptance_execution_error_records_failure_and_does_not_claim_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(acceptance, "_run_case", lambda *args: (_ for _ in ()).throw(ValueError("budget exceeded")))
    with pytest.raises(ValueError, match="budget exceeded"):
        acceptance.validate(tmp_path, BUILD, cases=(acceptance.CASES[0],))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "error" and summary["passed"] is False and summary["native_gl_readback"] is False
    assert summary["extension_available"] is None and summary["aggregate_draw_step_budget"] == 2_000_000


def require_harness():
    if not HARNESS.is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("rebuild Mesa mipmap lowering/capture and the actual GL harness first")


@pytest.mark.parametrize("driver", ("mgc", "softpipe"))
@pytest.mark.parametrize("contents", malformed_mipmaps())
def test_actual_harness_rejects_malformed_files_without_capture_or_native_output(tmp_path, driver, contents):
    require_harness()
    shader = acceptance._shader(acceptance.CASES[0], tmp_path)
    environment = run_render.mesa_environment(BUILD, tmp_path)
    for name in FLAGS:
        environment.pop(name, None)
    texture = tmp_path / "texture.bin"
    texture.write_bytes(contents)
    captures = tmp_path / "capture"
    captures.mkdir()
    raw = tmp_path / "native.rgba8"
    environment.update(GALLIUM_DRIVER=driver, MGC_HOST_TEXTURE="9", MGC_HOST_TEXTURE_DATA=str(texture),
                       MGC_HOST_TEXTURE_FILTER="nearest", MGC_HOST_TEXTURE_WRAP_S="repeat",
                       MGC_HOST_TEXTURE_WRAP_T="clamp-to-edge", MGC_HOST_CAPTURE_DIR=str(captures),
                       MGC_HOST_GLSL_PROFILE="desktop-120")
    if driver == "mgc":
        environment["MGC_HOST_EXECUTOR_SOCKET"] = str(tmp_path / "unused-executor.sock")
    completed = subprocess.run([str(HARNESS), str(shader / "vert.glsl"), str(shader / "frag.glsl"),
                                "--reference" if driver == "softpipe" else "--host-execute", str(raw)],
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode != 0
    assert "MGCMIP1" in completed.stderr, completed.stdout + completed.stderr
    assert not raw.exists() and not list(captures.glob("*.json"))
    assert "MGC_HOST_EXECUTOR_COMMIT" not in completed.stderr and '"native_gl_readback"' not in completed.stdout


@pytest.mark.parametrize("case", acceptance.CASES)
def test_actual_softpipe_explicit_lod_same_inputs_against_fraction_oracle(tmp_path, monkeypatch, case):
    require_harness()
    monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
    shader = acceptance._shader(case, tmp_path)
    texture, mesh = tmp_path / "texture.bin", tmp_path / "vertices.bin"
    levels = acceptance.input_levels(case.width, case.height)
    texture.write_bytes(acceptance.encode_mipmaps(levels))
    mesh.write_bytes(acceptance.case_mesh(case))
    image = softpipe_reference(shader, BUILD, tmp_path / "reference", mesh=mesh, texture=texture, mipmaps=True,
                               texture_filter=case.texture_filter, texture_wrap_s=case.texture_wrap_s,
                               texture_wrap_t=case.texture_wrap_t, depth_test=case.depth_test, repeats=case.repeats)
    expected, coverage = acceptance.rational_oracle(levels, lod=case.lod, phase=case.phase, texture_filter=case.texture_filter,
                                                   texture_wrap_s=case.texture_wrap_s, texture_wrap_t=case.texture_wrap_t,
                                                   discard=case.variant != "sample")
    tolerance = 2 if case.texture_filter == "linear" else 0
    assert acceptance.image_comparison(image, expected, tolerance)["passed"]
    assert coverage.sum() == (256 if case.variant == "sample" else 128)


@pytest.fixture(scope="module")
def fresh_mipmap_texture(tmp_path_factory):
    require_harness()
    output = tmp_path_factory.mktemp("fresh-mipmap-texture")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
        summary = acceptance.validate(output, BUILD)
    return output, summary


def test_actual_capture_managed_native_and_softpipe_default_matrix(fresh_mipmap_texture):
    output, summary = fresh_mipmap_texture
    assert summary["passed"] and summary["native_gl_readback"] and summary["extension_available"]
    assert summary["fpga_execution"] is False and summary["maximum_channel_error"] <= 2
    assert summary["tested_cases"] == [case.name for case in acceptance.DEFAULT_CASES]
    for result in summary["cases"]:
        assert result["comparisons"]["native_vs_managed"]["max_channel_error"] == 0
        assert all(check["passed"] for check in result["comparisons"].values())
        if result["texture_filter"] == "nearest":
            assert result["maximum_channel_error"] == 0
        assert result["selected_level"] == min(result["source_lod"], len(result["level_dimensions"]) - 1)
        for name in ("managed", "native", "softpipe", "rational"):
            assert (output / result["case"] / f"{name}.rgba8").stat().st_size == 1024
        selected = result["input_controls"]["selected"]
        assert selected["applicable"] and all(check["max_channel_error"] > 2 for check in selected["image_changes"].values())
        unselected = result["input_controls"]["unselected"]
        if unselected["applicable"]:
            assert all(check["max_channel_error"] == 0 for check in unselected["image_changes"].values())
        for report in result["native_draw_reports"]:
            assert report["fpga_execution"] is False and 256 < report["host_steps"] <= 2_000_000


def test_actual_c_capture_preserves_every_level_and_rejects_corruption(fresh_mipmap_texture):
    output, summary = fresh_mipmap_texture
    case = acceptance.DEFAULT_CASES[0]
    artifact = json.loads((output / case.name / "capture/draw.json").read_text())
    levels = acceptance.input_levels(case.width, case.height)
    before = copy.deepcopy(artifact)
    acceptance.check_contract(artifact, levels, case)
    assert artifact == before
    for corruption in ("level_data", "level_metadata", "lod", "sampler"):
        broken = copy.deepcopy(artifact)
        if corruption == "level_data":
            broken["textures"][0]["mipmaps"][-1]["data_hex"] = "00000000"
        elif corruption == "level_metadata":
            broken["textures"][0]["mipmaps"][-1]["level"] = True
        elif corruption == "lod":
            broken["shaders"]["fs"]["layout"]["host_texture"]["lod"] = 4
        else:
            broken["textures"][0]["sampler"]["mip_filter"] = "linear"
        with pytest.raises(ValueError):
            acceptance.check_contract(broken, levels, case)


def test_actual_transparent_repeated_depth_kill_commits_have_exact_z16(fresh_mipmap_texture):
    output, summary = fresh_mipmap_texture
    for result in summary["cases"]:
        if not result["depth_test"]:
            continue
        assert result["repeats"] == 2 and result["source_lod"] == 4 and result["selected_level"] == 0
        first, second = result["native_draw_reports"]
        assert first["committed_fragments"] == 128 and first["discarded_fragments"] == 256
        assert second["committed_fragments"] == 0 and second["discarded_fragments"] == 256
        depth = np.frombuffer((output / result["case"] / "native/depth.z16").read_bytes(), dtype="<u2").reshape(16, 16)
        assert np.all(depth[:, :8] == 65535) and np.all(depth[:, 8:] == 16384)
        native = np.frombuffer((output / result["case"] / "native.rgba8").read_bytes(), dtype=np.uint8).reshape(16, 16, 4)
        assert np.all(native[:, :8] == 0) and np.all(native[:, 8:] == [13, 127, 250, 0])
