"""Direct C static explicit-LOD mip capture; no derivative or hardware claims."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from test_instancing_capture import HOST_INSTANCING
from test_mixed_texture_capture import CONTRACT_REJECTIONS, OWNERSHIP_REJECTIONS, WRAPS
from test_texture_capture import (
    BINDING_REJECTIONS,
    DISCARD_CONTRACT_REJECTIONS,
    RESOURCE_REJECTIONS,
    SAMPLER_REJECTIONS,
    TRANSACTIONAL_BINDINGS,
    build_texture_guard,
    read_capture,
)


SOURCE = Path(__file__).parent / "fixtures/mipmap-texture-capture/guard.c"
VERSIONS = (9, 10)
SHAPES = ((1, 1), (1, 16), (16, 1), (2, 2), (3, 5), (7, 9), (16, 16))
MIP_METADATA_REJECTIONS = (
    "actual_lod_invalid", "actual_lod_mismatch", "actual_vs_lod",
) + tuple(
    f"metadata_lod_{value}"
    for value in ("missing", "duplicate", "mismatch", '"2"', "2.0", "-1", "5",
                  "4294967296", "null", "[]", "{}", "true")
) + tuple(
    f"metadata_mip_{value}"
    for value in ("missing", "duplicate", '"none"', '"linear"', '"Nearest"', "null", "0")
) + tuple(
    f"metadata_levels_{value}"
    for value in ("missing", "duplicate", "4", "6", "5.0", '"5"', "null")
)
GUARD_REJECTIONS = tuple(case for case in CONTRACT_REJECTIONS if case != "abi_filter_mismatch") + (
    MIP_METADATA_REJECTIONS + OWNERSHIP_REJECTIONS + BINDING_REJECTIONS +
    DISCARD_CONTRACT_REJECTIONS + tuple(case for case in RESOURCE_REJECTIONS if case != "mip_chain") +
    tuple(case for case in SAMPLER_REJECTIONS if case not in
          ("min_linear", "mag_linear", "mip_nearest", "wrap_s", "wrap_t")) +
    ("min_wrong", "mag_wrong", "both_wrong", "mip_none", "mip_unknown", "chain_short", "chain_long",
     "custom_texture", "custom_texture_option_zero", "callback_disabled")
)


@pytest.fixture(scope="module")
def mipmap_guard(tmp_path_factory):
    return build_texture_guard(tmp_path_factory, SOURCE)


def run_mipmap_guard(helper, directory, case, version=9, lod=2, shape=(16, 16),
                     wrap_s="repeat", wrap_t="mirrored-repeat", instancing=None, target=None):
    environment = dict(os.environ)
    for key in ("MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T",
                "MGC_HOST_INSTANCING", "MGC_FRAGMENT_STATUS", "MGC_COMPILE_TARGET",
                "MGC_HOST_MESH", "MGC_ASM_DUMP", "MGC_HOST_EXECUTOR_SOCKET"):
        environment.pop(key, None)
    if version is not None:
        environment["MGC_HOST_TEXTURE"] = str(version)
    for axis, value in (("S", wrap_s), ("T", wrap_t)):
        if value is not None:
            environment[f"MGC_HOST_TEXTURE_WRAP_{axis}"] = value
    if instancing is not None:
        environment["MGC_HOST_INSTANCING"] = instancing
    if target is not None:
        environment["MGC_COMPILE_TARGET"] = target
    environment["MGC_HOST_CAPTURE_DIR"] = str(directory)
    result = subprocess.run([str(helper), case, str(lod), *(str(value) for value in shape)],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def assert_mip_capture(artifact, version, lod, shape, wrap_s="repeat", wrap_t="mirrored-repeat",
                       instanced=False):
    width, height = shape
    last = max(shape).bit_length() - 1
    filtering = "linear" if version == 10 else "nearest"
    assert artifact["schema"] == ("mgc-host-draw-v3" if instanced else "mgc-host-draw-v2")
    assert artifact["program_format"] == "legacy-mgcsim-v1"
    assert artifact["shader_abi_version"] == (3 if instanced else 1)
    assert artifact["shaders"]["vs"]["layout"]["version"] == (3 if instanced else 1)
    assert "host_texture" not in artifact["shaders"]["vs"]["layout"]
    fragment = artifact["shaders"]["fs"]["layout"]
    assert fragment["version"] == (3 if instanced else 2)
    assert fragment["host_texture"] == {
        "version": version, "texture": 0, "sampler": 0, "storage_block": 15,
        "max_width": 16, "max_height": 16, "filter": filtering, "wrap_s": wrap_s,
        "wrap_t": wrap_t, "lod": lod, "mip_filter": "nearest", "max_levels": 5,
    }
    assert len(fragment["host_texture"]) == 12 and "wrap" not in fragment["host_texture"]
    texture, = artifact["textures"]
    assert set(texture) == {"width", "height", "format", "data_hex", "sampler", "mipmaps"}
    assert (texture["width"], texture["height"]) == shape
    assert texture["format"] == "R8G8B8A8_UNORM"
    assert texture["sampler"] == {
        "normalized_coords": True, "min_filter": filtering, "mag_filter": filtering,
        "mip_filter": "nearest", "wrap_s": wrap_s, "wrap_t": wrap_t,
    }
    assert len(texture["mipmaps"]) == last
    total_bytes = 0
    for level, payload in enumerate((texture, *texture["mipmaps"])):
        expected_shape = (max(1, width >> level), max(1, height >> level))
        assert (payload["width"], payload["height"]) == expected_shape
        if level:
            assert set(payload) == {"level", "width", "height", "data_hex"}
            assert payload["level"] == level
        expected = bytes((index * 17 + level * 37 + 3) & 255
                         for index in range(expected_shape[0] * expected_shape[1] * 4))
        assert bytes.fromhex(payload["data_hex"]) == expected
        total_bytes += len(expected)
    assert total_bytes <= 1364
    if instanced:
        assert artifact["host_instancing"] == HOST_INSTANCING
        assert artifact["draw"]["instance_count"] == 4 and artifact["draw"]["start_instance"] == 2
    else:
        assert "host_instancing" not in artifact


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("lod", range(5))
@pytest.mark.parametrize("shape", SHAPES)
def test_actual_all_level_capture_keeps_source_lod(mipmap_guard, tmp_path, version, lod, shape):
    status = run_mipmap_guard(mipmap_guard, tmp_path, "valid", version, lod, shape)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert_mip_capture(read_capture(tmp_path), version, lod, shape)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("wrap_s", WRAPS)
@pytest.mark.parametrize("wrap_t", WRAPS)
def test_independent_wrap_axes_survive_mip_capture(mipmap_guard, tmp_path, version, wrap_s, wrap_t):
    status = run_mipmap_guard(mipmap_guard, tmp_path, "valid", version, 4, (3, 5), wrap_s, wrap_t)
    assert status["accepted"] == 1 and status["error"] == 0
    assert_mip_capture(read_capture(tmp_path), version, 4, (3, 5), wrap_s, wrap_t)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("lod", range(5))
@pytest.mark.parametrize("shape", SHAPES)
def test_max_lod_accepts_clamped_selected_level(mipmap_guard, tmp_path, version, lod, shape):
    status = run_mipmap_guard(mipmap_guard, tmp_path, "max_lod_selected", version, lod, shape)
    assert status["accepted"] == 1 and status["error"] == 0
    assert_mip_capture(read_capture(tmp_path), version, lod, shape)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("lod", range(5))
@pytest.mark.parametrize("shape", SHAPES)
def test_max_lod_cannot_clamp_below_selected_level(mipmap_guard, tmp_path, version, lod, shape):
    status = run_mipmap_guard(mipmap_guard, tmp_path, "max_lod_low", version, lod, shape)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", GUARD_REJECTIONS)
def test_resource_sampler_and_metadata_guards_fail_closed(mipmap_guard, tmp_path, version, case):
    status = run_mipmap_guard(mipmap_guard, tmp_path, case, version)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and status["operation_error"] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("level", range(5))
@pytest.mark.parametrize("field", ("width", "height", "layers", "stride", "layer_stride", "offset", "span"))
def test_every_level_storage_is_bounded(mipmap_guard, tmp_path, version, level, field):
    status = run_mipmap_guard(mipmap_guard, tmp_path, f"level_{level}_{field}", version)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("level", range(5))
def test_mapping_any_level_rejects_the_entire_snapshot(mipmap_guard, tmp_path, version, level):
    status = run_mipmap_guard(mipmap_guard, tmp_path, f"mapped_level_{level}", version)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", TRANSACTIONAL_BINDINGS + ("valid_copy", "valid_snapshot", "valid_release_resource"))
def test_all_level_snapshot_and_transferred_resource_ownership(mipmap_guard, tmp_path, version, case):
    status = run_mipmap_guard(mipmap_guard, tmp_path, case, version)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert bool(status["operation_error"]) == (case in TRANSACTIONAL_BINDINGS)
    assert_mip_capture(read_capture(tmp_path), version, 2, (16, 16))


@pytest.mark.parametrize("version", VERSIONS)
def test_instancing_with_mipmaps_uses_v3(mipmap_guard, tmp_path, version):
    status = run_mipmap_guard(mipmap_guard, tmp_path, "valid_instanced", version, 4, (3, 5), instancing="1")
    assert status["accepted"] == 1 and status["error"] == 0
    assert_mip_capture(read_capture(tmp_path), version, 4, (3, 5), instanced=True)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("axis", ("s", "t"))
@pytest.mark.parametrize("value", (None, "", "0", "repeat ", "Repeat", "mirror-repeat", "unknown"))
def test_explicit_axes_remain_required(mipmap_guard, tmp_path, version, axis, value):
    arguments = {"wrap_s": "repeat", "wrap_t": "mirrored-repeat", f"wrap_{axis}": value}
    status = run_mipmap_guard(mipmap_guard, tmp_path, "compile_reject", version, **arguments)
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("lod", (-1, 5, 1.5, "nan", "inf", "-inf"))
def test_only_bounded_integer_source_lod_compiles(mipmap_guard, tmp_path, version, lod):
    status = run_mipmap_guard(mipmap_guard, tmp_path, "compile_reject", version, lod)
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", (None, 0, 11, "09", "010", "+9", "10.0", "10 "))
def test_default_and_true_unknown_versions_reject(mipmap_guard, tmp_path, version):
    status = run_mipmap_guard(mipmap_guard, tmp_path, "compile_reject", version, wrap_s=None, wrap_t=None)
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
def test_hardware_mipmaps_are_not_a_capture_escape(mipmap_guard, tmp_path, version):
    status = run_mipmap_guard(mipmap_guard, tmp_path, "compile_reject", version, target="fp16-hardware-v1")
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", range(1, 9))
def test_old_versions_remain_single_level_without_mip_metadata(mipmap_guard, tmp_path, version):
    axes = {} if version >= 7 else {"wrap_s": None, "wrap_t": None}
    status = run_mipmap_guard(mipmap_guard, tmp_path, "valid", version, lod=0, **axes)
    assert status["accepted"] == 1 and status["error"] == 0
    artifact = read_capture(tmp_path)
    texture, = artifact["textures"]
    assert "mipmaps" not in texture and texture["sampler"]["mip_filter"] == "none"
    metadata = artifact["shaders"]["fs"]["layout"]["host_texture"]
    assert len(metadata) == (9 if version >= 7 else 8)
    assert all(key not in metadata for key in ("lod", "mip_filter", "max_levels"))


@pytest.mark.parametrize("version", range(1, 9))
def test_old_versions_reject_multiple_resource_levels(mipmap_guard, tmp_path, version):
    axes = {} if version >= 7 else {"wrap_s": None, "wrap_t": None}
    status = run_mipmap_guard(mipmap_guard, tmp_path, "old_mip_chain", version, lod=0, **axes)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []
