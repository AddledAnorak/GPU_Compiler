"""Direct C mixed-axis capture guards; no sampling or rendered-output claims."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from test_texture_capture import (
    BINDING_REJECTIONS,
    DISCARD_CONTRACT_REJECTIONS,
    RESOURCE_REJECTIONS,
    SAMPLER_REJECTIONS,
    TRANSACTIONAL_BINDINGS,
    VALID_TEXTURES,
    assert_texture_capture,
    build_texture_guard,
    read_capture,
)


SOURCE = Path(__file__).parent / "fixtures/mixed-texture-capture/guard.c"
WRAPS = ("clamp-to-edge", "repeat", "mirrored-repeat")
VERSIONS = (7, 8)
OWNERSHIP_REJECTIONS = (
    "view_context", "view_texture_null", "resource_screen", "resource_next", "storage_busy",
)
CONTRACT_REJECTIONS = (
    "abi_zero", "abi_unknown", "abi_vs_unknown", "abi_hardware", "uses_texture_false",
    "abi_wrap_s_invalid", "abi_wrap_t_invalid", "abi_wrap_s_negative", "abi_wrap_t_negative",
    "abi_vs_wrap_invalid", "abi_wrap_s_mismatch", "abi_wrap_t_mismatch", "abi_filter_mismatch",
    "layout_version", "layout_filter", "layout_unknown", "layout_axis_unknown", "layout_duplicate",
    "layout_missing_texture", "layout_missing_sampler", "layout_missing_block",
    "layout_missing_width", "layout_missing_height", "layout_texture_slot", "layout_sampler_slot",
    "layout_block", "layout_width", "layout_height", "layout_legacy_wrap", "layout_host_null",
    "layout_stage", "layout_shader_abi", "layout_host_version", "layout_host_version_missing",
    "layout_trailing", "layout_truncated",
) + tuple(
    f"layout_wrap_{axis}_{value}"
    for axis in ("s", "t")
    for value in ("missing", "duplicate", "legacy", "mismatch", '"Repeat"', '"clamp"',
                  '"mirror-clamp"', '"unknown"', '"repeat "', "null", "0", "[]", "{}")
)
SAMPLER_GUARDS = tuple(
    case for case in SAMPLER_REJECTIONS
    if case not in ("min_linear", "mag_linear", "wrap_s", "wrap_t")
) + ("min_wrong", "mag_wrong", "both_wrong", "mip_unknown")


@pytest.fixture(scope="module")
def mixed_texture_guard(tmp_path_factory):
    return build_texture_guard(tmp_path_factory, SOURCE)


def run_mixed_guard(helper, directory, case, version=7, wrap_s="repeat",
                    wrap_t="mirrored-repeat", target=None):
    environment = dict(os.environ)
    for key in ("MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T",
                "MGC_FRAGMENT_STATUS", "MGC_COMPILE_TARGET", "MGC_HOST_MESH", "MGC_ASM_DUMP",
                "MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_INSTANCING"):
        environment.pop(key, None)
    if version is not None:
        environment["MGC_HOST_TEXTURE"] = str(version)
    for axis, value in (("S", wrap_s), ("T", wrap_t)):
        if value is not None:
            environment[f"MGC_HOST_TEXTURE_WRAP_{axis}"] = value
    if target is not None:
        environment["MGC_COMPILE_TARGET"] = target
    environment["MGC_HOST_CAPTURE_DIR"] = str(directory)
    result = subprocess.run([str(helper), case], env=environment, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def assert_mixed_capture(artifact, case, version, wrap_s, wrap_t):
    filtering = "linear" if version == 8 else "nearest"
    metadata = {
        "version": version, "texture": 0, "sampler": 0, "storage_block": 15,
        "max_width": 16, "max_height": 16, "filter": filtering,
        "wrap_s": wrap_s, "wrap_t": wrap_t,
    }
    sampler = {
        "normalized_coords": True, "min_filter": filtering, "mag_filter": filtering,
        "mip_filter": "none", "wrap_s": wrap_s, "wrap_t": wrap_t,
    }
    assert_texture_capture(artifact, case, host_texture=metadata, sampler=sampler)
    assert len(artifact["shaders"]["fs"]["layout"]["host_texture"]) == 9
    assert "wrap" not in artifact["shaders"]["fs"]["layout"]["host_texture"]


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("wrap_s", WRAPS)
@pytest.mark.parametrize("wrap_t", WRAPS)
def test_all_axis_pairs_capture_actual_sampler(mixed_texture_guard, tmp_path, version, wrap_s, wrap_t):
    status = run_mixed_guard(mixed_texture_guard, tmp_path, "valid", version, wrap_s, wrap_t)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["callbacks"] == 0 and status["operation_error"] == 0
    assert_mixed_capture(read_capture(tmp_path), "valid", version, wrap_s, wrap_t)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", VALID_TEXTURES + TRANSACTIONAL_BINDINGS + ("valid_release_resource",))
def test_capture_bounds_snapshot_and_ownership(mixed_texture_guard, tmp_path, version, case):
    status = run_mixed_guard(mixed_texture_guard, tmp_path, case, version)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["callbacks"] == 0
    assert bool(status["operation_error"]) == (case in TRANSACTIONAL_BINDINGS)
    assert_mixed_capture(read_capture(tmp_path), case, version, "repeat", "mirrored-repeat")


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", SAMPLER_GUARDS + RESOURCE_REJECTIONS + BINDING_REJECTIONS +
                         DISCARD_CONTRACT_REJECTIONS + CONTRACT_REJECTIONS + OWNERSHIP_REJECTIONS +
                         ("custom_texture", "custom_texture_option_zero", "callback_disabled"))
def test_fail_closed_before_capture_or_callback(mixed_texture_guard, tmp_path, version, case):
    status = run_mixed_guard(mixed_texture_guard, tmp_path, case, version)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and status["operation_error"] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("axis", ("s", "t"))
@pytest.mark.parametrize("wrap_s,wrap_t", (("repeat", "mirrored-repeat"), ("mirrored-repeat", "clamp-to-edge")))
@pytest.mark.parametrize("wrap", range(8))
def test_each_sampler_axis_must_match(mixed_texture_guard, tmp_path, version, axis, wrap_s, wrap_t, wrap):
    expected = {"repeat": 0, "clamp-to-edge": 2, "mirrored-repeat": 4}[wrap_s if axis == "s" else wrap_t]
    status = run_mixed_guard(mixed_texture_guard, tmp_path, f"sampler_wrap_{axis}_{wrap}",
                             version, wrap_s, wrap_t)
    assert status["accepted"] == int(wrap == expected)
    assert status["rejected"] == int(wrap != expected)
    assert bool(status["error"]) == (wrap != expected)
    assert status["callbacks"] == 0
    if wrap == expected:
        assert_mixed_capture(read_capture(tmp_path), "valid", version, wrap_s, wrap_t)
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("axis", ("s", "t"))
@pytest.mark.parametrize("value", (None, "", "0", "1", "2", "Repeat", "clamp", "mirror-repeat",
                                  "repeat ", " repeat", "repeat,mirrored-repeat", "unknown"))
def test_mixed_requires_explicit_canonical_axes(mixed_texture_guard, tmp_path, version, axis, value):
    arguments = {"wrap_s": "repeat", "wrap_t": "mirrored-repeat", f"wrap_{axis}": value}
    status = run_mixed_guard(mixed_texture_guard, tmp_path, "compile_reject", version, **arguments)
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert status["accepted"] == status["rejected"] == status["callbacks"] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", (None, 0, 1, 2, 3, 4, 5, 6))
@pytest.mark.parametrize("wrap_s,wrap_t", (("repeat", None), (None, "mirrored-repeat"), ("", "")))
def test_old_versions_reject_any_per_axis_environment(mixed_texture_guard, tmp_path, version, wrap_s, wrap_t):
    status = run_mixed_guard(mixed_texture_guard, tmp_path, "compile_reject", version, wrap_s, wrap_t)
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", (None, 0, 9, "-1", "07", "08", "7.0", "8 ", " 7", "+7", ""))
def test_default_and_unknown_reject_textures(mixed_texture_guard, tmp_path, version):
    status = run_mixed_guard(mixed_texture_guard, tmp_path, "compile_reject", version, None, None)
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", VERSIONS)
def test_hardware_rejects_mixed_even_capture_only(mixed_texture_guard, tmp_path, version):
    status = run_mixed_guard(mixed_texture_guard, tmp_path, "compile_reject", version,
                             target="fp16-hardware-v1")
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", range(1, 7))
def test_old_metadata_remains_eight_keys(mixed_texture_guard, tmp_path, version):
    status = run_mixed_guard(mixed_texture_guard, tmp_path, "valid", version, None, None)
    assert status["accepted"] == 1 and status["error"] == 0
    artifact = read_capture(tmp_path)
    metadata = artifact["shaders"]["fs"]["layout"]["host_texture"]
    assert len(metadata) == 8 and "wrap" in metadata
    assert "wrap_s" not in metadata and "wrap_t" not in metadata
    assert metadata["version"] == version and artifact["schema"] == "mgc-host-draw-v2"
