"""Fresh full-RGBA linear-clamp capture/native evidence and independent rational checks."""

import copy
from fractions import Fraction
import inspect
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from tools import run_render
from tools.host_render import render_artifact
from tools.host_texture import (HOST_TEXTURE_LAYOUT, HOST_TEXTURE_LINEAR_LAYOUT,
                                LINEAR_SAMPLER_STATE, SAMPLER_STATE)
from tools.linear_texture_acceptance import (CASES, MAX_CHANNEL_ERROR, input_texels,
                                             main, pixel_coordinates, rational_oracle,
                                             rational_sample, validate)
from tools.mgcdriver import Driver, ProgramFormat
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
FIXTURE = ROOT / "tests/fixtures/host-texture"
HELPERS = (capture_draw, softpipe_reference, native_host_readback)


class InterceptedDraw(Exception):
    pass


def helper_options(helper, texture=None):
    options = {"texture": texture}
    if helper is capture_draw:
        options["host_texture"] = texture is not None
    if helper is native_host_readback:
        options["socket_path"] = Path("/tmp/mgc-linear-test.sock")
    return options


def test_linear_contract_changes_only_host_object_version_and_filter():
    expected = {**HOST_TEXTURE_LAYOUT, "version": 2, "filter": "linear"}
    assert HOST_TEXTURE_LINEAR_LAYOUT == expected
    assert expected == {"version": 2, "texture": 0, "sampler": 0, "storage_block": 15,
                        "max_width": 16, "max_height": 16, "filter": "linear", "wrap": "clamp-to-edge"}
    assert LINEAR_SAMPLER_STATE == {**SAMPLER_STATE, "min_filter": "linear", "mag_filter": "linear"}
    assert LINEAR_SAMPLER_STATE["mip_filter"] == "none"


@pytest.mark.parametrize("helper", HELPERS)
def test_filter_is_optional_keyword_and_keeps_nearest_default(helper):
    parameter = inspect.signature(helper).parameters["texture_filter"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "nearest"
    if helper is capture_draw:
        assert inspect.signature(helper).parameters["host_texture"].default is False


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("texture_filter", [None, True, False, "", "LINEAR", "bilinear", "repeat", 2, b"linear", [], {}])
def test_helpers_reject_unknown_filters_before_gl(monkeypatch, tmp_path, helper, texture_filter):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid filter must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="texture filter"):
        helper(FIXTURE, BUILD, tmp_path, texture_filter=texture_filter,
               **helper_options(helper, tmp_path / "texture.bin"))


@pytest.mark.parametrize("helper", HELPERS)
def test_helpers_reject_linear_without_actual_texture_before_gl(monkeypatch, tmp_path, helper):
    def forbidden(*args, **kwargs):
        pytest.fail("linear without texture must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="linear.*actual texture"):
        helper(FIXTURE, BUILD, tmp_path, texture_filter="linear", **helper_options(helper))


@pytest.mark.parametrize("host_texture", ["2", "1", 1, 2, None, np.bool_(True)])
def test_capture_keeps_boolean_host_texture_api(monkeypatch, tmp_path, host_texture):
    def forbidden(*args, **kwargs):
        pytest.fail("nonboolean host_texture must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="explicitly boolean"):
        capture_draw(FIXTURE, BUILD, tmp_path, host_texture=host_texture,
                     texture=tmp_path / "texture.bin", texture_filter="linear")


def test_capture_linear_remains_legacy_only(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("hardware texture must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="unsupported on hardware"):
        capture_draw(FIXTURE, BUILD, tmp_path, ProgramFormat.HARDWARE,
                     host_texture=True, texture=tmp_path / "texture.bin", texture_filter="linear")


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("texture_filter", [None, "nearest", "linear"])
def test_helpers_clear_inherited_env_and_select_actual_filter(monkeypatch, tmp_path, helper, texture_filter):
    inherited = ("MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_FILTER", "MGC_HOST_TEXTURE_DATA",
                 "MGC_HOST_BLEND", "MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST", "MGC_FRAGMENT_STATUS",
                 "MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT", "MGC_HOST_DRAW_REPEATS")
    for name in inherited:
        monkeypatch.setenv(name, "untrusted-inherited-value")
    texture = tmp_path / "texture.bin" if texture_filter is not None else None
    options = helper_options(helper, texture)
    if texture_filter is not None:
        options["texture_filter"] = texture_filter

    def intercepted(command, *, env, **kwargs):
        assert env["GALLIUM_DRIVER"] == ("softpipe" if helper is softpipe_reference else "mgc")
        assert env["MGC_COMPILE_TARGET"] == ProgramFormat.LEGACY.value
        assert env.get("MGC_HOST_TEXTURE_FILTER") == texture_filter
        assert env.get("MGC_HOST_TEXTURE_DATA") == (str(texture) if texture else None)
        expected_mode = None if helper is softpipe_reference or texture is None else (
            "2" if texture_filter == "linear" else "1")
        assert env.get("MGC_HOST_TEXTURE") == expected_mode
        assert "MGC_HOST_BLEND" not in env and "MGC_HOST_MESH" not in env and "MGC_HOST_DEPTH_TEST" not in env
        if helper is native_host_readback:
            assert env["MGC_HOST_EXECUTOR_SOCKET"] == str(options["socket_path"])
            assert env["MGC_HOST_WIDTH"] == env["MGC_HOST_HEIGHT"] == "16"
        else:
            assert "MGC_HOST_EXECUTOR_SOCKET" not in env
            assert "MGC_HOST_WIDTH" not in env and "MGC_HOST_HEIGHT" not in env
        if helper is softpipe_reference:
            assert "MGC_FRAGMENT_STATUS" not in env and "MGC_HOST_CAPTURE_DIR" not in env
        else:
            assert env["MGC_FRAGMENT_STATUS"] == "1"
        raise InterceptedDraw

    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **options)


@pytest.mark.parametrize("helper", HELPERS)
def test_old_calls_with_actual_texture_still_select_v1_nearest(monkeypatch, tmp_path, helper):
    def intercepted(command, *, env, **kwargs):
        assert env["MGC_HOST_TEXTURE_FILTER"] == "nearest"
        assert env.get("MGC_HOST_TEXTURE") == (None if helper is softpipe_reference else "1")
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper, tmp_path / "texture.bin"))


def test_rational_midpoint_blends_four_channels_instead_of_nearest():
    texels = input_texels(2, 2)
    sampled = rational_sample(texels, Fraction(1, 2), Fraction(1, 2))
    assert sampled == (Fraction(255, 4), Fraction(255, 4), Fraction(255, 4), Fraction(383, 4))
    assert tuple(round(channel) for channel in sampled) == (64, 64, 64, 96)
    assert sampled != rational_sample(texels, Fraction(1, 2), Fraction(1, 2), "nearest")


@pytest.mark.parametrize("width,height", [(1, 1), (1, 16), (16, 1), (2, 2), (4, 4), (3, 5), (5, 3), (16, 16)])
def test_rational_texel_centers_and_extreme_clamp_are_exact(width, height):
    texels = input_texels(width, height)
    original = texels.copy()
    for row in range(height):
        for column in range(width):
            sampled = rational_sample(texels, Fraction(2 * column + 1, 2 * width),
                                      Fraction(2 * row + 1, 2 * height))
            assert sampled == tuple(int(channel) for channel in texels[row, column])
    for horizontal, column in ((Fraction(-1000000), 0), (Fraction(1000000), width - 1)):
        for vertical, row in ((Fraction(-1000000), 0), (Fraction(1000000), height - 1)):
            assert rational_sample(texels, horizontal, vertical) == tuple(int(channel) for channel in texels[row, column])
    assert np.array_equal(texels, original)


def test_coordinate_oracle_matches_screen_gradient_and_outside_uv():
    assert pixel_coordinates(0, 0) == (Fraction(1, 32), Fraction(31, 32))
    assert pixel_coordinates(15, 15) == (Fraction(31, 32), Fraction(1, 32))
    assert pixel_coordinates(0, 0, "clamp") == (Fraction(-11, 8), Fraction(19, 8))
    assert pixel_coordinates(0, 0, "phase") == (Fraction(1, 32), Fraction(1, 32))


def test_zero_alpha_oracle_preserves_rgb_and_explicit_coverage():
    image, coverage = rational_oracle(input_texels(1, 1), phase="clamp")
    assert np.all(image == [13, 127, 250, 0])
    assert coverage.dtype == np.uint32 and np.all(coverage == 1)
    killed, coverage = rational_oracle(input_texels(1, 1), discard=True)
    assert np.all(killed[:, :8] == 0) and np.all(killed[:, 8:] == [13, 127, 250, 0])
    assert np.all(coverage[:, :8] == 0) and np.all(coverage[:, 8:] == 1)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_case_inputs_have_independent_interpolation_witnesses(case):
    texels = input_texels(case.width, case.height)
    expected, coverage = rational_oracle(texels, phase=case.phase, texture_filter=case.texture_filter,
                                         discard=case.variant != "sample")
    nearest, nearest_coverage = rational_oracle(texels, phase=case.phase, texture_filter="nearest",
                                               discard=case.variant != "sample")
    assert np.array_equal(coverage, nearest_coverage)
    if case.texture_filter == "linear" and case.width not in (1, 16):
        assert np.max(np.abs(expected.astype(int) - nearest.astype(int))) > MAX_CHANNEL_ERROR
    else:
        assert np.array_equal(expected, nearest)


@pytest.mark.parametrize("texels", [np.zeros((0, 1, 4), np.uint8), np.zeros((17, 1, 4), np.uint8),
                                    np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 4), np.float32), []])
def test_rational_oracle_rejects_unbounded_or_non_rgba8_input(texels):
    with pytest.raises(ValueError, match="RGBA8"):
        rational_oracle(texels)


def test_commands_expose_explicit_options_without_building(capsys):
    with pytest.raises(SystemExit) as status:
        main(["--help"])
    assert status.value.code == 0
    help_text = capsys.readouterr().out
    assert "--case" in help_text and "--build" in help_text and "--output" in help_text
    with pytest.raises(SystemExit) as status:
        run_render.main(["--help"])
    assert status.value.code == 0
    assert "--texture-filter {nearest,linear}" in capsys.readouterr().out


def test_render_cli_forwards_filter_to_capture_and_softpipe(monkeypatch, tmp_path):
    filters = []
    image = np.zeros((16, 16, 4), dtype=np.uint8)

    def capture(*args, **kwargs):
        filters.append(("capture", kwargs["texture_filter"]))
        return {}

    def reference(*args, **kwargs):
        filters.append(("reference", kwargs["texture_filter"]))
        return image

    monkeypatch.setattr(run_render, "capture_draw", capture)
    monkeypatch.setattr(run_render, "softpipe_reference", reference)
    monkeypatch.setattr(run_render, "render_artifact", lambda *args, **kwargs: (image, np.zeros((16, 16), np.uint32), {}))
    monkeypatch.setattr(run_render, "write_png", lambda *args, **kwargs: None)
    assert run_render.main([str(FIXTURE), "--host-texture", "--texture", str(tmp_path / "texture.bin"),
                            "--texture-filter", "linear", "--compare-softpipe", "--output", str(tmp_path)]) == 0
    assert filters == [("capture", "linear"), ("reference", "linear")]


def test_render_cli_rejects_linear_override_for_saved_capture(tmp_path, capsys):
    assert run_render.main(["--capture", str(tmp_path / "absent.json"), "--texture-filter", "linear"]) == 1
    assert "require a fresh shader draw" in capsys.readouterr().err


@pytest.fixture(scope="module")
def fresh_linear_texture(tmp_path_factory):
    if not (ROOT / "tools/mgc_harness").is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("parent must build the existing Mesa compiler and updated GL harness first")
    output = tmp_path_factory.mktemp("fresh-linear-texture")
    submissions, draws = [], []
    original_submit = Driver.submit
    original_run = run_render.subprocess.run

    def submit(driver, program, *args, **kwargs):
        submissions.append(program.format)
        return original_submit(driver, program, *args, **kwargs)

    def actual_gl(command, *args, **kwargs):
        environment = kwargs.get("env", {})
        draws.append((command[3], environment.get("GALLIUM_DRIVER"),
                      environment.get("MGC_HOST_TEXTURE"), environment.get("MGC_HOST_TEXTURE_FILTER")))
        return original_run(command, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Driver, "submit", submit)
        patch.setattr(run_render.subprocess, "run", actual_gl)
        summary = validate(output, BUILD)
    return output, summary, submissions, draws


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_fresh_case_has_actual_capture_managed_native_softpipe_and_rational_evidence(fresh_linear_texture, case):
    output, summary, submissions, draws = fresh_linear_texture
    result = next(result for result in summary["cases"] if result["case"] == case.name)
    directory = output / case.name
    assert result["passed"] and result["native_gl_readback"] and result["reply_matches_gl_readback"]
    assert result["host_texture_version"] == (2 if case.texture_filter == "linear" else 1)
    assert result["layout_version"] == 2 and result["native_commits"] == case.repeats
    assert all(comparison["passed"] for comparison in result["comparisons"].values())
    assert result["maximum_channel_error"] <= result["tested_channel_tolerance"]
    texels = input_texels(case.width, case.height)
    assert (directory / "texture.bin").read_bytes() == struct.pack("<II", case.width, case.height) + texels.tobytes()
    native = np.frombuffer((directory / "native/native.rgba8").read_bytes(), np.uint8).reshape(16, 16, 4)[::-1]
    managed = np.frombuffer((directory / "managed.rgba8").read_bytes(), np.uint8).reshape(16, 16, 4)
    assert np.array_equal(native, managed)
    for evidence_name in ("draw", "mesh", "texture", "managed_rgba8", "native_report",
                          "native_raw_bottom_left", "softpipe_raw_bottom_left", "rational_rgba8", "coverage"):
        assert Path(result["evidence"][evidence_name]).is_file()
    if result["interpolation_required"]:
        assert result["native_vs_nearest"]["max_channel_error"] > MAX_CHANNEL_ERROR
        assert result["softpipe_vs_nearest"]["max_channel_error"] > MAX_CHANNEL_ERROR
    assert submissions and all(target == ProgramFormat.LEGACY for target in submissions)
    assert ("--draw", "mgc", "2", "linear") in draws
    assert ("--host-execute", "mgc", "2", "linear") in draws
    assert ("--reference", "softpipe", None, "linear") in draws


def test_fresh_summary_names_exact_cases_and_never_claims_fp32_or_fpga(fresh_linear_texture):
    output, summary, submissions, draws = fresh_linear_texture
    assert summary["tested_cases"] == [case.name for case in CASES]
    assert len(summary["cases"]) == len(CASES)
    assert summary["exact_fp32_conformance"] is False and summary["fpga_execution"] is False
    assert summary["maximum_channel_error"] <= MAX_CHANNEL_ERROR
    assert len(submissions) >= len(CASES) * 4
    assert len([draw for draw in draws if draw[0] == "--draw"]) == len(CASES)
    assert len([draw for draw in draws if draw[0] == "--host-execute"]) == len(CASES)
    control = summary["cases"][0]["actual_nearest_control"]
    assert control["nearest_oracle"]["passed"]
    assert control["linear_vs_actual_nearest"]["max_channel_error"] > MAX_CHANNEL_ERROR


def test_fresh_zero_alpha_commits_and_repeated_depth_kill_stays_explicit(fresh_linear_texture):
    output, summary, _, _ = fresh_linear_texture
    zero = next(result for result in summary["cases"] if result["case"] == "linear-zero-alpha-1x1")
    assert zero["managed_report"]["committed_fragments"] == 256
    assert zero["managed_report"]["discarded_fragments"] == 0
    assert (output / zero["case"] / "managed.rgba8").read_bytes() == bytes([13, 127, 250, 0]) * 256
    repeated = next(result for result in summary["cases"] if result["case"] == "linear-depth-loop-repeated-2x2")
    first, second = repeated["native_draw_reports"]
    assert first["committed_fragments"] == 128 and first["discarded_fragments"] == 256
    assert second["committed_fragments"] == 0 and second["discarded_fragments"] == 256
    depth = np.frombuffer((output / repeated["case"] / "native/depth.z16").read_bytes(), "<u2").reshape(16, 16)
    assert np.all(depth[:, :8] == 65535) and np.all(depth[:, 8:] == 16384)


@pytest.mark.parametrize("mutation", ["layout-v3", "host-v1", "host-filter", "min-nearest", "mag-nearest",
                                       "mips", "binding", "storage-block", "width", "format", "extra-texture",
                                       "missing-status", "hardware"])
def test_fresh_linear_artifact_fails_closed_on_incompatible_contract(fresh_linear_texture, mutation):
    output, _, _, _ = fresh_linear_texture
    artifact = copy.deepcopy(json.loads((output / "linear-2x2/capture/draw.json").read_text()))
    layout = artifact["shaders"]["fs"]["layout"]
    texture = artifact["textures"][0]
    if mutation == "layout-v3":
        layout["version"] = 3
    elif mutation == "host-v1":
        layout["host_texture"]["version"] = 1
    elif mutation == "host-filter":
        layout["host_texture"]["filter"] = "nearest"
    elif mutation == "min-nearest":
        texture["sampler"]["min_filter"] = "nearest"
    elif mutation == "mag-nearest":
        texture["sampler"]["mag_filter"] = "nearest"
    elif mutation == "mips":
        texture["sampler"]["mip_filter"] = "nearest"
    elif mutation == "binding":
        layout["host_texture"]["texture"] = 1
    elif mutation == "storage-block":
        layout["host_texture"]["storage_block"] = 14
    elif mutation == "width":
        texture["width"] = 17
    elif mutation == "format":
        texture["format"] = "R8G8B8A8_SRGB"
    elif mutation == "extra-texture":
        artifact["textures"].append(copy.deepcopy(texture))
    elif mutation == "missing-status":
        del layout["fragment_status"]
        layout["uses_discard"] = True
    else:
        artifact["program_format"] = ProgramFormat.HARDWARE.value
    with pytest.raises(ValueError):
        render_artifact(artifact)
