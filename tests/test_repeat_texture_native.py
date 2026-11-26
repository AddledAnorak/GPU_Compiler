"""Bounded repeat API guards and fresh CPU-host GL readback evidence."""

import inspect
from fractions import Fraction
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from tools import run_render
from tools import repeat_texture_acceptance as acceptance
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
        options["socket_path"] = Path("/tmp/mgc-repeat-test.sock")
    return options


@pytest.mark.parametrize("helper", HELPERS)
def test_wrap_is_optional_keyword_with_unchanged_clamp_default(helper):
    parameter = inspect.signature(helper).parameters["texture_wrap"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "clamp-to-edge"
    assert inspect.signature(helper).parameters["texture_filter"].default == "nearest"


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("texture_wrap", [None, True, 3, "", "REPEAT", "mirror-repeat",
                                          "clamp", b"repeat", [], {}])
def test_helpers_reject_unknown_wrap_before_gl(monkeypatch, tmp_path, helper, texture_wrap):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid wrap must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="texture wrap"):
        helper(FIXTURE, BUILD, tmp_path, texture_wrap=texture_wrap,
               **helper_options(helper, tmp_path / "texture.bin"))


@pytest.mark.parametrize("helper", HELPERS)
def test_helpers_reject_repeat_without_actual_texture_before_gl(monkeypatch, tmp_path, helper):
    def forbidden(*args, **kwargs):
        pytest.fail("repeat without texture must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="repeat.*actual texture"):
        helper(FIXTURE, BUILD, tmp_path, texture_wrap="repeat", **helper_options(helper))


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
@pytest.mark.parametrize("texture_wrap", ["clamp-to-edge", "repeat"])
@pytest.mark.parametrize("host_texture", [False, True])
def test_capture_rejects_all_hardware_texture_inputs(monkeypatch, tmp_path, texture_filter,
                                                     texture_wrap, host_texture):
    def forbidden(*args, **kwargs):
        pytest.fail("hardware texture must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="unsupported on hardware"):
        capture_draw(FIXTURE, BUILD, tmp_path, ProgramFormat.HARDWARE, host_texture=host_texture,
                     texture=tmp_path / "texture.bin", texture_filter=texture_filter,
                     texture_wrap=texture_wrap)


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("mode", [None, ("nearest", "clamp-to-edge", "1"),
                                  ("linear", "clamp-to-edge", "2"),
                                  ("nearest", "repeat", "3"), ("linear", "repeat", "4")])
def test_helpers_manage_inherited_wrap_and_exact_versions(monkeypatch, tmp_path, helper, mode):
    inherited = ("MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_FILTER", "MGC_HOST_TEXTURE_WRAP",
                 "MGC_HOST_TEXTURE_DATA", "MGC_HOST_BLEND", "MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST",
                 "MGC_FRAGMENT_STATUS", "MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_WIDTH",
                 "MGC_HOST_HEIGHT", "MGC_HOST_DRAW_REPEATS")
    for name in inherited:
        monkeypatch.setenv(name, "inherited-invalid")
    texture = tmp_path / "texture.bin" if mode else None
    options = helper_options(helper, texture)
    if mode:
        options.update(texture_filter=mode[0], texture_wrap=mode[1])

    def intercepted(command, *, env, **kwargs):
        assert env["GALLIUM_DRIVER"] == ("softpipe" if helper is softpipe_reference else "mgc")
        assert env["MGC_COMPILE_TARGET"] == ProgramFormat.LEGACY.value
        assert env.get("MGC_HOST_TEXTURE_WRAP") == (mode[1] if mode else None)
        assert env.get("MGC_HOST_TEXTURE_FILTER") == (mode[0] if mode else None)
        assert env.get("MGC_HOST_TEXTURE_DATA") == (str(texture) if texture else None)
        assert env.get("MGC_HOST_TEXTURE") == (mode[2] if mode and helper is not softpipe_reference else None)
        for name in ("MGC_HOST_BLEND", "MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST"):
            assert name not in env
        if helper is native_host_readback:
            assert env["MGC_HOST_EXECUTOR_SOCKET"] == str(options["socket_path"])
            assert env["MGC_HOST_WIDTH"] == env["MGC_HOST_HEIGHT"] == "16"
        else:
            assert "MGC_HOST_EXECUTOR_SOCKET" not in env
            assert "MGC_HOST_WIDTH" not in env and "MGC_HOST_HEIGHT" not in env
        assert env.get("MGC_FRAGMENT_STATUS") == (None if helper is softpipe_reference else "1")
        assert env.get("MGC_HOST_DRAW_REPEATS") == (None if helper is capture_draw else "1")
        raise InterceptedDraw

    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **options)


@pytest.mark.parametrize("helper", HELPERS)
def test_old_texture_calls_explicitly_override_inherited_repeat(monkeypatch, tmp_path, helper):
    monkeypatch.setenv("MGC_HOST_TEXTURE_WRAP", "repeat")
    def intercepted(command, *, env, **kwargs):
        assert env["MGC_HOST_TEXTURE_WRAP"] == "clamp-to-edge"
        assert env["MGC_HOST_TEXTURE_FILTER"] == "nearest"
        assert env.get("MGC_HOST_TEXTURE") == (None if helper is softpipe_reference else "1")
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper, tmp_path / "texture.bin"))


def test_render_cli_forwards_repeat_to_capture_and_real_reference(monkeypatch, tmp_path):
    modes = []
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    def capture(*args, **kwargs):
        modes.append(("capture", kwargs["texture_filter"], kwargs["texture_wrap"]))
        return {}
    def reference(*args, **kwargs):
        modes.append(("reference", kwargs["texture_filter"], kwargs["texture_wrap"]))
        return image
    monkeypatch.setattr(run_render, "capture_draw", capture)
    monkeypatch.setattr(run_render, "softpipe_reference", reference)
    monkeypatch.setattr(run_render, "render_artifact",
                        lambda *args, **kwargs: (image, np.ones((16, 16), np.uint32), {}))
    monkeypatch.setattr(run_render, "write_png", lambda *args, **kwargs: None)
    assert run_render.main([str(FIXTURE), "--host-texture", "--texture", str(tmp_path / "texture.bin"),
                            "--texture-filter", "linear", "--texture-wrap", "repeat",
                            "--compare-softpipe", "--output", str(tmp_path)]) == 0
    assert modes == [("capture", "linear", "repeat"), ("reference", "linear", "repeat")]


def test_render_cli_rejects_repeat_override_for_saved_capture(tmp_path, capsys):
    assert run_render.main(["--capture", str(tmp_path / "absent.json"), "--texture-wrap", "repeat"]) == 1
    assert "require a fresh shader draw" in capsys.readouterr().err


def test_render_cli_documents_explicit_wrap(capsys):
    with pytest.raises(SystemExit) as status:
        run_render.main(["--help"])
    assert status.value.code == 0
    assert "--texture-wrap {clamp-to-edge,repeat,mirrored-repeat}" in capsys.readouterr().out


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
@pytest.mark.parametrize("width,height", [(1, 1), (1, 16), (16, 1), (2, 2), (3, 5), (5, 3), (16, 16)])
def test_rational_repeat_centers_and_periodic_translations(width, height, texture_filter):
    texels = acceptance.input_texels(width, height)
    original = texels.copy()
    for row in range(height):
        for column in range(width):
            horizontal, vertical = Fraction(2 * column + 1, 2 * width), Fraction(2 * row + 1, 2 * height)
            expected = tuple(int(channel) for channel in texels[row, column])
            assert acceptance.rational_sample(texels, horizontal, vertical, texture_filter) == expected
            assert acceptance.rational_sample(texels, horizontal - 2, vertical + 3, texture_filter) == expected
    assert np.array_equal(texels, original)


@pytest.mark.parametrize("horizontal,vertical", [(0, 0), (1, 1), (-1, 2)])
def test_linear_repeat_seam_blends_wrapped_border_taps(horizontal, vertical):
    texels = acceptance.input_texels(2, 2)
    sampled = acceptance.rational_sample(texels, horizontal, vertical, "linear")
    assert sampled == (Fraction(255, 4), Fraction(255, 4), Fraction(255, 4), Fraction(383, 4))
    assert tuple(round(channel) for channel in sampled) == (64, 64, 64, 96)
    assert acceptance.rational_sample(texels, horizontal, vertical, "nearest") == tuple(int(channel) for channel in texels[0, 0])
    assert sampled != acceptance.clamp_sample(texels, horizontal, vertical)


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
def test_rational_negative_fraction_uses_last_texels_not_truncation(texture_filter):
    texels = acceptance.input_texels(3, 5)
    horizontal, vertical = Fraction(-1, 6), Fraction(-1, 10)
    assert acceptance.rational_sample(texels, horizontal, vertical, texture_filter) == tuple(int(channel) for channel in texels[-1, -1])
    assert acceptance.rational_sample(texels, horizontal, vertical, texture_filter) != acceptance.clamp_sample(texels, horizontal, vertical, texture_filter)


def test_repeat_pixel_coordinates_include_exact_positive_and_negative_seams():
    assert acceptance.repeat_coordinates(0, 0) == (Fraction(-11, 8), Fraction(19, 8))
    assert acceptance.repeat_coordinates(15, 0, "seams") == (0, 0)
    assert acceptance.repeat_coordinates(7, 8, "seams") == (1, 1)
    assert acceptance.repeat_coordinates(15, 0, "negative-seams") == (-1, -1)
    assert acceptance.repeat_coordinates(7, 8, "negative-seams") == (0, 0)


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
def test_repeat_oracle_keeps_zero_alpha_rgb_and_independent_coverage(texture_filter):
    texels = acceptance.input_texels(1, 1)
    image, coverage = acceptance.rational_oracle(texels, texture_filter=texture_filter)
    assert np.all(image == [13, 127, 250, 0]) and np.all(coverage == 1)
    image, coverage = acceptance.rational_oracle(texels, texture_filter=texture_filter, discard=True)
    assert np.all(image[:, :8] == 0) and np.all(image[:, 8:] == [13, 127, 250, 0])
    assert np.all(coverage[:, :8] == 0) and np.all(coverage[:, 8:] == 1)
    assert coverage.dtype == np.uint32


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_named_repeat_cases_have_wrap_and_filter_witnesses(case):
    texels = acceptance.input_texels(case.width, case.height)
    options = {"phase": case.phase, "texture_filter": case.texture_filter,
               "discard": case.variant != "sample"}
    repeat, coverage = acceptance.rational_oracle(texels, **options)
    clamp, clamp_coverage = acceptance.rational_oracle(texels, texture_wrap="clamp-to-edge", **options)
    assert np.array_equal(coverage, clamp_coverage)
    if case.width > 1:
        assert acceptance.image_comparison(repeat, clamp, 0)["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR
    else:
        assert np.array_equal(repeat, clamp)
    if case.texture_filter == "linear" and case.width > 1:
        nearest, _ = acceptance.rational_oracle(texels, **{**options, "texture_filter": "nearest"})
        if case.width == 4 and case.phase == "clamp":
            assert np.array_equal(repeat, nearest)
        else:
            assert acceptance.image_comparison(repeat, nearest, 0)["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR


@pytest.mark.parametrize("cases", [(), (acceptance.CASES[0], acceptance.CASES[0])])
def test_acceptance_rejects_empty_or_duplicate_cases_before_gl(monkeypatch, tmp_path, cases):
    def forbidden(*args, **kwargs):
        pytest.fail("unbounded selection must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="distinct named bounded"):
        acceptance.validate(tmp_path, BUILD, cases=cases)


def test_acceptance_records_exact_failure_in_json(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("exact repeat failure for guard test")
    monkeypatch.setattr(acceptance, "_run_case", fail)
    with pytest.raises(ValueError, match="exact repeat failure for guard test"):
        acceptance.validate(tmp_path, BUILD, cases=(acceptance.CASES[0],))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] is False and summary["status"] == "error"
    assert summary["message"] == "exact repeat failure for guard test"
    assert summary["failed_case"] == acceptance.CASES[0].name
    assert summary["native_gl_readback"] is False and summary["fpga_execution"] is False


def test_acceptance_cli_exposes_bounded_selection_and_json_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as status:
        acceptance.main(["--help"])
    assert status.value.code == 0
    help_text = capsys.readouterr().out
    assert "--case" in help_text and "--build" in help_text and "--output" in help_text
    assert acceptance.main(["--output", str(tmp_path), "--case", acceptance.CASES[0].name,
                            "--case", acceptance.CASES[0].name]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "error"


def test_default_acceptance_is_a_bounded_core_with_optional_extended_cases():
    assert len(acceptance.DEFAULT_CASES) == 10
    assert set(acceptance.DEFAULT_CASES).issubset(acceptance.CASES)
    for texture_filter in ("nearest", "linear"):
        cases = [case for case in acceptance.DEFAULT_CASES if case.texture_filter == texture_filter]
        assert {(case.width, case.height) for case in cases} == {(1, 1), (2, 2), (3, 5), (4, 4), (16, 16)}
        assert {case.phase for case in cases} == {"clamp", "seams", "negative-seams"}
        assert any(case.depth_test and case.repeats == 2 and case.variant == "runtime-loop" for case in cases)


@pytest.mark.parametrize("name,value,message", [("MGC_HOST_TEXTURE_WRAP", "", "must be clamp-to-edge or repeat"),
                                               ("MGC_HOST_TEXTURE_WRAP", "REPEAT", "must be clamp-to-edge or repeat"),
                                               ("MGC_HOST_TEXTURE_WRAP", "mirror-repeat", "must be clamp-to-edge or repeat"),
                                               ("MGC_HOST_TEXTURE_WRAP", "repeat", "requires explicit MGC_HOST_TEXTURE_DATA"),
                                               ("MGC_HOST_TEXTURE_FILTER", "repeat", "must be nearest or linear")])
def test_real_harness_rejects_invalid_sampler_flags(tmp_path, name, value, message):
    harness = run_render.ROOT / "tools/mgc_harness"
    if not harness.is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("parent must rebuild Mesa and the GL harness")
    environment = run_render.mesa_environment(BUILD, tmp_path)
    environment["GALLIUM_DRIVER"] = "softpipe"
    for flag in ("MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_FILTER", "MGC_HOST_TEXTURE_WRAP",
                 "MGC_HOST_TEXTURE_DATA", "MGC_HOST_BLEND", "MGC_HOST_EXECUTOR_SOCKET"):
        environment.pop(flag, None)
    environment[name] = value
    raw = tmp_path / "rejected.rgba8"
    result = subprocess.run([str(harness), str(FIXTURE / "vert.glsl"), str(FIXTURE / "frag.glsl"),
                             "--reference", str(raw)], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and message in result.stderr, result.stdout + result.stderr
    assert not raw.exists()


@pytest.mark.parametrize("version,texture_filter,texture_wrap", [("1", "nearest", "repeat"),
                                                                ("2", "nearest", "clamp-to-edge"),
                                                                ("3", "nearest", "clamp-to-edge"),
                                                                ("4", "linear", "clamp-to-edge")])
def test_real_mgc_harness_rejects_filter_wrap_version_mismatch(tmp_path, version, texture_filter, texture_wrap):
    harness = run_render.ROOT / "tools/mgc_harness"
    if not harness.is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("parent must rebuild Mesa and the GL harness")
    texture = tmp_path / "texture.bin"
    texture.write_bytes(struct.pack("<II", 2, 2) + acceptance.input_texels(2, 2).tobytes())
    environment = run_render.mesa_environment(BUILD, tmp_path)
    for flag in ("MGC_HOST_BLEND", "MGC_HOST_MESH", "MGC_HOST_CAPTURE_DIR"):
        environment.pop(flag, None)
    environment.update(MGC_HOST_TEXTURE=version, MGC_HOST_TEXTURE_FILTER=texture_filter,
                       MGC_HOST_TEXTURE_WRAP=texture_wrap, MGC_HOST_TEXTURE_DATA=str(texture))
    result = subprocess.run([str(harness), str(FIXTURE / "vert.glsl"), str(FIXTURE / "frag.glsl"), "--draw"],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "filter/wrap must match" in result.stderr, result.stdout + result.stderr


@pytest.fixture(scope="module")
def fresh_repeat_texture(tmp_path_factory):
    if not (run_render.ROOT / "tools/mgc_harness").is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("parent must rebuild Mesa lowering/capture and the GL harness first")
    output = tmp_path_factory.mktemp("fresh-repeat-texture")
    submissions, draws = [], []
    original_submit, original_run = Driver.submit, run_render.subprocess.run
    def submit(driver, program, *args, **kwargs):
        submissions.append(program.format)
        return original_submit(driver, program, *args, **kwargs)
    def actual_gl(command, *args, **kwargs):
        environment = kwargs.get("env", {})
        draws.append((command[3], environment.get("GALLIUM_DRIVER"), environment.get("MGC_HOST_TEXTURE"),
                      environment.get("MGC_HOST_TEXTURE_FILTER"), environment.get("MGC_HOST_TEXTURE_WRAP")))
        return original_run(command, *args, **kwargs)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Driver, "submit", submit)
        monkeypatch.setattr(run_render.subprocess, "run", actual_gl)
        summary = acceptance.validate(output, BUILD, cases=acceptance.CASES)
    return output, summary, submissions, draws


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_fresh_repeat_capture_reflection_managed_and_native_gl(case, fresh_repeat_texture):
    output, summary, submissions, draws = fresh_repeat_texture
    result = next(result for result in summary["cases"] if result["case"] == case.name)
    directory = output / case.name
    assert result["passed"] and result["texture_wrap"] == "repeat"
    assert result["host_texture_version"] == (4 if case.texture_filter == "linear" else 3)
    assert result["layout_version"] == 2 and result["shader_abi_version"] == 1
    assert result["native_commits"] == case.repeats and result["reply_matches_gl_readback"]
    assert all(comparison["passed"] for comparison in result["comparisons"].values())
    assert result["maximum_channel_error"] <= result["tested_channel_tolerance"]
    texels = acceptance.input_texels(case.width, case.height)
    artifact = json.loads((directory / "capture/draw.json").read_text())
    acceptance.check_contract(artifact, texels, case)
    assert (directory / "texture.bin").read_bytes() == struct.pack("<II", case.width, case.height) + texels.tobytes()
    native = np.frombuffer((directory / "native/native.rgba8").read_bytes(), np.uint8).reshape(16, 16, 4)[::-1]
    managed = np.frombuffer((directory / "managed.rgba8").read_bytes(), np.uint8).reshape(16, 16, 4)
    assert np.array_equal(native, managed)
    for name in ("draw", "mesh", "texture", "managed_rgba8", "native_report", "native_raw_bottom_left",
                 "softpipe_raw_bottom_left", "rational_rgba8", "coverage"):
        assert Path(result["evidence"][name]).is_file()
    assert submissions and all(target == ProgramFormat.LEGACY for target in submissions)
    version = str(result["host_texture_version"])
    assert ("--draw", "mgc", version, case.texture_filter, "repeat") in draws
    assert ("--host-execute", "mgc", version, case.texture_filter, "repeat") in draws
    assert ("--reference", "softpipe", None, case.texture_filter, "repeat") in draws
    if result["interpolation_required"]:
        assert result["native_vs_nearest"]["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR
        assert result["softpipe_vs_nearest"]["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR


def test_fresh_repeat_summary_and_same_input_clamp_controls(fresh_repeat_texture):
    output, summary, submissions, draws = fresh_repeat_texture
    assert summary["passed"] and summary["tested_cases"] == [case.name for case in acceptance.CASES]
    assert summary["native_gl_readback"] and summary["fpga_execution"] is False
    assert summary["exact_fp32_conformance"] is False and "binary16" in summary["input_precision"]
    assert "binary16" in summary["output_precision"] and summary["maximum_channel_error"] <= acceptance.MAX_CHANNEL_ERROR
    assert len(submissions) >= len(acceptance.CASES) * 4
    assert len([draw for draw in draws if draw[0] == "--draw"]) == len(acceptance.CASES)
    assert len([draw for draw in draws if draw[0] == "--host-execute"]) == len(acceptance.CASES) + 2
    for result in summary["cases"]:
        if result["case"] not in ("nearest-repeat-2x2", "linear-repeat-2x2"):
            continue
        control = result["actual_clamp_control"]
        assert control["same_input"] and control["native_gl_readback"]
        assert control["clamp_native_vs_rational"]["passed"] and control["clamp_softpipe_vs_rational"]["passed"]
        assert control["repeat_vs_clamp_native"]["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR
        assert control["repeat_vs_clamp_softpipe"]["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR
    assert (output / "summary.json").is_file()


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
def test_fresh_repeat_zero_alpha_and_depth_kill_are_explicit(texture_filter, fresh_repeat_texture):
    output, summary, _, _ = fresh_repeat_texture
    zero = next(result for result in summary["cases"] if result["case"] == f"{texture_filter}-repeat-zero-alpha-1x1")
    assert zero["managed_report"]["committed_fragments"] == 256
    assert (output / zero["case"] / "managed.rgba8").read_bytes() == bytes([13, 127, 250, 0]) * 256
    repeated = next(result for result in summary["cases"] if result["case"] == f"{texture_filter}-repeat-depth-loop-repeated-1x1")
    first, second = repeated["native_draw_reports"]
    assert first["committed_fragments"] == 128 and first["discarded_fragments"] == 256
    assert second["committed_fragments"] == 0 and second["discarded_fragments"] == 256
    depth = np.frombuffer((output / repeated["case"] / "native/depth.z16").read_bytes(), "<u2").reshape(16, 16)
    assert np.all(depth[:, :8] == 65535) and np.all(depth[:, 8:] == 16384)
    image = np.frombuffer((output / repeated["case"] / "managed.rgba8").read_bytes(), np.uint8).reshape(16, 16, 4)
    assert np.all(image[:, :8] == 0) and np.all(image[:, 8:] == [13, 127, 250, 0])
