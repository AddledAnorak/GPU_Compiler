"""Exact mirrored API guards, independent mirror math and real full-RGBA GL evidence."""

import copy
from dataclasses import replace
from fractions import Fraction
import inspect
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from tools import mirrored_texture_acceptance as acceptance
from tools import run_render
from tools.mgcdriver import Driver, ProgramFormat
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
FIXTURE = ROOT / "tests/fixtures/host-texture"
HARNESS = Path(os.environ.get("MGC_MIRRORED_TEST_HARNESS", ROOT / "tools/mgc_harness"))
HELPERS = (capture_draw, softpipe_reference, native_host_readback)
FLAGS = ("MGC_HOST_CAPTURE_DIR", "MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_FILTER", "MGC_HOST_TEXTURE_WRAP",
         "MGC_HOST_TEXTURE_DATA", "MGC_HOST_BLEND", "MGC_HOST_CLEAR_RGBA", "MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST",
         "MGC_FRAGMENT_STATUS", "MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT",
         "MGC_HOST_DRAW_REPEATS")


class InterceptedDraw(Exception):
    pass


class WrapString(str):
    pass


def helper_options(helper, texture=None):
    options = {"texture": texture}
    if helper is capture_draw:
        options["host_texture"] = texture is not None
    if helper is native_host_readback:
        options["socket_path"] = Path("/tmp/mgc-mirror-test.sock")
    return options


@pytest.mark.parametrize("helper", HELPERS)
def test_wrap_is_keyword_opt_in_with_unchanged_defaults(helper):
    parameter = inspect.signature(helper).parameters["texture_wrap"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "clamp-to-edge"
    assert inspect.signature(helper).parameters["texture_filter"].default == "nearest"


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("texture_wrap", [None, True, False, 5, "", "MIRRORED-REPEAT", "mirror",
                                          "mirror-repeat", b"mirrored-repeat", [], {},
                                          WrapString("mirrored-repeat")])
def test_helpers_reject_invalid_wrap_before_gl(monkeypatch, tmp_path, helper, texture_wrap):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid sampler must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="texture wrap"):
        helper(FIXTURE, BUILD, tmp_path, texture_wrap=texture_wrap,
               **helper_options(helper, tmp_path / "texture.bin"))


@pytest.mark.parametrize("helper", HELPERS)
def test_helpers_reject_mirrored_without_texture_before_gl(monkeypatch, tmp_path, helper):
    def forbidden(*args, **kwargs):
        pytest.fail("mirrored wrap without texture must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="mirrored-repeat.*actual texture"):
        helper(FIXTURE, BUILD, tmp_path, texture_wrap="mirrored-repeat", **helper_options(helper))


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
@pytest.mark.parametrize("host_texture", [False, True])
def test_capture_rejects_mirrored_hardware_before_gl(monkeypatch, tmp_path, texture_filter, host_texture):
    def forbidden(*args, **kwargs):
        pytest.fail("hardware sampler must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="unsupported on hardware"):
        capture_draw(FIXTURE, BUILD, tmp_path, ProgramFormat.HARDWARE, host_texture=host_texture,
                     texture=tmp_path / "texture.bin", texture_filter=texture_filter,
                     texture_wrap="mirrored-repeat")


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("mode", [None, ("nearest", "clamp-to-edge", "1"),
                                  ("linear", "clamp-to-edge", "2"),
                                  ("nearest", "repeat", "3"), ("linear", "repeat", "4"),
                                  ("nearest", "mirrored-repeat", "5"),
                                  ("linear", "mirrored-repeat", "6")])
def test_helpers_isolate_inherited_state_and_select_exact_versions(monkeypatch, tmp_path, helper, mode):
    for name in FLAGS:
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
        for name in ("MGC_HOST_BLEND", "MGC_HOST_CLEAR_RGBA", "MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST"):
            assert name not in env
        if helper is native_host_readback:
            assert env["MGC_HOST_EXECUTOR_SOCKET"] == str(options["socket_path"])
            assert env["MGC_HOST_WIDTH"] == env["MGC_HOST_HEIGHT"] == "16"
        else:
            assert "MGC_HOST_EXECUTOR_SOCKET" not in env
            assert "MGC_HOST_WIDTH" not in env and "MGC_HOST_HEIGHT" not in env
        assert env.get("MGC_FRAGMENT_STATUS") == (None if helper is softpipe_reference else "1")
        assert env.get("MGC_HOST_DRAW_REPEATS") == (None if helper is capture_draw else "1")
        assert command[3] == {capture_draw: "--draw", native_host_readback: "--host-execute",
                              softpipe_reference: "--reference"}[helper]
        raise InterceptedDraw

    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **options)


@pytest.mark.parametrize("helper", HELPERS)
def test_old_calls_reset_inherited_mirror_to_v1_nearest_clamp(monkeypatch, tmp_path, helper):
    monkeypatch.setenv("MGC_HOST_TEXTURE", "6")
    monkeypatch.setenv("MGC_HOST_TEXTURE_FILTER", "linear")
    monkeypatch.setenv("MGC_HOST_TEXTURE_WRAP", "mirrored-repeat")
    def intercepted(command, *, env, **kwargs):
        assert env["MGC_HOST_TEXTURE_WRAP"] == "clamp-to-edge"
        assert env["MGC_HOST_TEXTURE_FILTER"] == "nearest"
        assert env.get("MGC_HOST_TEXTURE") == (None if helper is softpipe_reference else "1")
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper, tmp_path / "texture.bin"))


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
def test_cli_forwards_mirror_to_capture_and_same_input_reference(monkeypatch, tmp_path, texture_filter):
    modes = []
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    def capture(*args, **kwargs):
        modes.append(("capture", kwargs["texture_filter"], kwargs["texture_wrap"], kwargs["texture"]))
        return {}
    def reference(*args, **kwargs):
        modes.append(("reference", kwargs["texture_filter"], kwargs["texture_wrap"], kwargs["texture"]))
        return image
    monkeypatch.setattr(run_render, "capture_draw", capture)
    monkeypatch.setattr(run_render, "softpipe_reference", reference)
    monkeypatch.setattr(run_render, "render_artifact",
                        lambda *args, **kwargs: (image, np.ones((16, 16), np.uint32), {}))
    monkeypatch.setattr(run_render, "write_png", lambda *args, **kwargs: None)
    texture = tmp_path / "texture.bin"
    assert run_render.main([str(FIXTURE), "--host-texture", "--texture", str(texture),
                            "--texture-filter", texture_filter, "--texture-wrap", "mirrored-repeat",
                            "--compare-softpipe", "--output", str(tmp_path)]) == 0
    assert modes == [("capture", texture_filter, "mirrored-repeat", texture),
                     ("reference", texture_filter, "mirrored-repeat", texture)]


def test_cli_rejects_saved_capture_mirrored_override_and_documents_opt_in(tmp_path, capsys):
    assert run_render.main(["--capture", str(tmp_path / "absent.json"), "--texture-wrap", "mirrored-repeat"]) == 1
    assert "require a fresh shader draw" in capsys.readouterr().err
    with pytest.raises(SystemExit) as status:
        run_render.main(["--help"])
    assert status.value.code == 0
    assert "--texture-wrap {clamp-to-edge,repeat,mirrored-repeat}" in capsys.readouterr().out


@pytest.mark.parametrize("coordinate,expected", [(0, 0), (1, 1), (2, 0), (-1, 1), (-2, 0),
                                               (Fraction(-1, 4), Fraction(1, 4)),
                                               (Fraction(-9, 4), Fraction(1, 4)),
                                               (Fraction(13, 4), Fraction(3, 4))])
def test_exact_period_two_fold(coordinate, expected):
    assert acceptance.mirror_fold(coordinate) == expected
    assert isinstance(acceptance.mirror_fold(coordinate), Fraction)


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
@pytest.mark.parametrize("width,height", [(1, 1), (1, 16), (16, 1), (2, 2), (3, 5), (5, 3), (16, 16)])
def test_rational_centers_reflections_and_multiple_periods(width, height, texture_filter):
    texels = acceptance.input_texels(width, height)
    original = texels.copy()
    for row in range(height):
        for column in range(width):
            horizontal, vertical = Fraction(2 * column + 1, 2 * width), Fraction(2 * row + 1, 2 * height)
            expected = tuple(int(channel) for channel in texels[row, column])
            for translated_horizontal, translated_vertical in ((horizontal, vertical),
                                                               (horizontal - 6, vertical + 4),
                                                               (-horizontal, 2 - vertical),
                                                               (2 - horizontal, -vertical - 4)):
                assert acceptance.rational_sample(texels, translated_horizontal, translated_vertical,
                                                   texture_filter) == expected
            assert acceptance.rational_sample(texels, horizontal + 1, vertical, texture_filter) == tuple(int(channel) for channel in texels[row, width - 1 - column])
            assert acceptance.rational_sample(texels, horizontal, vertical - 1, texture_filter) == tuple(int(channel) for channel in texels[height - 1 - row, column])
            assert acceptance.rational_sample(texels, horizontal + 3, vertical - 3, texture_filter) == tuple(int(channel) for channel in texels[height - 1 - row, width - 1 - column])
    assert np.array_equal(texels, original)


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
@pytest.mark.parametrize("horizontal,vertical,column,row", [(0, 0, 0, 0), (1, 1, 1, 1),
                                                           (-1, 2, 1, 0), (2, -1, 0, 1),
                                                           (-4, 5, 0, 1), (3, -3, 1, 1)])
def test_even_odd_seams_clamp_to_edge_without_repeat_border_mix(texture_filter, horizontal, vertical,
                                                              column, row):
    texels = acceptance.input_texels(2, 2)
    assert acceptance.rational_sample(texels, horizontal, vertical, texture_filter) == tuple(int(channel) for channel in texels[row, column])
    if texture_filter == "linear":
        assert acceptance.rational_sample(texels, horizontal, vertical, texture_filter) != acceptance.repeat_sample(texels, horizontal, vertical, texture_filter)


def test_bilinear_fraction_weights_and_byte_rounding_are_independent(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mirror oracle must not use production packing/rendering or other wrap samplers")
    monkeypatch.setattr(acceptance, "clamp_sample", forbidden)
    monkeypatch.setattr(acceptance, "repeat_sample", forbidden)
    monkeypatch.setattr(acceptance, "render_artifact", forbidden)
    texels = acceptance.input_texels(2, 2)
    assert acceptance.rational_sample(texels, Fraction(-1, 2), Fraction(5, 2)) == (
        Fraction(255, 4), Fraction(255, 4), Fraction(255, 4), Fraction(383, 4))
    image, coverage = acceptance.rational_oracle(texels, phase="identity")
    assert image.dtype == np.uint8 and np.all(coverage == 1)
    assert all(round(channel) == expected for channel, expected in
               zip(acceptance.rational_sample(texels, Fraction(1, 2), Fraction(1, 2)), (64, 64, 64, 96)))


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
def test_negative_tiny_coordinates_remain_exact_and_bounded(texture_filter):
    texels = acceptance.input_texels(3, 5)
    coordinate = Fraction(-1, 1 << 24)
    assert acceptance.rational_sample(texels, coordinate, coordinate, texture_filter) == tuple(int(channel) for channel in texels[0, 0])


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
def test_zero_alpha_rgb_is_legitimate_and_coverage_is_not_alpha(texture_filter):
    texels = acceptance.input_texels(1, 1)
    image, coverage = acceptance.rational_oracle(texels, texture_filter=texture_filter)
    assert np.all(image == [13, 127, 250, 0]) and np.all(coverage == 1)
    image, coverage = acceptance.rational_oracle(texels, texture_filter=texture_filter, discard=True)
    assert np.all(image[:, :8] == 0) and np.all(image[:, 8:] == [13, 127, 250, 0])
    assert np.all(coverage[:, :8] == 0) and np.all(coverage[:, 8:] == 1)


def test_coordinate_cases_hit_negative_multiple_even_and_odd_seams():
    assert acceptance.mirrored_coordinates(15, 0, "seams") == (0, 0)
    assert acceptance.mirrored_coordinates(7, 8, "seams") == (1, 1)
    assert acceptance.mirrored_coordinates(15, 0, "negative-seams") == (-1, -1)
    assert acceptance.mirrored_coordinates(7, 8, "even-odd-seams") == (2, 2)
    assert acceptance.mirrored_coordinates(15, 0, "multiple-periods") == (-4, -4)
    assert acceptance.mirrored_coordinates(3, 12, "multiple-periods") == (5, 5)


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_named_cases_have_independent_wrap_and_filter_witnesses(case):
    texels = acceptance.input_texels(case.width, case.height)
    options = {"phase": case.phase, "texture_filter": case.texture_filter, "discard": case.variant != "sample"}
    mirror, coverage = acceptance.rational_oracle(texels, **options)
    for wrap in ("repeat", "clamp-to-edge"):
        control, control_coverage = acceptance.rational_oracle(texels, texture_wrap=wrap, **options)
        assert np.array_equal(coverage, control_coverage)
        if case.width > 1:
            assert acceptance.image_comparison(mirror, control, 0)["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR
        else:
            assert np.array_equal(mirror, control)
    if case.texture_filter == "linear" and case.width > 1:
        nearest, _ = acceptance.rational_oracle(texels, **{**options, "texture_filter": "nearest"})
        if case.width == 4 and case.phase == "clamp":
            assert np.array_equal(mirror, nearest)
        else:
            assert acceptance.image_comparison(mirror, nearest, 0)["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR


@pytest.mark.parametrize("cases", [(), (acceptance.CASES[0], acceptance.CASES[0]),
                                  (replace(acceptance.CASES[0], name="unregistered"),)])
def test_acceptance_rejects_invalid_selection_before_gl(monkeypatch, tmp_path, cases):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid acceptance selection must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="distinct named bounded"):
        acceptance.validate(tmp_path, BUILD, cases=cases)


@pytest.mark.parametrize("texels", [np.zeros((0, 1, 4), np.uint8), np.zeros((17, 1, 4), np.uint8),
                                   np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 4), np.float16), []])
def test_oracle_rejects_invalid_texels(texels):
    with pytest.raises(ValueError, match="bounded"):
        acceptance.rational_sample(texels, 0, 0)
    with pytest.raises(ValueError, match="bounded"):
        acceptance.rational_oracle(texels)


@pytest.mark.parametrize("options", [{"discard": 1}, {"discard": None}, {"texture_filter": "LINEAR"},
                                    {"texture_wrap": "mirror"}, {"phase": "unknown"}])
def test_oracle_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        acceptance.rational_oracle(acceptance.input_texels(2, 2), **options)


def test_acceptance_records_failure_and_cli_selection(monkeypatch, tmp_path, capsys):
    def fail(*args, **kwargs):
        raise ValueError("exact mirrored failure")
    monkeypatch.setattr(acceptance, "_run_case", fail)
    with pytest.raises(ValueError, match="exact mirrored failure"):
        acceptance.validate(tmp_path, BUILD, cases=(acceptance.CASES[0],))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] is False and summary["status"] == "error"
    assert summary["message"] == "exact mirrored failure" and summary["failed_case"] == acceptance.CASES[0].name
    assert summary["native_gl_readback"] is False and summary["fpga_execution"] is False
    with pytest.raises(SystemExit) as status:
        acceptance.main(["--help"])
    assert status.value.code == 0 and "--case" in capsys.readouterr().out
    assert acceptance.main(["--output", str(tmp_path), "--case", acceptance.CASES[0].name,
                            "--case", acceptance.CASES[0].name]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "error"


def test_default_subset_and_extended_cases_preserve_required_coverage():
    assert len(acceptance.DEFAULT_CASES) == 10 and set(acceptance.DEFAULT_CASES).issubset(acceptance.CASES)
    for texture_filter in ("nearest", "linear"):
        core = [case for case in acceptance.DEFAULT_CASES if case.texture_filter == texture_filter]
        assert {(case.width, case.height) for case in core} == {(1, 1), (2, 2), (3, 5), (5, 3), (16, 16)}
        assert {case.phase for case in core} == {"clamp", "negative-seams", "even-odd-seams"}
        assert any(case.depth_test and case.repeats == 2 and case.variant == "runtime-loop" for case in core)
        full = [case for case in acceptance.CASES if case.texture_filter == texture_filter]
        assert {case.variant for case in full} == {"sample", "early", "late", "runtime-loop"}
        assert any(case.phase == "multiple-periods" for case in full)


def harness_environment(tmp_path, driver="softpipe"):
    if not HARNESS.is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("parent must rebuild Mesa and the GL harness")
    environment = run_render.mesa_environment(BUILD, tmp_path)
    for flag in FLAGS:
        environment.pop(flag, None)
    environment["GALLIUM_DRIVER"] = driver
    return environment


@pytest.mark.parametrize("name,value,message", [("MGC_HOST_TEXTURE", value, "must be 1, 2, 3, 4, 5 or 6")
                                               for value in ("11", "0", "05", "5 ", "+5", "", "6.0")] +
                         [("MGC_HOST_TEXTURE_WRAP", value, "must be clamp-to-edge or repeat or mirrored-repeat")
                          for value in ("", "MIRRORED-REPEAT", "mirror-repeat", "mirrored-repeat ")] +
                         [("MGC_HOST_TEXTURE", "5", "requires explicit MGC_HOST_TEXTURE_DATA"),
                          ("MGC_HOST_TEXTURE", "6", "requires explicit MGC_HOST_TEXTURE_DATA"),
                          ("MGC_HOST_TEXTURE_WRAP", "mirrored-repeat", "requires explicit MGC_HOST_TEXTURE_DATA")])
def test_real_harness_rejects_invalid_flags(tmp_path, name, value, message):
    environment = harness_environment(tmp_path)
    environment[name] = value
    raw = tmp_path / "rejected.rgba8"
    completed = subprocess.run([str(HARNESS), str(FIXTURE / "vert.glsl"), str(FIXTURE / "frag.glsl"),
                                "--reference", str(raw)], env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0 and message in completed.stderr, completed.stdout + completed.stderr
    assert not raw.exists()


@pytest.mark.parametrize("driver", ["mgc", "softpipe"])
@pytest.mark.parametrize("version,texture_filter", [("5", "nearest"), ("6", "linear")])
def test_real_harness_infers_mirror_sampler_from_exact_version(tmp_path, driver, version, texture_filter):
    environment = harness_environment(tmp_path, driver)
    texels = acceptance.input_texels(3, 5)
    texture = tmp_path / "texture.bin"
    texture.write_bytes(struct.pack("<II", 3, 5) + texels.tobytes())
    case = acceptance.CASES[1]
    mesh = tmp_path / "vertices.bin"
    mesh.write_bytes(acceptance.case_mesh(case))
    environment.update(MGC_HOST_TEXTURE=version, MGC_HOST_TEXTURE_DATA=str(texture), MGC_HOST_MESH=str(mesh))
    raw = tmp_path / "inferred.rgba8"
    if driver == "mgc":
        captures = tmp_path / "captures"
        captures.mkdir()
        environment.update(MGC_HOST_CAPTURE_DIR=str(captures), MGC_FRAGMENT_STATUS="1")
        arguments = ["--draw"]
    else:
        arguments = ["--reference", str(raw)]
    shader = FIXTURE / "clamp"
    completed = subprocess.run([str(HARNESS), str(shader / "vert.glsl"), str(shader / "frag.glsl"), *arguments],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if driver == "mgc":
        paths = list(captures.glob("*.json"))
        assert len(paths) == 1
        artifact = json.loads(paths[0].read_text())
        acceptance.check_contract(artifact, texels, replace(case, texture_filter=texture_filter))
    else:
        image = np.frombuffer(raw.read_bytes(), np.uint8).reshape(16, 16, 4)[::-1]
        expected, _ = acceptance.rational_oracle(texels, texture_filter=texture_filter)
        tolerance = acceptance.MAX_CHANNEL_ERROR if texture_filter == "linear" else 0
        assert acceptance.image_comparison(image, expected, tolerance)["passed"]


@pytest.mark.parametrize("version,texture_filter,texture_wrap", [("5", "nearest", "repeat"),
                                                                ("5", "linear", "mirrored-repeat"),
                                                                ("5", "nearest", "clamp-to-edge"),
                                                                ("6", "nearest", "mirrored-repeat"),
                                                                ("6", "linear", "repeat"),
                                                                ("4", "linear", "mirrored-repeat")])
def test_real_mgc_harness_rejects_sampler_version_mismatch(tmp_path, version, texture_filter, texture_wrap):
    environment = harness_environment(tmp_path, "mgc")
    texture = tmp_path / "texture.bin"
    texture.write_bytes(struct.pack("<II", 2, 2) + acceptance.input_texels(2, 2).tobytes())
    environment.update(MGC_HOST_TEXTURE=version, MGC_HOST_TEXTURE_DATA=str(texture),
                       MGC_HOST_TEXTURE_FILTER=texture_filter, MGC_HOST_TEXTURE_WRAP=texture_wrap)
    completed = subprocess.run([str(HARNESS), str(FIXTURE / "vert.glsl"), str(FIXTURE / "frag.glsl"), "--draw"],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0 and "filter/wrap must match" in completed.stderr, completed.stdout + completed.stderr


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_real_softpipe_full_rgba_same_input_oracle(monkeypatch, tmp_path, case):
    harness_environment(tmp_path)
    monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
    shader = acceptance._shader(case, tmp_path)
    texture, mesh = tmp_path / "texture.bin", tmp_path / "vertices.bin"
    texels = acceptance.input_texels(case.width, case.height)
    texture.write_bytes(struct.pack("<II", case.width, case.height) + texels.tobytes())
    mesh.write_bytes(acceptance.case_mesh(case))
    actual = softpipe_reference(shader, BUILD, tmp_path / "reference", mesh=mesh, texture=texture,
                                texture_filter=case.texture_filter, texture_wrap="mirrored-repeat",
                                depth_test=case.depth_test, repeats=case.repeats)
    expected, coverage = acceptance.rational_oracle(texels, phase=case.phase,
                                                   texture_filter=case.texture_filter,
                                                   discard=case.variant != "sample")
    tolerance = acceptance.MAX_CHANNEL_ERROR if case.texture_filter == "linear" else 0
    assert acceptance.image_comparison(actual, expected, tolerance)["passed"]
    assert coverage.sum() == (256 if case.variant == "sample" else 128)
    assert (tmp_path / "reference/softpipe.rgba8").stat().st_size == 16 * 16 * 4


@pytest.fixture(scope="module")
def fresh_mirrored_texture(tmp_path_factory):
    if not HARNESS.is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("parent must rebuild Mesa lowering/capture and the GL harness first")
    output = tmp_path_factory.mktemp("fresh-mirrored-texture")
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
        monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
        monkeypatch.setattr(Driver, "submit", submit)
        monkeypatch.setattr(run_render.subprocess, "run", actual_gl)
        summary = acceptance.validate(output, BUILD, cases=acceptance.CASES)
    return output, summary, submissions, draws


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_fresh_full_rgba_reflection_managed_native_and_softpipe(case, fresh_mirrored_texture):
    output, summary, submissions, draws = fresh_mirrored_texture
    result = next(result for result in summary["cases"] if result["case"] == case.name)
    directory = output / case.name
    assert result["passed"] and result["texture_wrap"] == "mirrored-repeat"
    assert result["host_texture_version"] == (6 if case.texture_filter == "linear" else 5)
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
    assert ("--draw", "mgc", version, case.texture_filter, "mirrored-repeat") in draws
    assert ("--host-execute", "mgc", version, case.texture_filter, "mirrored-repeat") in draws
    assert ("--reference", "softpipe", None, case.texture_filter, "mirrored-repeat") in draws
    resources = result["resources"]
    assert resources["aggregate_draw_step_budget"] == acceptance.MAX_DRAW_STEPS
    assert all(0 < stage["instruction_words"] and stage["maximum_scalar_register"] <= 31
               for stage in resources["stages"].values())
    assert all(256 < report["host_steps"] <= acceptance.MAX_DRAW_STEPS for report in result["native_draw_reports"])
    if result["interpolation_required"]:
        assert result["native_vs_nearest"]["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR
        assert result["softpipe_vs_nearest"]["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR


def test_fresh_summary_and_actual_same_input_clamp_repeat_controls(fresh_mirrored_texture):
    output, summary, submissions, draws = fresh_mirrored_texture
    assert summary["passed"] and summary["tested_cases"] == [case.name for case in acceptance.CASES]
    assert summary["native_gl_readback"] and summary["fpga_execution"] is False
    assert summary["exact_fp32_conformance"] is False and "binary16" in summary["input_precision"]
    assert "binary16" in summary["output_precision"] and summary["maximum_channel_error"] <= acceptance.MAX_CHANNEL_ERROR
    assert len(submissions) >= len(acceptance.CASES) * 2
    for texture_filter in ("nearest", "linear"):
        result = next(result for result in summary["cases"] if result["case"] == f"{texture_filter}-mirrored-repeat-2x2")
        assert set(result["actual_wrap_controls"]) == {"clamp-to-edge", "repeat"}
        for wrap, control in result["actual_wrap_controls"].items():
            assert control["same_input"] and control["native_gl_readback"]
            assert control["mirror_vs_control_native"]["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR
            assert control["mirror_vs_control_softpipe"]["max_channel_error"] > acceptance.MAX_CHANNEL_ERROR
            assert ("--host-execute", "mgc", str(control["host_texture_version"]), texture_filter, wrap) in draws


@pytest.mark.parametrize("texture_filter", ["nearest", "linear"])
@pytest.mark.parametrize("size", [1, 2])
def test_fresh_repeated_z16_kill_and_equal_depth_commit_counts(fresh_mirrored_texture, texture_filter, size):
    output, summary, submissions, draws = fresh_mirrored_texture
    name = f"{texture_filter}-mirrored-repeat-depth-loop-repeated-{size}x{size}"
    result = next(result for result in summary["cases"] if result["case"] == name)
    first, second = result["native_draw_reports"]
    assert first["committed_fragments"] == 128 and first["discarded_fragments"] == 256
    assert second["committed_fragments"] == 0 and second["discarded_fragments"] == 256
    assert second["covered_pixels"] == 0 and second["max_overdraw"] == 0
    depth = np.frombuffer((output / name / "native/depth.z16").read_bytes(), "<u2").reshape(16, 16)
    assert np.all(depth[:, :8] == 65535) and np.all(depth[:, 8:] == 16384)
    if size == 1:
        native = np.frombuffer((output / name / "native/native.rgba8").read_bytes(), np.uint8).reshape(16, 16, 4)[::-1]
        assert np.all(native[:, :8] == 0) and np.all(native[:, 8:] == [13, 127, 250, 0])


@pytest.mark.parametrize("mutation", ["wrap", "version", "abi", "texels", "filter"])
def test_fresh_contract_rejects_mismatch_without_relaxing_oracle(fresh_mirrored_texture, mutation):
    output, summary, submissions, draws = fresh_mirrored_texture
    case = acceptance.CASES[0]
    artifact = copy.deepcopy(json.loads((output / case.name / "capture/draw.json").read_text()))
    if mutation == "wrap":
        artifact["textures"][0]["sampler"]["wrap_t"] = "repeat"
    elif mutation == "version":
        artifact["shaders"]["fs"]["layout"]["host_texture"]["version"] = 11
    elif mutation == "abi":
        artifact["shader_abi_version"] = 2
    elif mutation == "texels":
        artifact["textures"][0]["data_hex"] = "00" * (case.width * case.height * 4)
    else:
        artifact["textures"][0]["sampler"]["min_filter"] = "linear"
    with pytest.raises(ValueError):
        acceptance.check_contract(artifact, acceptance.input_texels(case.width, case.height), case)
