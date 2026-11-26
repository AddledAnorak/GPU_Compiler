"""Strict independent S/T API, rational oracle and fresh actual GL acceptance."""

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

from tools import mixed_texture_acceptance as acceptance
from tools import run_render
from tools.linear_texture_acceptance import rational_sample as clamp_sample
from tools.mirrored_texture_acceptance import rational_sample as mirrored_sample
from tools.repeat_texture_acceptance import rational_sample as repeat_sample
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
FIXTURE = ROOT / "tests/fixtures/host-texture"
HARNESS = Path(os.environ.get("MGC_MIXED_TEST_HARNESS", ROOT / "tools/mgc_harness"))
HELPERS = (capture_draw, softpipe_reference, native_host_readback)
FLAGS = ("MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_DATA", "MGC_HOST_TEXTURE_FILTER", "MGC_HOST_TEXTURE_WRAP",
         "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T", "MGC_HOST_CAPTURE_DIR", "MGC_ASM_DUMP",
         "MGC_FRAGMENT_STATUS", "MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST",
         "MGC_HOST_BLEND", "MGC_HOST_CLEAR_RGBA", "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT", "MGC_HOST_DRAW_REPEATS",
         "MGC_HOST_GLSL_PROFILE", "MGC_HOST_MIPMAPS", "MGC_HOST_FEATURES")


class InterceptedDraw(Exception):
    pass


class WrapString(str):
    pass


def helper_options(helper, texture=None):
    options = {"texture": texture}
    if helper is capture_draw:
        options["host_texture"] = texture is not None
    if helper is native_host_readback:
        options["socket_path"] = Path("/tmp/mgc-mixed-test.sock")
    return options


@pytest.mark.parametrize("helper", HELPERS)
def test_per_axis_contract_is_keyword_only_with_unchanged_defaults(helper):
    parameters = inspect.signature(helper).parameters
    for name in ("texture_wrap_s", "texture_wrap_t"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None
    assert parameters["texture_wrap"].default == "clamp-to-edge"
    assert parameters["texture_filter"].default == "nearest"


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("name", ("texture_wrap", "texture_wrap_s", "texture_wrap_t"))
@pytest.mark.parametrize("invalid", (True, 1, [], {}, "REPEAT", "repeat ", "mirror", WrapString("repeat")))
def test_api_rejects_noncanonical_or_nonexact_string_before_gl(monkeypatch, tmp_path, helper, name, invalid):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid API reached GL"))
    with pytest.raises(ValueError, match="texture wrap must be explicitly"):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper, tmp_path / "texture.bin"), **{name: invalid})


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("options", ({"texture_wrap": None}, {"texture_wrap_s": "repeat"},
                                     {"texture_wrap_t": "mirrored-repeat"}))
def test_invalid_base_or_nondefault_actual_axes_require_texture(monkeypatch, tmp_path, helper, options):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid API reached GL"))
    with pytest.raises(ValueError):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper), **options)


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("texture_filter", ("nearest", "linear"))
@pytest.mark.parametrize("wrap_s,wrap_t", acceptance.AXES)
def test_helpers_clear_inherited_axes_and_emit_only_explicit_mixed_version(monkeypatch, tmp_path, helper,
                                                                         texture_filter, wrap_s, wrap_t):
    for name in FLAGS:
        monkeypatch.setenv(name, "inherited-invalid")
    texture = tmp_path / "texture.bin"
    def intercepted(command, *, env, **kwargs):
        assert env["GALLIUM_DRIVER"] == ("softpipe" if helper is softpipe_reference else "mgc")
        assert env["MGC_HOST_TEXTURE"] == ("8" if texture_filter == "linear" else "7")
        assert env["MGC_HOST_TEXTURE_WRAP_S"] == wrap_s and env["MGC_HOST_TEXTURE_WRAP_T"] == wrap_t
        assert "MGC_HOST_TEXTURE_WRAP" not in env
        assert env["MGC_HOST_TEXTURE_FILTER"] == texture_filter
        assert env["MGC_HOST_TEXTURE_DATA"] == str(texture)
        for name in ("MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST", "MGC_HOST_BLEND", "MGC_HOST_CLEAR_RGBA",
                     "MGC_HOST_GLSL_PROFILE", "MGC_HOST_MIPMAPS"):
            assert name not in env
        assert command[3] == {capture_draw: "--draw", softpipe_reference: "--reference",
                              native_host_readback: "--host-execute"}[helper]
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper, texture), texture_filter=texture_filter,
               texture_wrap_s=wrap_s, texture_wrap_t=wrap_t)


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("texture_filter", ("nearest", "linear"))
@pytest.mark.parametrize("wrap,offset", (("clamp-to-edge", 0), ("repeat", 2), ("mirrored-repeat", 4)))
def test_same_axes_and_none_overrides_preserve_paired_versions(monkeypatch, tmp_path, helper, texture_filter, wrap, offset):
    for name in FLAGS:
        monkeypatch.setenv(name, "inherited-invalid")
    def intercepted(command, *, env, **kwargs):
        assert env["MGC_HOST_TEXTURE_WRAP"] == wrap
        assert "MGC_HOST_TEXTURE_WRAP_S" not in env and "MGC_HOST_TEXTURE_WRAP_T" not in env
        assert "MGC_HOST_GLSL_PROFILE" not in env and "MGC_HOST_MIPMAPS" not in env
        assert env.get("MGC_HOST_TEXTURE") == (None if helper is softpipe_reference else
                                               str(offset + (2 if texture_filter == "linear" else 1)))
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    for axes in ({}, {"texture_wrap_s": wrap}, {"texture_wrap_s": wrap, "texture_wrap_t": wrap}):
        with pytest.raises(InterceptedDraw):
            helper(FIXTURE, BUILD, tmp_path, **helper_options(helper, tmp_path / "texture.bin"),
                   texture_filter=texture_filter, texture_wrap=wrap, **axes)


@pytest.mark.parametrize("helper", HELPERS)
def test_actual_clamp_overrides_need_no_texture_and_clear_inherited_state(monkeypatch, tmp_path, helper):
    for name in FLAGS:
        monkeypatch.setenv(name, "inherited-invalid")
    def intercepted(command, *, env, **kwargs):
        assert not any(name in env for name in FLAGS[:6])
        assert "MGC_HOST_GLSL_PROFILE" not in env and "MGC_HOST_MIPMAPS" not in env
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(FIXTURE, BUILD, tmp_path, **helper_options(helper), texture_wrap="repeat",
               texture_wrap_s="clamp-to-edge", texture_wrap_t="clamp-to-edge")


def test_cli_forwards_axes_and_reports_resolved_mixed_state(monkeypatch, tmp_path):
    calls = []
    image = np.zeros((16, 16, 4), np.uint8)
    def capture(*args, **kwargs):
        calls.append(kwargs)
        return {}
    def reference(*args, **kwargs):
        calls.append(kwargs)
        return image
    monkeypatch.setattr(run_render, "capture_draw", capture)
    monkeypatch.setattr(run_render, "softpipe_reference", reference)
    monkeypatch.setattr(run_render, "render_artifact", lambda *args, **kwargs: (image, np.ones((16, 16), np.uint32), {}))
    monkeypatch.setattr(run_render, "write_png", lambda *args, **kwargs: None)
    assert run_render.main([str(FIXTURE), "--host-texture", "--texture", str(tmp_path / "texture.bin"),
                            "--texture-wrap", "repeat", "--texture-wrap-t", "mirrored-repeat",
                            "--compare-softpipe", "--output", str(tmp_path)]) == 0
    assert len(calls) == 2
    assert all(call["texture_wrap"] == "repeat" and call["texture_wrap_s"] is None and
               call["texture_wrap_t"] == "mirrored-repeat" for call in calls)
    report = json.loads((tmp_path / "render.json").read_text())
    assert (report["texture_wrap"], report["texture_wrap_s"], report["texture_wrap_t"]) == ("mixed", "repeat", "mirrored-repeat")


@pytest.mark.parametrize("axis", ("s", "t"))
def test_cli_rejects_any_saved_capture_axis_override(tmp_path, capsys, axis):
    assert run_render.main(["--capture", str(tmp_path / "absent.json"), f"--texture-wrap-{axis}", "clamp-to-edge"]) == 1
    assert "require a fresh shader draw" in capsys.readouterr().err


@pytest.mark.parametrize("wrap_s,wrap_t", acceptance.AXES)
def test_nearest_negative_coordinates_select_independent_axis_indices(wrap_s, wrap_t):
    texels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    columns = {"clamp-to-edge": 0, "repeat": 2, "mirrored-repeat": 0}
    rows = {"clamp-to-edge": 1, "repeat": 0, "mirrored-repeat": 1}
    actual = acceptance.rational_sample(texels, Fraction(-1, 4), Fraction(5, 4), "nearest",
                                        texture_wrap_s=wrap_s, texture_wrap_t=wrap_t)
    assert actual == tuple(Fraction(int(channel)) for channel in texels[rows[wrap_t], columns[wrap_s]])


def test_linear_seam_taps_wrap_only_the_repeat_axis():
    texels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    actual = acceptance.rational_sample(texels, 0, 1, texture_wrap_s="repeat", texture_wrap_t="clamp-to-edge")
    assert actual == tuple(Fraction(int(first) + int(last), 2) for first, last in zip(texels[1, 0], texels[1, 2]))
    transposed = acceptance.rational_sample(texels, 0, 1, texture_wrap_s="clamp-to-edge", texture_wrap_t="repeat")
    assert transposed == tuple(Fraction(int(first) + int(last), 2) for first, last in zip(texels[0, 0], texels[1, 0]))
    assert actual != transposed


@pytest.mark.parametrize("wrap,sampler", (("clamp-to-edge", clamp_sample), ("repeat", repeat_sample),
                                        ("mirrored-repeat", mirrored_sample)))
@pytest.mark.parametrize("texture_filter", ("nearest", "linear"))
def test_same_axis_fraction_math_matches_retained_independent_oracles(wrap, sampler, texture_filter):
    texels = acceptance.input_texels(3, 5)
    for horizontal in (Fraction(-33, 16), Fraction(-1, 16), 0, 1, Fraction(31, 16), 4):
        for vertical in (Fraction(-17, 16), 0, Fraction(15, 16), 1, Fraction(49, 16)):
            assert acceptance.rational_sample(texels, horizontal, vertical, texture_filter,
                                               texture_wrap_s=wrap, texture_wrap_t=wrap) == sampler(
                                                   texels, horizontal, vertical, texture_filter)


@pytest.mark.parametrize("options", ({"discard": 1}, {"texture_filter": "LINEAR"}, {"texture_wrap_s": None},
                                     {"texture_wrap_t": WrapString("repeat")}, {"phase": "unknown"}))
def test_oracle_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        acceptance.rational_oracle(acceptance.input_texels(2, 2), **options)


def test_bounded_matrix_covers_all_orientations_filters_and_completion_cases():
    assert len(acceptance.DEFAULT_CASES) == 12 and len(acceptance.CASES) == 30
    assert len({case.name for case in acceptance.CASES}) == len(acceptance.CASES)
    for texture_filter in ("nearest", "linear"):
        core = [case for case in acceptance.DEFAULT_CASES if case.texture_filter == texture_filter]
        assert {(case.texture_wrap_s, case.texture_wrap_t) for case in core} == set(acceptance.AXES)
        assert {case.variant for case in core} == {"sample", "early", "late", "runtime-loop"}
        assert any(case.width == case.height == 1 and case.depth_test and case.repeats == 2 for case in core)


def harness_environment(tmp_path, driver="softpipe"):
    if not HARNESS.is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("rebuild Mesa mixed lowering/capture and the GL harness first")
    environment = run_render.mesa_environment(BUILD, tmp_path)
    for name in FLAGS:
        environment.pop(name, None)
    environment["GALLIUM_DRIVER"] = driver
    return environment


@pytest.mark.parametrize("driver", ("mgc", "softpipe"))
@pytest.mark.parametrize("changes", ({"MGC_HOST_TEXTURE_WRAP_S": None}, {"MGC_HOST_TEXTURE_WRAP_T": None},
                                     {"MGC_HOST_TEXTURE_WRAP_S": ""}, {"MGC_HOST_TEXTURE_WRAP_T": "mirror"},
                                     {"MGC_HOST_TEXTURE_WRAP": "repeat"}, {"MGC_HOST_TEXTURE_FILTER": "linear"},
                                     {"MGC_HOST_TEXTURE": "0"}, {"MGC_HOST_TEXTURE": "07"},
                                     {"MGC_HOST_TEXTURE": "11"}, {"MGC_HOST_TEXTURE": "+7"}))
def test_real_harness_rejects_incomplete_conflicting_or_invalid_mixed_state(tmp_path, driver, changes):
    environment = harness_environment(tmp_path, driver)
    texture = tmp_path / "texture.bin"
    texture.write_bytes(struct.pack("<II", 2, 2) + acceptance.input_texels(2, 2).tobytes())
    environment.update(MGC_HOST_TEXTURE="7", MGC_HOST_TEXTURE_DATA=str(texture), MGC_HOST_TEXTURE_FILTER="nearest",
                       MGC_HOST_TEXTURE_WRAP_S="repeat", MGC_HOST_TEXTURE_WRAP_T="clamp-to-edge")
    for name, value in changes.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    completed = subprocess.run([str(HARNESS), str(FIXTURE / "vert.glsl"), str(FIXTURE / "frag.glsl"), "--draw"],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0, completed.stdout + completed.stderr
    if driver == "mgc" and changes == {"MGC_HOST_TEXTURE": "11"}:
        assert "texture versions 11 through 14 require bounded HOST FEATURES" in completed.stderr
    else:
        assert ("MGC_HOST_TEXTURE" in completed.stderr or "filter/wrap" in completed.stderr or
                "sampler/texture ABI is undefined" in completed.stderr)


@pytest.mark.parametrize("version,profile,target", ((None, "desktop-120", "legacy-mgcsim-v1"),
    ("1", "desktop-120", "legacy-mgcsim-v1"), ("7", "desktop-120", "legacy-mgcsim-v1"),
    ("8", "desktop-120", "legacy-mgcsim-v1"), ("9", None, "legacy-mgcsim-v1"),
    ("10", None, "legacy-mgcsim-v1"), ("9", "gles3", "legacy-mgcsim-v1"),
    (None, "unknown", "legacy-mgcsim-v1"), ("9", "desktop-120", "fp16-hardware-v1"),
    ("10", "desktop-120", None)))
def test_real_harness_rejects_profile_conflicts_before_egl(tmp_path, version, profile, target):
    environment = harness_environment(tmp_path, "softpipe")
    for name, value in (("MGC_HOST_TEXTURE", version), ("MGC_HOST_GLSL_PROFILE", profile),
                        ("MGC_COMPILE_TARGET", target)):
        if value is not None:
            environment[name] = value
        else:
            environment.pop(name, None)
    completed = subprocess.run([str(HARNESS), str(FIXTURE / "vert.glsl"), str(FIXTURE / "frag.glsl"), "--draw"],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert "MGC_HOST_GLSL_PROFILE" in completed.stderr or "explicit legacy-mgcsim-v1" in completed.stderr
    assert "GL_RENDERER:" not in completed.stderr


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_real_softpipe_all_30_same_input_full_rgba_oracles(monkeypatch, tmp_path, case):
    harness_environment(tmp_path)
    monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
    shader = acceptance._shader(case, tmp_path)
    texture, mesh = tmp_path / "texture.bin", tmp_path / "vertices.bin"
    texels = acceptance.input_texels(case.width, case.height)
    texture.write_bytes(struct.pack("<II", case.width, case.height) + texels.tobytes())
    mesh.write_bytes(acceptance.case_mesh(case))
    image = softpipe_reference(shader, BUILD, tmp_path / "reference", mesh=mesh, texture=texture,
                               texture_filter=case.texture_filter, texture_wrap_s=case.texture_wrap_s,
                               texture_wrap_t=case.texture_wrap_t, depth_test=case.depth_test, repeats=case.repeats)
    expected, coverage = acceptance.rational_oracle(texels, phase=case.phase, texture_filter=case.texture_filter,
                                                   texture_wrap_s=case.texture_wrap_s, texture_wrap_t=case.texture_wrap_t,
                                                   discard=case.variant != "sample")
    assert acceptance.image_comparison(image, expected, acceptance.MAX_CHANNEL_ERROR if case.texture_filter == "linear" else 0)["passed"]
    assert coverage.sum() == (256 if case.variant == "sample" else 128)


@pytest.fixture(scope="module")
def fresh_mixed_texture(tmp_path_factory):
    output = tmp_path_factory.mktemp("fresh-mixed-texture")
    harness_environment(output)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
        return output, acceptance.validate(output, BUILD)


def test_fresh_capture_managed_native_and_softpipe_default_matrix(fresh_mixed_texture):
    output, summary = fresh_mixed_texture
    assert summary["passed"] and summary["native_gl_readback"]
    assert summary["fpga_execution"] is False and summary["maximum_channel_error"] <= 2
    assert summary["tested_cases"] == [case.name for case in acceptance.DEFAULT_CASES]
    for result in summary["cases"]:
        assert result["comparisons"]["native_vs_managed"]["max_channel_error"] == 0
        assert all(check["passed"] for check in result["comparisons"].values())
        if result["texture_filter"] == "nearest":
            assert result["maximum_channel_error"] == 0
        for name in ("managed", "native", "softpipe", "rational"):
            assert (output / result["case"] / f"{name}.rgba8").stat().st_size == 1024


def test_actual_capture_rejects_axis_metadata_or_sampler_corruption(fresh_mixed_texture):
    output, summary = fresh_mixed_texture
    case = acceptance.DEFAULT_CASES[0]
    artifact = json.loads((output / case.name / "capture/draw.json").read_text())
    texels = acceptance.input_texels(case.width, case.height)
    for section in ("metadata", "sampler"):
        broken = copy.deepcopy(artifact)
        target = broken["shaders"]["fs"]["layout"]["host_texture"] if section == "metadata" else broken["textures"][0]["sampler"]
        target["wrap_s"] = case.texture_wrap_t
        with pytest.raises(ValueError):
            acceptance.check_contract(broken, texels, case)


@pytest.mark.parametrize("cases", ((), (acceptance.CASES[0], acceptance.CASES[0]),
                                  (replace(acceptance.CASES[0], name="unregistered"),)))
def test_invalid_selection_never_reaches_gl(tmp_path, cases):
    with pytest.raises(ValueError, match="distinct named bounded"):
        acceptance.validate(tmp_path, BUILD, cases=cases)


def test_failure_is_retained_without_weakening_reference(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("exact mixed failure")
    monkeypatch.setattr(acceptance, "_run_case", fail)
    with pytest.raises(ValueError, match="exact mixed failure"):
        acceptance.validate(tmp_path, BUILD, cases=(acceptance.CASES[0],))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] is False and summary["status"] == "error"
    assert summary["native_gl_readback"] is False and summary["message"] == "exact mixed failure"
