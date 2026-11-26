import json
from fractions import Fraction
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tools.blend_acceptance import CASES, analytic_oracle, validate
from tools.host_blend import blend_rgba8
from tools.host_render import render_artifact
from tools.mgcdriver import Driver, ProgramFormat
from tools import run_render
from tools.run_render import capture_draw, softpipe_reference
from tools.run_shader import ROOT, mesa_environment


FIXTURE = ROOT / "tests/fixtures/host-blend"
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))


class BlendString(str):
    pass


BAD_MODES = [None, False, True, 0, 1, "", "source_over", "SOURCE-OVER", "add", "linear",
             {}, [], b"source-over", BlendString("source-over")]


@pytest.mark.parametrize("source_channel", (0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 1))
@pytest.mark.parametrize("alpha", (0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 1))
def test_source_over_rounds_exact_stored_byte_ties_without_normalization_drift(source_channel, alpha):
    source = (float(source_channel),) * 3 + (float(alpha),)
    for destination_byte in range(256):
        destination = np.full(4, destination_byte, np.uint8)
        actual = blend_rgba8(source, destination, "source-over")
        expected_color = round(Fraction(source_channel) * alpha * 255 + destination_byte * (1 - alpha))
        expected_alpha = round(alpha * 255 + destination_byte * (1 - alpha))
        assert actual.tolist() == [expected_color] * 3 + [expected_alpha]


@pytest.mark.parametrize("blend", BAD_MODES)
def test_renderer_rejects_nonexact_and_unsupported_blend_modes(blend):
    with pytest.raises(ValueError, match="blend"):
        render_artifact({}, blend=blend)


@pytest.mark.parametrize("blend", BAD_MODES)
@pytest.mark.parametrize("helper", [capture_draw, softpipe_reference])
def test_helpers_reject_unsupported_blend_before_gl_invocation(monkeypatch, tmp_path, blend, helper):
    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported blend must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="blend"):
        helper(FIXTURE, BUILD, tmp_path, blend=blend)


@pytest.mark.parametrize("blend", [None, "disabled", "source-over"])
def test_capture_blend_is_explicit_and_inherited_state_is_cleared(monkeypatch, tmp_path, blend):
    monkeypatch.setenv("MGC_HOST_BLEND", "untrusted")
    mesh = tmp_path / "mesh.bin"
    texture = tmp_path / "texture.bin"
    def fake_run(command, *, env, **kwargs):
        assert len(command) == 4 and command[-1] == "--draw"
        assert env["MGC_HOST_MESH"] == str(mesh.resolve())
        assert env["MGC_HOST_DEPTH_TEST"] == env["MGC_FRAGMENT_STATUS"] == "1"
        assert env["MGC_HOST_TEXTURE"] == "1" and env["MGC_HOST_TEXTURE_DATA"] == str(texture.absolute())
        if blend == "source-over":
            assert env["MGC_HOST_BLEND"] == "source-over"
        else:
            assert "MGC_HOST_BLEND" not in env
        (Path(env["MGC_HOST_CAPTURE_DIR"]) / "draw.json").write_text('{"schema":"blend-helper-test"}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    options = {} if blend is None else {"blend": blend}
    assert capture_draw(FIXTURE, BUILD, tmp_path / "output", mesh=mesh, depth_test=True,
                        host_texture=True, texture=texture, **options) == {"schema": "blend-helper-test"}


@pytest.mark.parametrize("blend", [None, "disabled", "source-over"])
def test_reference_blend_matches_explicit_input_and_preserves_texture_depth(monkeypatch, tmp_path, blend):
    monkeypatch.setenv("MGC_HOST_BLEND", "untrusted")
    mesh = tmp_path / "mesh.bin"
    texture = tmp_path / "texture.bin"
    def fake_run(command, *, env, **kwargs):
        assert len(command) == 5 and command[-2] == "--reference"
        assert env["GALLIUM_DRIVER"] == "softpipe"
        assert env["MGC_HOST_MESH"] == str(mesh.resolve()) and env["MGC_HOST_DEPTH_TEST"] == "1"
        assert env["MGC_HOST_TEXTURE_DATA"] == str(texture.absolute())
        assert "MGC_HOST_TEXTURE" not in env and "MGC_FRAGMENT_STATUS" not in env
        if blend == "source-over":
            assert env["MGC_HOST_BLEND"] == "source-over"
        else:
            assert "MGC_HOST_BLEND" not in env
        Path(command[-1]).write_bytes(bytes(16 * 16 * 4))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    options = {} if blend is None else {"blend": blend}
    assert softpipe_reference(FIXTURE, BUILD, tmp_path / "output", mesh=mesh,
                              depth_test=True, texture=texture, **options).shape == (16, 16, 4)


@pytest.mark.parametrize("blend", [None, "disabled", "source-over"])
def test_cli_blend_forwards_capture_renderer_and_reference(monkeypatch, tmp_path, blend):
    calls = []
    selected = "disabled" if blend is None else blend
    def fake_capture(*args, **kwargs):
        assert kwargs["blend"] == selected
        calls.append("capture")
        return {}
    def fake_render(*args, **kwargs):
        assert kwargs["blend"] == selected
        calls.append("render")
        return np.zeros((16, 16, 4), dtype=np.uint8), np.ones((16, 16), dtype=np.uint32), {"blend": selected}
    def fake_reference(*args, **kwargs):
        assert kwargs["blend"] == selected
        calls.append("reference")
        return np.zeros((16, 16, 4), dtype=np.uint8)
    monkeypatch.setattr(run_render, "capture_draw", fake_capture)
    monkeypatch.setattr(run_render, "render_artifact", fake_render)
    monkeypatch.setattr(run_render, "softpipe_reference", fake_reference)
    arguments = [str(FIXTURE), "--compare-softpipe", "--output", str(tmp_path / "output")]
    if blend is not None:
        arguments.extend(("--blend", blend))
    assert run_render.main(arguments) == 0 and calls == ["capture", "render", "reference"]


def test_cli_rejects_unsupported_blend():
    with pytest.raises(SystemExit) as result:
        run_render.main([str(FIXTURE), "--blend", "add"])
    assert result.value.code == 2


@pytest.mark.parametrize("depths,expected", [((-0.5, 0.5), [64, 0, 128, 192]),
                                            ((0.5, -0.5), [128, 0, 64, 192]),
                                            ((-0.5,), [128, 0, 0, 128])])
def test_rational_oracle_separate_alpha_factors_and_per_commit_rounding(depths, expected):
    image, coverage, counts = analytic_oracle(depths)
    assert np.all(image == expected) and np.all(coverage == len(depths))
    assert counts["fragments"] == counts["committed_fragments"] == 256 * len(depths)


def test_rational_oracle_zero_alpha_commits_depth_and_discard_does_not():
    image, coverage, counts = analytic_oracle((-0.5, 0.5), "zero-alpha.frag.glsl", depth_test=True)
    assert not np.any(image) and np.all(coverage == 1) and counts["fragments"] == 256
    image, coverage, counts = analytic_oracle((-0.5, 0.5), "discard-zero-alpha.frag.glsl", depth_test=True)
    assert np.all(image[:, :8] == [0, 0, 255, 255]) and not np.any(image[:, 8:])
    assert np.all(coverage == 1) and counts == {"fragments": 384, "discarded_fragments": 128,
                                               "committed_fragments": 256}


@pytest.fixture(scope="module")
def fresh_blend_runs(tmp_path_factory):
    output = tmp_path_factory.mktemp("fresh-host-blend")
    submissions = []
    original = Driver.submit
    def actual_submit(driver, program, *args, **kwargs):
        submissions.append(program.format)
        return original(driver, program, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Driver, "submit", actual_submit)
        summary = validate(output, BUILD)
    return output, summary, submissions


@pytest.mark.parametrize("case", [case[0] for case in CASES])
def test_fresh_blend_actual_compiled_driver_softpipe_and_analytic_oracles(fresh_blend_runs, case):
    output, summary, submissions = fresh_blend_runs
    report = next(result for result in summary["cases"] if result["case"] == case)
    artifact = json.loads((output / case / "draw.json").read_text())
    assert artifact["schema"] == "mgc-host-draw-v1" and artifact["shader_abi_version"] == 1
    assert report["blend"] in ("disabled", "source-over") and report["fpga_execution"] is False
    assert report["softpipe_comparison"]["passed"] and report["softpipe_comparison"]["max_channel_error"] <= 1
    assert report["softpipe_comparison"]["channel_tolerance"] == 1
    assert report["committed_fragments"] + report["discarded_fragments"] == report["fragments"]
    assert len(submissions) == sum(result["vertices"] + result["fragments"] for result in summary["cases"])
    assert all(target == ProgramFormat.LEGACY for target in submissions)
    assert all(Path(path).exists() for path in report["evidence"].values())


def test_fresh_default_disabled_remains_overwrite_not_source_over(fresh_blend_runs):
    output, _, _ = fresh_blend_runs
    artifact = json.loads((output / "disabled-near-first/draw.json").read_text())
    default, coverage, report = render_artifact(artifact)
    explicit, explicit_coverage, explicit_report = render_artifact(artifact, blend="disabled")
    blended, _, _ = render_artifact(artifact, blend="source-over")
    assert np.array_equal(default, explicit) and np.array_equal(coverage, explicit_coverage)
    assert np.all(default == [0, 0, 255, 128]) and np.all(blended == [64, 0, 128, 192])
    assert report["blend"] == explicit_report["blend"] == "disabled"


def test_fresh_alpha_zero_keeps_color_but_commits_depth_and_coverage(fresh_blend_runs):
    output, summary, _ = fresh_blend_runs
    for name, color, commits in [("zero-alpha-depth-near-first", [0, 0, 0, 0], 1),
                                  ("zero-alpha-depth-far-first", [0, 0, 255, 255], 2)]:
        image = np.frombuffer((output / name / "host.rgba8").read_bytes(), dtype=np.uint8).reshape(16, 16, 4)
        coverage = np.frombuffer((output / name / "coverage.u32").read_bytes(), dtype="<u4")
        report = next(result for result in summary["cases"] if result["case"] == name)
        assert np.all(image == color) and np.all(coverage == commits)
        assert report["fragments"] == report["committed_fragments"] == 256 * commits


@pytest.mark.parametrize("name", ["discard-near-first", "discard-far-first", "discard-no-depth",
                                  "discard-zero-alpha-near-first", "discard-zero-alpha-far-first"])
def test_fresh_discard_never_blends_or_changes_depth_or_coverage(fresh_blend_runs, name):
    output, summary, _ = fresh_blend_runs
    report = next(result for result in summary["cases"] if result["case"] == name)
    assert report["discarded_fragments"] == 128
    coverage = np.frombuffer((output / name / "coverage.u32").read_bytes(), dtype="<u4").reshape(16, 16)
    assert np.all(coverage[:, :8] == 1)
    assert np.all(coverage[:, 8:] == (1 if name.endswith("near-first") else 2))


@pytest.mark.parametrize("budget", [{"max_steps": 1}, {"max_fragments": 1}])
def test_fresh_blend_instruction_and_fragment_budgets_fail_closed(fresh_blend_runs, budget):
    output, _, _ = fresh_blend_runs
    artifact = json.loads((output / "source-over-near-first/draw.json").read_text())
    with pytest.raises(ValueError, match="budget"):
        render_artifact(artifact, blend="source-over", **budget)


@pytest.mark.parametrize("blend", ["", "add", "source_over", "SOURCE-OVER", "1"])
@pytest.mark.parametrize("mode", ["compile-only", "draw", "reference"])
def test_fresh_harness_rejects_unknown_blend_environment(tmp_path, blend, mode):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment["GALLIUM_DRIVER"] = "softpipe"
    environment["MGC_HOST_BLEND"] = blend
    for name in ("MGC_HOST_CAPTURE_DIR", "MGC_ASM_DUMP", "MGC_HOST_TEXTURE_DATA", "MGC_HOST_DEPTH_TEST", "MGC_HOST_MESH"):
        environment.pop(name, None)
    command = [str(ROOT / "tools/mgc_harness"), str(FIXTURE / "vert.glsl"), str(FIXTURE / "frag.glsl")]
    if mode == "draw":
        command.append("--draw")
    elif mode == "reference":
        command.extend(("--reference", str(tmp_path / "reference.rgba8")))
    completed = subprocess.run(command,
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode != 0 and "unsupported MGC_HOST_BLEND" in completed.stderr


def test_fresh_harness_compile_only_three_arguments_remain_supported(tmp_path):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment["MGC_HOST_BLEND"] = "source-over"
    for name in ("MGC_HOST_CAPTURE_DIR", "MGC_HOST_TEXTURE_DATA", "MGC_HOST_DEPTH_TEST", "MGC_HOST_MESH"):
        environment.pop(name, None)
    completed = subprocess.run([str(ROOT / "tools/mgc_harness"), str(FIXTURE / "vert.glsl"),
                                str(FIXTURE / "frag.glsl")], env=environment,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
