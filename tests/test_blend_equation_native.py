"""Independent equation oracles and fresh reflected native GL pixel acceptance."""

import inspect
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tools import blend_equation_acceptance as acceptance
from tools import run_render
from tools.host_blend import BLEND_MODES
from tools.mgcdriver import Driver, ProgramFormat
from tools.run_shader import ROOT


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))


@pytest.mark.parametrize("mode", BLEND_MODES[2:])
def test_rational_equations_use_initialized_rgba_and_clamp_signed_results(mode):
    source = (Fraction(3, 2), Fraction(-1, 2), Fraction(1, 4), Fraction(0))
    background = (64, 128, 191, 255)
    expected = {"additive": (255, 128, 255, 255), "subtract": (191, 0, 0, 0),
                "reverse-subtract": (0, 128, 127, 255), "minimum": (64, 0, 64, 0),
                "maximum": (255, 128, 191, 255)}
    assert acceptance.rational_commit(source, background, mode) == expected[mode]


@pytest.mark.parametrize("mode", (None, True, 2, "min", "max", "add", "minimum ", "ADDITIVE"))
def test_rational_equations_reject_nonexact_modes(mode):
    with pytest.raises(ValueError, match="blend"):
        acceptance.rational_commit((0, 0, 0, 0), (0, 0, 0, 0), mode)


@pytest.mark.parametrize("mode", BLEND_MODES[2:])
def test_rational_kill_does_not_commit_color_or_depth_and_alpha_zero_is_valid(mode):
    case = acceptance.BlendCase("kill", mode, "discard-zero-alpha.frag.glsl", (-0.5, 0.5), True)
    image, coverage, counts = acceptance.rational_oracle(case)
    assert counts == {"fragments": 384, "discarded_fragments": 128, "committed_fragments": 256}
    assert np.all(coverage == 1)
    background = (64, 128, 191, 255)
    killed_then_far = acceptance.rational_commit((0, 0, 1, 1), background, mode)
    alpha_zero_near = acceptance.rational_commit((1, 0, 0, 0), background, mode)
    assert np.all(image[:, :8] == killed_then_far)
    assert np.all(image[:, 8:] == alpha_zero_near)


@pytest.mark.parametrize("helper", (run_render.capture_draw, run_render.softpipe_reference,
                                    run_render.native_host_readback))
def test_clear_and_equation_api_preserves_legacy_defaults(helper):
    parameters = inspect.signature(helper).parameters
    assert parameters["blend"].default == "disabled"
    assert parameters["clear_color"].default == (0, 0, 0, 0)
    assert parameters["clear_color"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("helper", (run_render.capture_draw, run_render.softpipe_reference,
                                    run_render.native_host_readback))
@pytest.mark.parametrize("clear_color", (None, True, "0,0,0,0", (0, 0, 0), (0, 0, 0, 0, 0),
                                        (True, 0, 0, 0), (-0.1, 0, 0, 0), (1.1, 0, 0, 0),
                                        (float("inf"), 0, 0, 0), (float("nan"), 0, 0, 0)))
def test_clear_inputs_fail_closed_before_gl(monkeypatch, tmp_path, helper, clear_color):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid clear color must never invoke GL")

    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    options = {"socket_path": tmp_path / "executor.sock"} if helper is run_render.native_host_readback else {}
    with pytest.raises(ValueError, match="clear"):
        helper(ROOT / "tests/fixtures/host-blend", BUILD, tmp_path / "output",
               clear_color=clear_color, **options)


@pytest.fixture(scope="module")
def fresh_equations(tmp_path_factory):
    if not (BUILD / "build.ninja").is_file():
        pytest.skip("configured Mesa required for actual host GL integration")
    directory = tmp_path_factory.mktemp("fresh-blend-equations")
    original_submit = Driver.submit
    original_run = run_render.subprocess.run
    submissions = []
    invocations = []

    def observed_submit(driver, program, **kwargs):
        submissions.append(program.format)
        return original_submit(driver, program, **kwargs)

    def observed_run(command, **kwargs):
        if command and str(command[0]) == str(ROOT / "tools/mgc_harness"):
            environment = kwargs["env"]
            invocations.append({"command": list(map(str, command)),
                                "driver": environment.get("GALLIUM_DRIVER"),
                                "blend": environment.get("MGC_HOST_BLEND"),
                                "clear": environment.get("MGC_HOST_CLEAR_RGBA"),
                                "socket": environment.get("MGC_HOST_EXECUTOR_SOCKET")})
        return original_run(command, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Driver, "submit", observed_submit)
        patch.setattr(run_render.subprocess, "run", observed_run)
        summary = acceptance.validate(directory, BUILD, cases=acceptance.CASES)
    return directory, summary, submissions, invocations


def test_fresh_equations_are_real_legacy_compilation_and_native_readback(fresh_equations):
    directory, summary, submissions, invocations = fresh_equations
    assert summary["passed"] and summary["fpga_execution"] is False
    assert len(summary["cases"]) == len(acceptance.CASES) == 40
    assert submissions and set(submissions) == {ProgramFormat.LEGACY}
    assert len(invocations) == 3 * len(acceptance.CASES)
    assert {invocation["blend"] for invocation in invocations} == set(BLEND_MODES[2:])
    assert all(invocation["clear"] == "0.25,0.5,0.75,1" for invocation in invocations)
    for mode in BLEND_MODES[2:]:
        selected = [invocation for invocation in invocations if invocation["blend"] == mode]
        assert {invocation["driver"] for invocation in selected} == {"mgc", "softpipe"}
        native = [invocation for invocation in selected if "--host-execute" in invocation["command"]]
        assert len(native) == 8 and all(invocation["socket"] for invocation in native)
    assert (directory / "summary.json").is_file()


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_full_rgba_equations_match_rational_native_and_softpipe(fresh_equations, case):
    directory, summary, submissions, invocations = fresh_equations
    report = next(item for item in summary["cases"] if item["case"] == case.name)
    case_directory = directory / case.name
    expected, coverage, counts = acceptance.rational_oracle(case)
    for path in ("host.rgba8", "expected.rgba8", "native/native.rgba8"):
        actual = np.frombuffer((case_directory / path).read_bytes(), dtype=np.uint8).reshape(16, 16, 4)
        assert np.array_equal(actual, expected)
    actual_coverage = np.frombuffer((case_directory / "coverage.u32").read_bytes(), dtype="<u4").reshape(16, 16)
    assert np.array_equal(actual_coverage, coverage)
    reference = np.frombuffer((case_directory / "softpipe.rgba8").read_bytes(), dtype=np.uint8).reshape(16, 16, 4)
    assert np.max(np.abs(reference.astype(np.int16) - expected.astype(np.int16))) <= 1
    assert report["counts"] == counts and report["native_gl_readback"]
    assert report["native_rational_comparison"]["max_channel_error"] == 0
    proof = json.loads((case_directory / "native/native-report.json").read_text())
    assert proof["blend"] == case.blend and proof["repeats"] == case.repeats
    assert len(proof["commits"]) == case.repeats
    assert proof["native_gl_readback"] and proof["fpga_execution"] is False
