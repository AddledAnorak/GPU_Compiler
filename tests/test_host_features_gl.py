"""Bounded fresh GL ABI4 acceptance; missing integration fails, never skips."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tools import host_features_acceptance as acceptance
from tools.run_shader import ROOT
from tools.mgcdriver import DriverError


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
CASES = acceptance.ALL_CASES


@pytest.fixture(scope="module")
def private_host_features_harness(tmp_path_factory):
    return acceptance.build_private_harness(BUILD, tmp_path_factory.mktemp("abi4-private"))


def actual_case(name, binary, output):
    report = acceptance.run_case(CASES[name], BUILD, binary, output)
    assert report["passed"] is True
    assert report["native_gl_readback"] is True
    assert report["fpga_execution"] is False and report["not_fpga"] is True
    assert report["reference_only"] is False
    assert all(check["passed"] is True for check in report["checks"].values())
    assert json.loads((output / "acceptance.json").read_text())["passed"] is True


def test_real_gl_runtime_explicit_lod_nearest(private_host_features_harness, tmp_path):
    actual_case("runtime-lod-nearest", private_host_features_harness, tmp_path)


def test_real_gl_deep_report_path_uses_private_short_socket(private_host_features_harness, tmp_path):
    output = tmp_path / ("deep-report-" * 12)
    assert len(os.fsencode(output / "executor.sock")) > 107
    actual_case("runtime-lod-nearest", private_host_features_harness, output)
    report = json.loads((output / "acceptance.json").read_text())
    socket_path = Path(report["native_socket_path"])
    assert len(os.fsencode(socket_path)) <= 107
    assert socket_path.parent.parent == Path("/tmp")
    assert not socket_path.parent.exists()


def test_real_gl_native_failure_removes_private_socket(private_host_features_harness, tmp_path):
    output = tmp_path / ("deep-failed-report-" * 8)
    original_invoke = acceptance._invoke
    def fail_native(*arguments, **options):
        if arguments[7] == "native":
            socket_path = Path(options["socket_path"])
            assert socket_path.is_socket()
            assert socket_path.parent.stat().st_mode & 0o777 == 0o700
            raise acceptance.AcceptanceError("intentional native invocation failure")
        return original_invoke(*arguments, **options)
    with patch.object(acceptance, "_invoke", fail_native):
        with pytest.raises(acceptance.AcceptanceError, match="intentional native invocation failure"):
            acceptance.run_case(CASES["runtime-lod-nearest"], BUILD, private_host_features_harness, output)
    report = json.loads((output / "acceptance.json").read_text())
    assert report["passed"] is report["native_gl_readback"] is False
    assert report["failed_stage"] == "native"
    assert report["native_request_headers"] == report["native_reply_headers"] == []
    assert not Path(report["native_socket_path"]).parent.exists()


def test_real_gl_fractional_explicit_lod_linear(private_host_features_harness, tmp_path):
    actual_case("fractional-lod-linear", private_host_features_harness, tmp_path)


def test_real_gl_fractional_explicit_lod_trilinear(private_host_features_harness, tmp_path):
    actual_case("fractional-lod-trilinear", private_host_features_harness, tmp_path)


def test_real_gl_multiple_bindings_samples_sparse_32_nonrgba(private_host_features_harness, tmp_path):
    actual_case("multiple-bindings-samples-sparse-32-r8", private_host_features_harness, tmp_path)


def test_real_gl_fragcoord_frontfacing_opposite_winding(private_host_features_harness, tmp_path):
    actual_case("fragcoord-opposite-facing", private_host_features_harness, tmp_path)


def test_real_gl_dfdx_dfdy_actual_four_lane_quad(private_host_features_harness, tmp_path):
    actual_case("dfdx-dfdy-quad", private_host_features_harness, tmp_path)


def test_real_gl_implicit_lod_actual_four_lane_quad(private_host_features_harness, tmp_path):
    actual_case("implicit-lod-quad", private_host_features_harness, tmp_path)


def test_real_gl_local_instanceid_advertised_glsl120(private_host_features_harness, tmp_path):
    actual_case("local-instance-id", private_host_features_harness, tmp_path)


@pytest.mark.parametrize("name", [case.name for case in acceptance.OPTIONAL_CASES])
def test_real_gl_optional_builtin_frontend_capability_gate(private_host_features_harness, tmp_path, name):
    report = acceptance.run_case(CASES[name], BUILD, private_host_features_harness, tmp_path)
    if report["status"] == "frontend-blocked":
        assert report["passed"] is report["native_gl_readback"] is False
        assert report["frontend"]["actual_glsl_version"] < report["frontend"]["required_glsl_version"] == 130
        assert report["frontend"]["shader_rejected"] is True
        assert report["frontend"]["vertexid_gl_execution"] is False
        assert report["frontend"]["builtin_gl_execution"] is False
        assert report["frontend"]["compiler_exit_code"] > 0
        assert report["frontend"]["aborted"] is False
        assert report["frontend"]["advertised_extensions"]["GL_ARB_draw_instanced"] is True
        assert report["checks"]["frontend_fail_closed"]["passed"] is True
        assert report["submissions"] == report["native_request_headers"] == []
        assert json.loads((tmp_path / "acceptance.json").read_text())["passed"] is False
    else:
        assert report["passed"] is report["native_gl_readback"] is True
        assert all(check["passed"] is True for check in report["checks"].values())


def test_real_gl_separate_rgb_alpha_blend_factors(private_host_features_harness, tmp_path):
    actual_case("separate-blend-factors", private_host_features_harness, tmp_path)


def test_real_gl_z24s8_front_back_stencil_depth_failure(private_host_features_harness, tmp_path):
    actual_case("stencil-z24s8", private_host_features_harness, tmp_path)


def test_real_gl_rgb565_color_mask(private_host_features_harness, tmp_path):
    actual_case("rgb565-color-mask", private_host_features_harness, tmp_path)


def test_real_gl_r8_typed_framebuffer(private_host_features_harness, tmp_path):
    actual_case("r8-framebuffer", private_host_features_harness, tmp_path)


def test_real_gl_bgra8_typed_framebuffer(private_host_features_harness, tmp_path):
    actual_case("bgra8-framebuffer", private_host_features_harness, tmp_path)


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_softpipe_fixture_oracle_preflight(private_host_features_harness, tmp_path, case):
    report = acceptance.run_case(case, BUILD, private_host_features_harness, tmp_path, reference_only=True)
    assert report["passed"] is False
    assert report["status"] == "reference-preflight-only"
    assert report["checks"]["softpipe_vs_independent_oracle"]["passed"] is True
    assert report["submissions"] == report["native_request_headers"] == []
    assert all(count == 0 for phase in report["quad_derivative_barriers"].values() for count in phase.values())


@pytest.mark.parametrize("case", acceptance.ANISOTROPY_CASES, ids=lambda case: case.name)
def acceptance_v14_real_gl_line_tap(private_host_features_harness, tmp_path, case):
    actual_case(case.name, private_host_features_harness, tmp_path)
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert report["covered_pixels"] == 4
    assert report["kernel"]["max_anisotropy"] == 4 and report["kernel"]["not_softpipe_ewa"] is True
    assert report["kernel"]["taps"] == (3 if "three-tap" in case.name else 4)
    assert report["native_reply_headers"][0]["status"] == 0
    assert report["comparisons"]["softpipe_ewa_vs_line_tap_oracle"]["required_for_acceptance"] is False


@pytest.mark.parametrize("case", acceptance.ANISOTROPY_CASES, ids=lambda case: case.name)
def acceptance_v14_softpipe_ewa_observation(private_host_features_harness, tmp_path, case):
    report = acceptance.run_case(case, BUILD, private_host_features_harness, tmp_path, reference_only=True)
    assert report["passed"] is report["native_gl_readback"] is False
    assert report["status"] == "reference-preflight-only"
    assert report["checks"]["actual_softpipe_ewa_readback"]["passed"] is True
    comparison = report["comparisons"]["softpipe_ewa_vs_line_tap_oracle"]
    assert comparison["required_for_acceptance"] is comparison["kernel_equivalence_claimed"] is False
    if case.name == "anisotropy-v14-constant-ewa-control":
        assert report["checks"]["constant_fixture_softpipe_vs_oracle"]["passed"] is True


def acceptance_v14_native_instruction_budget_fails_closed(private_host_features_harness, tmp_path):
    case = CASES["anisotropy-v14-constant-ewa-control"]
    positive = tmp_path / "positive"
    actual_case(case.name, private_host_features_harness, positive)
    artifact = json.loads(next((positive / "capture/captures").glob("*.json")).read_text())
    with pytest.raises((ValueError, TimeoutError, DriverError), match="instruction|step|budget"):
        acceptance.render_artifact(artifact, width=8, height=8, max_steps=1, max_fragments=1024,
                                   execution_timeout=5)
    replies = []
    original_handle = acceptance.host_executor.HostExecutorServer._handle
    def observed_handle(executor, connection, pid):
        return original_handle(executor, acceptance.ReplyObserverSocket(connection, replies), pid)
    with patch.object(acceptance.host_executor.HostExecutorServer, "_handle", observed_handle):
        with acceptance.host_executor.HostExecutorServer(tmp_path / "budget.sock", max_requests=1,
                                                         max_steps=1, max_fragments=1024) as server:
            try:
                _, readback, _ = acceptance._invoke(private_host_features_harness, BUILD, tmp_path / "negative",
                    positive / "shader", case, positive / "mesh.fp32", positive / "textures.bin", "native", server.path)
            except acceptance.AcceptanceError:
                readback = None
    assert server.records == []
    assert server.errors and any("step" in error.lower() or "budget" in error.lower() for error in server.errors)
    assert len(replies) == 1 and replies[0]["magic"] == "MGCRG04" and replies[0]["bytes"] == 44
    assert replies[0]["status"] == 1 and replies[0]["color_bytes"] == replies[0]["depth_bytes"] == 0
    assert replies[0]["payload_bytes"] == 0
    negative_artifact = json.loads(next((tmp_path / "negative/captures").glob("*.json")).read_text())
    for name in ("schema", "shaders", "vertex_buffers", "vertex_elements", "draw", "textures", "render_state"):
        assert negative_artifact.get(name) == artifact.get(name)
    for stage in ("vs", "fs"):
        assert negative_artifact["shaders"][stage]["assembly"] == (
            tmp_path / "negative/compiler" / f"{stage}.asm").read_text()
        assert negative_artifact["shaders"][stage]["layout"] == json.loads((
            tmp_path / "negative/compiler" / f"{stage}.layout.json").read_text())
    if readback is not None:
        assert np.count_nonzero(readback) == 0
    (tmp_path / "budget-negative.json").write_text(json.dumps({
        "passed": False, "budget_rejected": True, "scope": "actual v14 native instruction-budget fail-closed probe",
        "max_steps": 1, "server_timeout_seconds": 5, "native_reply_headers": replies,
        "native_errors": server.errors, "fpga_execution": False, "not_fpga": True}, indent=2) + "\n")
