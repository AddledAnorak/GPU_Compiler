import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from tools import run_render
from tools.blend_acceptance import mesh_words
from tools.host_executor import HostExecutorServer
from tools.mgcdriver import ProgramFormat
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT, mesa_environment


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
FIXTURE = ROOT / "shaders/01_flat_color"
BLEND_FIXTURE = ROOT / "tests/fixtures/host-blend"


def harness_call(output, arguments, *, environment=None):
    if environment is None:
        environment = mesa_environment(BUILD, output, ProgramFormat.LEGACY)
    return subprocess.run([str(ROOT / "tools/mgc_harness"), str(FIXTURE / "vert.glsl"),
                           str(FIXTURE / "frag.glsl"), *arguments], env=environment,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("socket_path", [None, ""])
def test_native_harness_requires_explicit_executor_socket(tmp_path, socket_path):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment.pop("MGC_HOST_EXECUTOR_SOCKET", None)
    if socket_path is not None:
        environment["MGC_HOST_EXECUTOR_SOCKET"] = socket_path
    completed = harness_call(tmp_path, ["--host-execute", str(tmp_path / "native.rgba8")], environment=environment)
    assert completed.returncode != 0 and "requires explicit MGC_HOST_EXECUTOR_SOCKET" in completed.stderr
    assert not (tmp_path / "native.rgba8").exists()


@pytest.mark.parametrize("name", ["MGC_HOST_WIDTH", "MGC_HOST_HEIGHT"])
@pytest.mark.parametrize("value", ["", "0", "129", "9999", "-1", "+16", "16.0", "16 ", " 16", "True"])
def test_native_harness_rejects_unbounded_or_nonstrict_dimensions(tmp_path, name, value):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment["MGC_HOST_EXECUTOR_SOCKET"] = str(tmp_path / "unused.sock")
    environment[name] = value
    completed = harness_call(tmp_path, ["--host-execute", str(tmp_path / "native.rgba8")], environment=environment)
    assert completed.returncode != 0 and f"{name} must be a decimal integer in [1, 128]" in completed.stderr
    assert not (tmp_path / "native.rgba8").exists()


@pytest.mark.parametrize("value", ["2", "-1", "true", "", "less", "0 ", "1 "])
def test_native_harness_rejects_nonstrict_or_unsupported_depth_state(tmp_path, value):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment["MGC_HOST_EXECUTOR_SOCKET"] = str(tmp_path / "unused.sock")
    environment["MGC_HOST_DEPTH_TEST"] = value
    completed = harness_call(tmp_path, ["--host-execute", str(tmp_path / "native.rgba8")], environment=environment)
    assert completed.returncode != 0 and "depth state must be explicit 0 or 1" in completed.stderr


def test_native_harness_never_claims_softpipe_as_mgc_readback(tmp_path):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment["GALLIUM_DRIVER"] = "softpipe"
    environment["MGC_HOST_EXECUTOR_SOCKET"] = str(tmp_path / "unused.sock")
    for name in ("MGC_HOST_CAPTURE_DIR", "MGC_HOST_TEXTURE_DATA", "MGC_HOST_DEPTH_TEST", "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT"):
        environment.pop(name, None)
    completed = harness_call(tmp_path, ["--host-execute", str(tmp_path / "native.rgba8")], environment=environment)
    assert completed.returncode != 0 and "requires the mgc CPU host-executor renderer" in completed.stderr
    assert not (tmp_path / "native.rgba8").exists()


@pytest.mark.parametrize("target", [ProgramFormat.HARDWARE.value, "", "unknown"])
def test_native_harness_rejects_nonlegacy_compiler_targets(tmp_path, target):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment["MGC_HOST_EXECUTOR_SOCKET"] = str(tmp_path / "unused.sock")
    environment["MGC_COMPILE_TARGET"] = target
    completed = harness_call(tmp_path, ["--host-execute", str(tmp_path / "native.rgba8")], environment=environment)
    assert completed.returncode != 0 and "legacy compiler target only, not hardware" in completed.stderr
    assert not (tmp_path / "native.rgba8").exists()


@pytest.mark.parametrize("arguments", [["--host-execute"], ["--host-execute", "pixels.rgba8", "extra"]])
def test_native_harness_rejects_incomplete_or_extra_mode_arguments(tmp_path, arguments):
    completed = harness_call(tmp_path, arguments)
    assert completed.returncode != 0 and "usage:" in completed.stderr and "--host-execute" in completed.stderr


@pytest.mark.parametrize("mode", ["compile-only", "draw", "reference"])
def test_existing_harness_modes_clear_inherited_executor_socket_and_keep_sizes(tmp_path, mode):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment["MGC_HOST_EXECUTOR_SOCKET"] = str(tmp_path / "untrusted.sock")
    environment["MGC_HOST_WIDTH"] = "9999"
    environment["MGC_HOST_HEIGHT"] = "bad"
    if mode != "reference":
        environment["MGC_HOST_DRAW_REPEATS"] = "untrusted"
    for name in ("MGC_HOST_TEXTURE_DATA", "MGC_HOST_DEPTH_TEST", "MGC_HOST_BLEND"):
        environment.pop(name, None)
    if mode == "draw":
        captures = tmp_path / "capture"
        captures.mkdir()
        environment["MGC_HOST_CAPTURE_DIR"] = str(captures)
        arguments = ["--draw"]
    elif mode == "reference":
        environment["GALLIUM_DRIVER"] = "softpipe"
        arguments = ["--reference", str(tmp_path / "softpipe.rgba8")]
    else:
        arguments = []
    completed = harness_call(tmp_path, arguments, environment=environment)
    assert completed.returncode == 0, completed.stderr
    assert "native_gl_readback" not in completed.stdout
    if mode == "reference":
        assert (tmp_path / "softpipe.rgba8").stat().st_size == 16 * 16 * 4
    elif mode == "draw":
        captures = list((tmp_path / "capture").glob("*.json"))
        assert len(captures) == 1 and json.loads(captures[0].read_text())["schema"] == "mgc-host-draw-v1"


@pytest.mark.parametrize("mode", ["native", "reference"])
@pytest.mark.parametrize("value", ["", "0", "5", "-1", "+1", "1.0", "1 ", " 1", "01", "true"])
def test_harness_rejects_nonexact_or_unbounded_readback_repeats(tmp_path, mode, value):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.LEGACY)
    environment["MGC_HOST_DRAW_REPEATS"] = value
    environment["MGC_HOST_EXECUTOR_SOCKET"] = str(tmp_path / "unused.sock")
    flag = "--host-execute" if mode == "native" else "--reference"
    completed = harness_call(tmp_path, [flag, str(tmp_path / "pixels.rgba8")], environment=environment)
    assert completed.returncode != 0 and "MGC_HOST_DRAW_REPEATS must be a decimal integer in [1, 4]" in completed.stderr
    assert not (tmp_path / "pixels.rgba8").exists()


@pytest.mark.parametrize("helper", [native_host_readback, softpipe_reference])
@pytest.mark.parametrize("repeats", [None, True, False, 0, 5, -1, 1.0, "1", [], {}])
def test_readback_helpers_reject_bad_repeats_before_gl(monkeypatch, tmp_path, helper, repeats):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid repeats must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    options = {"socket_path": tmp_path / "s"} if helper is native_host_readback else {}
    with pytest.raises(ValueError, match="repeats"):
        helper(FIXTURE, BUILD, tmp_path, repeats=repeats, **options)


@pytest.mark.parametrize("name", ["width", "height"])
@pytest.mark.parametrize("value", [None, True, False, 0, 129, -1, 16.0, "16"])
def test_native_helper_rejects_nonexact_or_unbounded_dimensions(monkeypatch, tmp_path, name, value):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid geometry must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="framebuffer"):
        native_host_readback(FIXTURE, BUILD, tmp_path, socket_path=tmp_path / "s", **{name: value})


@pytest.mark.parametrize("blend", [None, True, False, "", "add", "SOURCE-OVER", b"source-over"])
def test_native_helper_rejects_nonexplicit_blend(monkeypatch, tmp_path, blend):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid blend must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="blend"):
        native_host_readback(FIXTURE, BUILD, tmp_path, socket_path=tmp_path / "s", blend=blend)


@pytest.mark.parametrize("socket_path", ["", "relative.sock", "/" + "s" * 108, "/tmp/invalid\0socket"])
def test_native_helper_requires_bounded_absolute_socket(monkeypatch, tmp_path, socket_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid socket must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="socket"):
        native_host_readback(FIXTURE, BUILD, tmp_path, socket_path=socket_path)


@pytest.mark.parametrize("depth_test", [None, 0, 1, "disabled", "1", "less", [], {}, np.bool_(True)])
def test_native_helper_depth_requires_exact_explicit_boolean(monkeypatch, tmp_path, depth_test):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid depth mode must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="depth mode must be explicitly boolean"):
        native_host_readback(FIXTURE, BUILD, tmp_path, socket_path=tmp_path / "s", depth_test=depth_test)


def native_process_stub(command, environment, *, defect=None):
    width = int(environment["MGC_HOST_WIDTH"])
    height = int(environment["MGC_HOST_HEIGHT"])
    repeats = int(environment["MGC_HOST_DRAW_REPEATS"])
    proof = {"native_gl_readback": True, "fpga_execution": False, "width": width,
             "height": height, "format": "R8G8B8A8_UNORM", "pixel_origin": "bottom-left", "repeats": repeats}
    pixels = np.arange(width * height * 4, dtype=np.uint8).reshape(height, width, 4)
    if defect != "missing-raw":
        Path(command[-1]).write_bytes(pixels.tobytes() if defect != "short-raw" else b"\0")
    captures = Path(environment["MGC_HOST_CAPTURE_DIR"])
    commits = []
    for index in range(repeats):
        sequence = index + 7
        if defect != "missing-capture":
            (captures / f"draw-{sequence}.json").write_text(json.dumps({"schema": "mgc-host-draw-v1", "sequence": sequence if defect != "capture-sequence" else 99}))
        commits.append(f"MGC_HOST_EXECUTOR_COMMIT sequence={sequence} width={width} height={height}")
    if defect == "missing-commit":
        commits = []
    elif defect == "duplicate-commit":
        commits *= 2
    elif defect == "wrong-geometry":
        commits[0] = f"MGC_HOST_EXECUTOR_COMMIT sequence=7 width={width + 1} height={height}"
    elif defect == "invalid-sequence":
        commits[0] = f"MGC_HOST_EXECUTOR_COMMIT sequence=0 width={width} height={height}"
    elif defect == "overflow-sequence":
        commits[0] = f"MGC_HOST_EXECUTOR_COMMIT sequence={1 << 64} width={width} height={height}"
    elif defect == "malformed-commit":
        commits[0] += " unexpected"
    elif defect == "executor-failure":
        commits.append("MGC_HOST_EXECUTOR_FAILURE message=target copy rejected")
    if defect == "false-native-proof":
        proof["native_gl_readback"] = False
    elif defect == "fpga-proof":
        proof["fpga_execution"] = True
    elif defect == "boolean-geometry-proof":
        proof["width"] = True
    elif defect == "wrong-repeats-proof":
        proof["repeats"] = repeats + 1
    elif defect == "zero-raw-no-commit":
        Path(command[-1]).write_bytes(bytes(width * height * 4))
        commits = []
    stdout = "not JSON" if defect == "invalid-proof-json" else json.dumps(proof)
    return SimpleNamespace(returncode=1 if defect == "process-failure" else 0,
                           stdout=stdout, stderr="GL_RENDERER: mgc\n" + "\n".join(commits) + "\n")


@pytest.mark.parametrize("defect", ["missing-raw", "short-raw", "missing-capture", "capture-sequence",
                                  "missing-commit", "duplicate-commit", "wrong-geometry", "invalid-sequence",
                                  "overflow-sequence", "malformed-commit", "executor-failure", "false-native-proof",
                                  "fpga-proof", "boolean-geometry-proof", "wrong-repeats-proof", "invalid-proof-json",
                                  "zero-raw-no-commit", "process-failure"])
def test_native_helper_fails_closed_without_exact_commit_and_gl_evidence(monkeypatch, tmp_path, defect):
    def fake_run(command, *, env, **kwargs):
        return native_process_stub(command, env, defect=defect)
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    with pytest.raises(ValueError):
        native_host_readback(FIXTURE, BUILD, tmp_path / "output", socket_path=tmp_path / "s", width=1, height=2)
    assert (tmp_path / "output/native.log").is_file()
    assert not (tmp_path / "output/native-report.json").exists()


@pytest.mark.parametrize("repeats", [1, 4])
@pytest.mark.parametrize("blend", ["disabled", "source-over"])
def test_native_helper_uses_explicit_inputs_and_retains_raw_gl_evidence(monkeypatch, tmp_path, repeats, blend):
    for name in ("MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_CAPTURE_DIR", "MGC_HOST_DEPTH_TEST",
                 "MGC_HOST_MESH", "MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_DATA", "MGC_HOST_BLEND",
                 "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT", "MGC_HOST_DRAW_REPEATS"):
        monkeypatch.setenv(name, "untrusted")
    mesh = tmp_path / "mesh.bin"
    texture = tmp_path / "texture.bin"
    def fake_run(command, *, env, **kwargs):
        assert command[-2] == "--host-execute" and len(command) == 5
        assert env["GALLIUM_DRIVER"] == "mgc" and env["MGC_COMPILE_TARGET"] == ProgramFormat.LEGACY.value
        assert env["MGC_HOST_EXECUTOR_SOCKET"] == str(tmp_path / "s")
        assert env["MGC_HOST_MESH"] == str(mesh.resolve()) and "MGC_HOST_DEPTH_TEST" not in env
        assert env["MGC_HOST_TEXTURE"] == env["MGC_FRAGMENT_STATUS"] == "1"
        assert env["MGC_HOST_TEXTURE_DATA"] == str(texture.absolute())
        assert env["MGC_HOST_DRAW_REPEATS"] == str(repeats)
        assert env.get("MGC_HOST_BLEND", "disabled") == blend
        return native_process_stub(command, env)
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    image = native_host_readback(FIXTURE, BUILD, tmp_path / "output", socket_path=tmp_path / "s",
                                 width=3, height=2, repeats=repeats, blend=blend, mesh=mesh, texture=texture)
    raw_path = tmp_path / "output/native.rgba8"
    raw = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8).reshape(2, 3, 4)
    np.testing.assert_array_equal(image, raw[::-1])
    report = json.loads((tmp_path / "output/native-report.json").read_text())
    assert report["native_gl_readback"] is True and report["fpga_execution"] is False
    assert report["pixel_origin"] == "top-left" and report["raw_pixel_origin"] == "bottom-left"
    assert [commit["sequence"] for commit in report["commits"]] == list(range(7, 7 + repeats))
    assert len(report["capture_paths"]) == repeats and all(Path(path).is_file() for path in report["capture_paths"])


@pytest.mark.parametrize("depth_test", [False, True])
def test_native_helper_depth_opt_in_is_explicit_and_default_preserves_v1(monkeypatch, tmp_path, depth_test):
    monkeypatch.setenv("MGC_HOST_DEPTH_TEST", "untrusted")
    def fake_run(command, *, env, **kwargs):
        if depth_test:
            assert env["MGC_HOST_DEPTH_TEST"] == "1"
        else:
            assert "MGC_HOST_DEPTH_TEST" not in env
        assert env["MGC_FRAGMENT_STATUS"] == "1"
        return native_process_stub(command, env)
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    native_host_readback(FIXTURE, BUILD, tmp_path, socket_path=tmp_path / "s", depth_test=depth_test)
    report = json.loads((tmp_path / "native-report.json").read_text())
    if depth_test:
        assert report["depth_test"] is True
    else:
        assert "depth_test" not in report


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_native_helper_rejects_nonordinary_raw_output_before_gl(monkeypatch, tmp_path, kind):
    path = tmp_path / "native.rgba8"
    target = tmp_path / "untouched"
    target.write_bytes(b"untouched")
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    def unexpected(*args, **kwargs):
        raise AssertionError("unsafe output must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="ordinary file"):
        native_host_readback(FIXTURE, BUILD, tmp_path, socket_path=tmp_path / "s")
    assert target.read_bytes() == b"untouched"


def test_native_helper_failure_does_not_reuse_stale_success_evidence(monkeypatch, tmp_path):
    (tmp_path / "native.rgba8").write_bytes(bytes(16 * 16 * 4))
    (tmp_path / "native-report.json").write_text('{"native_gl_readback":true}')
    def fake_run(command, *, env, **kwargs):
        return native_process_stub(command, env, defect="zero-raw-no-commit")
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="positive distinct commits"):
        native_host_readback(FIXTURE, BUILD, tmp_path, socket_path=tmp_path / "s")
    assert not (tmp_path / "native-report.json").exists()


def test_native_helper_timeout_preserves_failure_log_without_success_report(monkeypatch, tmp_path):
    def fake_run(command, *, env, **kwargs):
        raise subprocess.TimeoutExpired(command, 60, output=b"partial readback\n", stderr=b"waiting for peer\n")
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="timed out"):
        native_host_readback(FIXTURE, BUILD, tmp_path, socket_path=tmp_path / "s")
    assert (tmp_path / "native.log").read_text() == "partial readback\nwaiting for peer\n"
    assert not (tmp_path / "native-report.json").exists()


@pytest.mark.parametrize("helper", [capture_draw, softpipe_reference])
def test_old_helpers_remove_inherited_executor_and_repeat_state(monkeypatch, tmp_path, helper):
    for name in ("MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_DRAW_REPEATS", "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT"):
        monkeypatch.setenv(name, "untrusted")
    def fake_run(command, *, env, **kwargs):
        assert "MGC_HOST_EXECUTOR_SOCKET" not in env and "MGC_HOST_WIDTH" not in env and "MGC_HOST_HEIGHT" not in env
        if helper is capture_draw:
            assert "MGC_HOST_DRAW_REPEATS" not in env
            (Path(env["MGC_HOST_CAPTURE_DIR"]) / "draw.json").write_text('{"schema":"helper-test"}')
        else:
            assert env["MGC_HOST_DRAW_REPEATS"] == "1"
            Path(command[-1]).write_bytes(bytes(16 * 16 * 4))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    helper(FIXTURE, BUILD, tmp_path)


def quad_mesh(path, *, depth=0):
    positions = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
    path.write_bytes(b"".join(
        struct.pack("<7f", horizontal, vertical, depth, 1, 0, 0, 1)
        for horizontal, vertical in positions))
    return path


@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_softpipe_repeats_preserve_previous_source_over_results(tmp_path, repeats):
    image = softpipe_reference(BLEND_FIXTURE, BUILD, tmp_path / "reference", blend="source-over",
                               mesh=quad_mesh(tmp_path / "quad.bin"), repeats=repeats)
    expected_alpha = {1: 128, 2: 192, 4: 240}[repeats]
    assert np.max(np.abs(image[:, :, 3].astype(np.int16) - expected_alpha)) <= 1
    assert np.all(image[:, :, 1] == 0)


@pytest.mark.parametrize("width,height,blend,repeats", [(16, 16, "disabled", 1), (16, 16, "source-over", 2),
                                                       (16, 16, "source-over", 4), (17, 9, "disabled", 1),
                                                       (1, 1, "disabled", 1), (128, 1, "disabled", 1),
                                                       (1, 128, "disabled", 1)])
def test_actual_native_gl_readback_matches_accepted_server_reply(tmp_path, width, height, blend, repeats):
    mesh = quad_mesh(tmp_path / "quad.bin")
    with tempfile.TemporaryDirectory(prefix="btp-gl-peer-") as temporary:
        socket_path = Path(temporary) / "s"
        with HostExecutorServer(socket_path, max_requests=repeats) as server:
            image = native_host_readback(BLEND_FIXTURE, BUILD, tmp_path / "native", socket_path=socket_path,
                                         mesh=mesh, blend=blend, repeats=repeats, width=width, height=height)
    assert not server.errors and len(server.records) == repeats
    assert all(record["depth"] is None for record in server.records)
    report = json.loads((tmp_path / "native/native-report.json").read_text())
    assert [commit["sequence"] for commit in report["commits"]] == [record["sequence"] for record in server.records]
    np.testing.assert_array_equal(image, server.records[-1]["rgba"])
    assert all(record["report"]["committed_fragments"] == width * height for record in server.records)
    if (width, height) == (16, 16):
        reference = softpipe_reference(BLEND_FIXTURE, BUILD, tmp_path / "reference", mesh=mesh,
                                       blend=blend, repeats=repeats)
        assert np.max(np.abs(image.astype(np.int16) - reference.astype(np.int16))) <= 1


@pytest.mark.parametrize("repeats", [1, 4])
def test_actual_zero_rgba_native_readback_still_requires_positive_commits(tmp_path, repeats):
    shader_directory = tmp_path / "shader"
    shader_directory.mkdir()
    (shader_directory / "vert.glsl").write_bytes((BLEND_FIXTURE / "vert.glsl").read_bytes())
    (shader_directory / "frag.glsl").write_bytes((BLEND_FIXTURE / "zero-alpha.frag.glsl").read_bytes())
    mesh = quad_mesh(tmp_path / "quad.bin", depth=-.5)
    with tempfile.TemporaryDirectory(prefix="btp-gl-peer-") as temporary:
        socket_path = Path(temporary) / "s"
        with HostExecutorServer(socket_path, max_requests=repeats) as server:
            image = native_host_readback(shader_directory, BUILD, tmp_path / "native", socket_path=socket_path,
                                         mesh=mesh, blend="source-over", repeats=repeats)
    assert not server.errors and len(server.records) == repeats
    assert all(record["report"]["committed_fragments"] == 256 for record in server.records)
    np.testing.assert_array_equal(image, np.zeros((16, 16, 4), dtype=np.uint8))
    np.testing.assert_array_equal(image, server.records[-1]["rgba"])
    log = (tmp_path / "native/native.log").read_text()
    assert log.count("MGC_HOST_EXECUTOR_COMMIT sequence=") == repeats


def test_actual_native_budget_failure_cannot_claim_an_unchanged_zero_rt_as_success(tmp_path):
    with tempfile.TemporaryDirectory(prefix="btp-gl-peer-") as temporary:
        socket_path = Path(temporary) / "s"
        with HostExecutorServer(socket_path, max_requests=1, max_steps=1) as server:
            with pytest.raises(ValueError, match="host execution failed"):
                native_host_readback(FIXTURE, BUILD, tmp_path / "native", socket_path=socket_path)
    assert not server.records and server.errors
    log = (tmp_path / "native/native.log").read_text()
    assert "MGC_HOST_EXECUTOR_FAILURE" in log and "MGC_HOST_EXECUTOR_COMMIT" not in log
    assert not (tmp_path / "native/native-report.json").exists()


@pytest.mark.parametrize("blend,repeats", [("disabled", 1), ("source-over", 2)])
def test_actual_textured_native_readback_preserves_bytes_orientation_and_feedback(tmp_path, blend, repeats):
    texture = tmp_path / "texture.bin"
    texture.write_bytes(struct.pack("<II", 2, 2) + bytes([255, 0, 0, 128, 0, 255, 0, 0,
                                                        0, 0, 255, 255, 255, 255, 255, 64]))
    mesh = quad_mesh(tmp_path / "quad.bin")
    shader_directory = ROOT / "tests/fixtures/host-texture"
    with tempfile.TemporaryDirectory(prefix="btp-gl-peer-") as temporary:
        socket_path = Path(temporary) / "s"
        with HostExecutorServer(socket_path, max_requests=repeats) as server:
            image = native_host_readback(shader_directory, BUILD, tmp_path / "native", socket_path=socket_path,
                                         mesh=mesh, texture=texture, blend=blend, repeats=repeats)
    assert not server.errors and len(server.records) == repeats
    assert all(record["artifact"]["schema"] == "mgc-host-draw-v2" for record in server.records)
    assert all(record["report"]["committed_fragments"] == 256 for record in server.records)
    np.testing.assert_array_equal(image, server.records[-1]["rgba"])
    reference = softpipe_reference(shader_directory, BUILD, tmp_path / "reference", mesh=mesh,
                                   texture=texture, blend=blend, repeats=repeats)
    assert np.max(np.abs(image.astype(np.int16) - reference.astype(np.int16))) <= 1


@pytest.mark.parametrize("variant,depths,blend,repeats", [("frag.glsl", (-.5, .5), "disabled", 1),
                                                       ("frag.glsl", (.5, -.5), "disabled", 2),
                                                       ("frag.glsl", (-.5, .5), "source-over", 2),
                                                       ("zero-alpha.frag.glsl", (-.5, .5), "source-over", 2),
                                                       ("zero-alpha.frag.glsl", (.5, -.5), "source-over", 2),
                                                       ("discard.frag.glsl", (-.5, .5), "disabled", 2),
                                                       ("discard-zero-alpha.frag.glsl", (.5, -.5), "source-over", 2)])
def test_actual_native_depth_overlap_and_repeated_rt_feedback(tmp_path, variant, depths, blend, repeats):
    shader_directory = tmp_path / "shader"
    shader_directory.mkdir()
    (shader_directory / "vert.glsl").write_bytes((BLEND_FIXTURE / "vert.glsl").read_bytes())
    (shader_directory / "frag.glsl").write_bytes((BLEND_FIXTURE / variant).read_bytes())
    mesh = tmp_path / "overlap.bin"
    mesh.write_bytes(mesh_words(depths))
    with tempfile.TemporaryDirectory(prefix="btp-depth-peer-") as temporary:
        socket_path = Path(temporary) / "s"
        with HostExecutorServer(socket_path, max_requests=repeats) as server:
            image = native_host_readback(shader_directory, BUILD, tmp_path / "native", socket_path=socket_path,
                                         mesh=mesh, blend=blend, repeats=repeats, depth_test=True)
    assert not server.errors and len(server.records) == repeats
    report = json.loads((tmp_path / "native/native-report.json").read_text())
    assert report["depth_test"] is True
    assert [commit["sequence"] for commit in report["commits"]] == [record["sequence"] for record in server.records]
    np.testing.assert_array_equal(image, server.records[-1]["rgba"])
    for record in server.records:
        assert record["depth"].dtype == np.uint16 and record["depth"].shape == (16, 16)
        assert record["report"]["depth_storage"] == "Z16_UNORM"
    if "discard" in variant:
        np.testing.assert_array_equal(server.records[-1]["depth"][:, :8], np.full((16, 8), 49151, dtype=np.uint16))
        np.testing.assert_array_equal(server.records[-1]["depth"][:, 8:], np.full((16, 8), 16384, dtype=np.uint16))
        assert server.records[0]["report"]["discarded_fragments"] == 128
    else:
        np.testing.assert_array_equal(server.records[-1]["depth"], np.full((16, 16), 16384, dtype=np.uint16))
    if repeats > 1:
        assert server.records[-1]["report"]["committed_fragments"] == 0
        np.testing.assert_array_equal(image, server.records[0]["rgba"])
    if variant == "zero-alpha.frag.glsl" and depths[0] == -.5:
        assert not np.any(image) and server.records[0]["report"]["committed_fragments"] == 256
    reference = softpipe_reference(shader_directory, BUILD, tmp_path / "reference", mesh=mesh,
                                   blend=blend, repeats=repeats, depth_test=True)
    assert np.max(np.abs(image.astype(np.int16) - reference.astype(np.int16))) <= 1


@pytest.mark.parametrize("width,height", [(17, 9), (1, 128)])
def test_actual_native_depth_attachment_matches_bounded_framebuffer_geometry(tmp_path, width, height):
    mesh = tmp_path / "overlap.bin"
    mesh.write_bytes(mesh_words((-.5, .5)))
    with tempfile.TemporaryDirectory(prefix="btp-depth-peer-") as temporary:
        socket_path = Path(temporary) / "s"
        with HostExecutorServer(socket_path, max_requests=2) as server:
            image = native_host_readback(BLEND_FIXTURE, BUILD, tmp_path / "native", socket_path=socket_path,
                                         mesh=mesh, repeats=2, depth_test=True, width=width, height=height)
    assert not server.errors and len(server.records) == 2
    np.testing.assert_array_equal(image, server.records[-1]["rgba"])
    assert np.all(image == np.array([255, 0, 0, 128], dtype=np.uint8))
    for record in server.records:
        np.testing.assert_array_equal(record["depth"], np.full((height, width), 16384, dtype=np.uint16))
    assert server.records[0]["report"]["committed_fragments"] == width * height
    assert server.records[1]["report"]["committed_fragments"] == 0


def test_actual_native_depth_sloped_rows_keep_rt_feedback_orientation(tmp_path):
    positions = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
    mesh = tmp_path / "sloped-overlap.bin"
    mesh.write_bytes(b"".join(struct.pack("<7f", horizontal, vertical, vertical * .5, 1, 0, 0, 1)
                              for horizontal, vertical in positions) + mesh_words((0,)))
    with tempfile.TemporaryDirectory(prefix="btp-depth-peer-") as temporary:
        socket_path = Path(temporary) / "s"
        with HostExecutorServer(socket_path, max_requests=2) as server:
            image = native_host_readback(BLEND_FIXTURE, BUILD, tmp_path / "native", socket_path=socket_path,
                                         mesh=mesh, repeats=2, depth_test=True)
    assert not server.errors and len(server.records) == 2
    np.testing.assert_array_equal(image, server.records[-1]["rgba"])
    assert server.records[1]["report"]["committed_fragments"] == 0
    vertical_centers = 1 - 2 * (np.arange(16) + .5) / 16
    expected_rows = np.minimum(np.rint((.5 + .25 * vertical_centers) * 65535), 32768).astype(np.uint16)
    expected_depth = np.repeat(expected_rows[:, None], 16, axis=1)
    for record in server.records:
        np.testing.assert_array_equal(record["depth"], expected_depth)
    reference = softpipe_reference(BLEND_FIXTURE, BUILD, tmp_path / "reference", mesh=mesh,
                                   repeats=2, depth_test=True)
    assert np.max(np.abs(image.astype(np.int16) - reference.astype(np.int16))) <= 1
