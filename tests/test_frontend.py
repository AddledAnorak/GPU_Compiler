"""Frontend protocol, fail-closed capture and real Mesa negative tests."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import validate_frontend as frontend
import validate_week78_nir as week78


def nir(stage, label):
    output = "gl_Position" if stage == "VERTEX" else "FRAG_RESULT_COLOR"
    return (
        f"{label}\nshader: MESA_SHADER_{stage}\n"
        "impl main {\n"
        "    block b0:\n"
        "    32x4 %0 = load_const (1.000000, 0.000000, 0.000000, 1.000000)\n"
        f"             @store_output (%0, %0) (io location={output})\n"
        "    block b1:\n}\n\n"
    )


def trace():
    return (
        nir("VERTEX", "glsl_to_nir") + nir("FRAGMENT", "glsl_to_nir")
        + nir("VERTEX", "NIR before translation to TGSI:")
        + "TGSI after translation from NIR:\nVERT\n  0: MOV OUT[0], IMM[0]\n  1: END\n"
        + nir("FRAGMENT", "NIR before translation to TGSI:")
        + "TGSI after translation from NIR:\nFRAG\n  0: MOV OUT[0], IMM[0]\n  1: END\n"
    )


def test_stage_boundaries_exclude_tgsi_and_earlier_snapshots():
    text = trace()
    snapshots = frontend.parse_snapshots(text)
    assert len(snapshots) == 4
    final = frontend.selected(snapshots, "FRAGMENT", "NIR before translation to TGSI:")
    assert "TGSI" not in final.text
    assert final.line == text[:text.rindex(final.text)].count("\n") + 1
    assert frontend.operations(final.text) == {"load_const": 1}
    result = frontend.inspect_trace("extra_shader", text, 0)
    assert result["evidence"]["VERTEX"]["snapshots"] == 2
    assert result["evidence"]["FRAGMENT"]["final_intrinsics"] == {"store_output": 1}
    assert "no construct-specific contract" in result["observation"]


@pytest.mark.parametrize("returncode", [1, 9, -11, None])
def test_returncode_failure_never_counts_complete_or_stale_trace(returncode):
    with pytest.raises(ValueError, match="harness exited"):
        frontend.inspect_trace("extra_shader", trace(), returncode)


@pytest.mark.parametrize("diagnostic", [
    "shader compile failed:", "link failed:", "LINKING FAILED", "0:4(3): error: invalid type",
    "Assertion `valid' failed", "Segmentation fault", "frontend timeout",
])
def test_diagnostics_reject_even_zero_exit_and_complete_nir(diagnostic):
    with pytest.raises(ValueError, match="diagnostic"):
        frontend.inspect_trace("extra_shader", trace() + diagnostic, 0)


@pytest.mark.parametrize("stage", ["VERTEX", "FRAGMENT"])
@pytest.mark.parametrize("label", ["glsl_to_nir", "NIR before translation to TGSI:"])
def test_missing_initial_or_final_stage_is_failure(stage, label):
    text = trace().replace(nir(stage, label), "")
    with pytest.raises(ValueError, match="missing"):
        frontend.inspect_trace("extra_shader", text, 0)


def test_missing_tgsi_end_cannot_borrow_next_stages_end():
    text = trace().replace("  1: END\n", "", 1)
    with pytest.raises(ValueError, match="incomplete VERT TGSI"):
        frontend.inspect_trace("extra_shader", text, 0)


def test_truncated_nir_is_failure():
    text = trace().replace("    block b1:\n}\n", "    block b1:\n", 1)
    with pytest.raises(ValueError, match="incomplete"):
        frontend.inspect_trace("extra_shader", text, 0)


def test_missing_output_is_failure():
    text = trace().replace("io location=FRAG_RESULT_COLOR", "io location=OTHER")
    with pytest.raises(ValueError, match="fragment stage output"):
        frontend.inspect_trace("extra_shader", text, 0)


def test_opcode_counter_does_not_count_constant_annotations():
    text = "32 %0 = load_const (0x3f800000 = 1.000000 = 1065353216)\n32 %1 = fmul %0, %0"
    assert frontend.operations(text) == {"fmul": 1, "load_const": 1}


def test_texture_contract_rejects_broken_ssa_chain():
    vertex = (
        "shader: MESA_SHADER_VERTEX\n"
        "%0 = @load_input // a_position\n%1 = @load_input // a_texcoord\n"
        "@load_ubo_vec4\n@store_output gl_Position\n@store_output v_texcoord\n"
    )
    initial = "shader: MESA_SHADER_FRAGMENT\ntex sampler_deref_intrinsic\n"
    final = (
        "shader: MESA_SHADER_FRAGMENT\n"
        "32x2 %2 = @load_interpolated_input VARYING_SLOT_VAR0\n"
        "32x4 %3 = (float32)tex %9 (backend1), 0 (texture), 0 (sampler)\n"
        "@store_output (%3, 0) FRAG_RESULT_COLOR\n"
    )
    with pytest.raises(ValueError, match="SSA chain"):
        week78.validate_texture(vertex + initial + final)


def test_fixed_loop_contract_requires_live_multiply():
    initial = {"FRAGMENT": "loop {\n}"}
    final = {"VERTEX": "", "FRAGMENT": "2.083333\n@load_interpolated_input"}
    with pytest.raises(ValueError, match="live varying multiply"):
        frontend.validate_constructs("05_loop", initial, final, {})


def test_runtime_loop_contract_requires_break():
    initial = {"FRAGMENT": ""}
    final = {"VERTEX": "", "FRAGMENT": "loop {\n}"}
    with pytest.raises(ValueError, match="loop break"):
        frontend.validate_constructs("09_runtime_loop", initial, final, {})


def test_week78_compatibility_regenerates_instead_of_reading(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(frontend, "main", lambda argv: calls.append(argv) or 1)
    assert week78.main(tmp_path) == 1
    assert calls == [["--trace-dir", str(tmp_path)]]


def test_environment_overrides_driver_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("GALLIUM_DRIVER", "mgc")
    monkeypatch.setenv("NIR_DEBUG", "")
    monkeypatch.setenv("MESA_SHADER_CACHE_DISABLE", "false")
    env = frontend.environment(tmp_path)
    assert env["GALLIUM_DRIVER"] == "softpipe"
    assert env["NIR_DEBUG"] == "print,tgsi"
    assert env["MESA_SHADER_CACHE_DISABLE"] == "true"
    assert env["LIBGL_DRIVERS_PATH"].startswith(str(tmp_path))
    assert env["PATH"].split(os.pathsep)[0] == str(ROOT / ".venv/bin")


def test_timeout_keeps_partial_output(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial NIR\xff")
    monkeypatch.setattr(subprocess, "run", timeout)
    returncode, text = frontend.capture(["harness"], {}, 1)
    assert returncode is None
    assert "partial NIR" in text and "frontend timeout" in text


def test_launch_failure_is_failure(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing harness")
    monkeypatch.setattr(subprocess, "run", missing)
    returncode, text = frontend.capture(["harness"], {}, 1)
    assert returncode is None
    assert "launch failed" in text


def test_empty_corpus_replaces_stale_success_manifest(tmp_path):
    shaders = tmp_path / "shaders"
    output = tmp_path / "traces"
    shaders.mkdir()
    output.mkdir()
    (output / "manifest.json").write_text('{"successful": 99}')
    assert frontend.main(["--shader-dir", str(shaders), "--trace-dir", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["successful"] == 0
    assert manifest["attempted"] == 0
    assert "no shader pairs" in manifest["error"]


def test_setup_failure_removes_stale_traces(tmp_path):
    shaders = tmp_path / "shaders"
    pair = shaders / "valid"
    pair.mkdir(parents=True)
    (pair / "vert.glsl").write_text("vertex")
    (pair / "frag.glsl").write_text("fragment")
    output = tmp_path / "traces"
    output.mkdir()
    stale = output / "valid.nir.txt"
    stale.write_text(trace())
    assert frontend.main(["--build-dir", str(tmp_path / "missing"), "--shader-dir", str(shaders),
                          "--trace-dir", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["discovered"] == manifest["failed"] == 1
    assert manifest["successful"] == manifest["attempted"] == 0
    assert not stale.exists()


def test_malformed_build_metadata_replaces_stale_success(monkeypatch, tmp_path):
    pair = tmp_path / "shaders/valid"
    pair.mkdir(parents=True)
    (pair / "vert.glsl").write_text("vertex")
    (pair / "frag.glsl").write_text("fragment")
    def bad_metadata(*args):
        raise KeyError("directories")
    monkeypatch.setattr(frontend, "build_harness", bad_metadata)
    output = tmp_path / "traces"
    output.mkdir()
    (output / "manifest.json").write_text('{"successful": 99}')
    assert frontend.main(["--shader-dir", str(pair.parent), "--trace-dir", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["successful"] == manifest["attempted"] == 0
    assert "directories" in manifest["error"]


def test_mixed_corpus_counts_successes_and_continues(monkeypatch, tmp_path):
    shaders = tmp_path / "shaders"
    for name in ("00_failed", "01_missing", "02_good"):
        pair = shaders / name
        pair.mkdir(parents=True)
        (pair / "vert.glsl").write_text("vertex")
        if name != "01_missing":
            (pair / "frag.glsl").write_text("fragment")
    monkeypatch.setattr(frontend, "build_harness", lambda *args: {})
    calls = []
    def fake_capture(command, environment, timeout):
        calls.append(command)
        return (1, "shader compile failed:\n" + trace()) if "00_failed" in command[1] else (0, trace())
    monkeypatch.setattr(frontend, "capture", fake_capture)
    output = tmp_path / "traces"
    assert frontend.main(["--shader-dir", str(shaders), "--trace-dir", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(calls) == manifest["attempted"] == 2
    assert manifest["discovered"] == 3
    assert manifest["successful"] == 1 and manifest["failed"] == 2
    assert [result["ok"] for result in manifest["results"]] == [False, False, True]


def test_library_rebuild_during_capture_invalidates_success(monkeypatch, tmp_path):
    pair = tmp_path / "shaders/valid"
    pair.mkdir(parents=True)
    (pair / "vert.glsl").write_text("vertex")
    (pair / "frag.glsl").write_text("fragment")
    library = tmp_path / "library.so"
    library.write_text("old")
    monkeypatch.setattr(frontend, "build_harness", lambda *args: {
        "libraries": {str(library): frontend.digest(library)},
    })
    def changing_capture(*args):
        library.write_text("rebuilt")
        return 0, trace()
    monkeypatch.setattr(frontend, "capture", changing_capture)
    output = tmp_path / "traces"
    assert frontend.main(["--shader-dir", str(pair.parent), "--trace-dir", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["successful"] == 0 and manifest["failed"] == 1
    assert manifest["results"][0]["ok"] is False
    assert "libraries changed" in manifest["error"]


def test_source_change_during_capture_does_not_count(monkeypatch, tmp_path):
    pair = tmp_path / "shaders/valid"
    pair.mkdir(parents=True)
    (pair / "vert.glsl").write_text("vertex")
    (pair / "frag.glsl").write_text("fragment")
    monkeypatch.setattr(frontend, "build_harness", lambda *args: {})
    def changing_capture(*args):
        (pair / "frag.glsl").write_text("changed during compile")
        return 0, trace()
    monkeypatch.setattr(frontend, "capture", changing_capture)
    output = tmp_path / "traces"
    assert frontend.main(["--shader-dir", str(pair.parent), "--trace-dir", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["successful"] == 0
    assert "source changed" in manifest["results"][0]["error"]


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeouts_are_rejected(timeout):
    with pytest.raises(SystemExit) as error:
        frontend.main(["--timeout", timeout])
    assert error.value.code == 2


def test_real_mesa_compile_link_failures_and_shell_exit(tmp_path):
    if not (ROOT / "mesa/build/src/egl/libEGL.so.1").is_file():
        pytest.skip("local Mesa EGL build required for integration test")
    vertex = "#version 100\nattribute vec4 a_position; varying vec3 color;\nvoid main() { gl_Position = a_position; color = vec3(1.0); }\n"
    fragment = "#version 100\nprecision mediump float; varying vec3 color;\nvoid main() { gl_FragColor = vec4(color, 1.0); }\n"
    shaders = tmp_path / "shaders"
    for name, source in (("00_compile_error", "#version 100\nthis is invalid GLSL\n"),
                         ("01_link_error", fragment.replace("varying vec3 color", "varying vec4 color")
                          .replace("vec4(color, 1.0)", "color")),
                         ("02_good", fragment)):
        pair = shaders / name
        pair.mkdir(parents=True)
        (pair / "vert.glsl").write_text(vertex)
        (pair / "frag.glsl").write_text(source)
    output = tmp_path / "traces"
    result = subprocess.run(["bash", str(ROOT / "tools/dump_nir.sh"), str(ROOT / "mesa/build"),
                             "--shader-dir", str(shaders), "--trace-dir", str(output)],
                            env=dict(os.environ, PATH="/usr/bin:/bin"), text=True,
                            capture_output=True, timeout=60, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["attempted"] == 3 and manifest["successful"] == 1 and manifest["failed"] == 2
    assert "shader compile failed" in (output / "00_compile_error.nir.txt").read_text()
    assert "link failed" in (output / "01_link_error.nir.txt").read_text()
    assert manifest["results"][-1]["ok"] is True
    assert "Successful fresh traces: 1/3" in result.stdout


def test_real_all_shader_pairs(tmp_path):
    if not (ROOT / "mesa/build/src/egl/libEGL.so.1").is_file():
        pytest.skip("local Mesa EGL build required for integration test")
    output = tmp_path / "traces"
    assert frontend.main(["--trace-dir", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    pairs = {path.name for path in (ROOT / "shaders").iterdir()
             if path.is_dir() and not path.name.startswith(".")}
    assert {result["name"] for result in manifest["results"]} == pairs
    assert manifest["discovered"] == manifest["attempted"] == manifest["successful"] == len(pairs)
    assert manifest["failed"] == 0
    assert all("no construct-specific contract" not in result["observation"]
               for result in manifest["results"])
    assert all(result["returncode"] == 0 and result["trace_sha256"] for result in manifest["results"])
