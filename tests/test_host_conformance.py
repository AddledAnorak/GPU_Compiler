"""Bounded subprocess and manifest tests; mocked evidence is not real acceptance."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from tools import host_conformance as conformance


def test_registry_is_existing_cpu_host_commands_with_selectable_subsets():
    stages = conformance.select_stages(suite="all")
    assert len(stages) == 17
    assert {stage.name for stage in conformance.select_stages(("blend", "native-host"))} == {"blend", "native-host"}
    for stage in stages:
        assert stage.module.startswith("tools.")
        assert (conformance.ROOT / (stage.module.replace(".", "/") + ".py")).is_file()
        assert "board" not in stage.module
        command = stage.command(sys.executable, Path("/tmp/build"), Path("/tmp/evidence"))
        assert command[1:3] == ["-m", stage.module]
        assert "--build" in command and "--output" in command


@pytest.mark.parametrize("names", ((), ("unknown",), ("blend", "blend"), ("blend;echo unsafe",)))
def test_unknown_empty_duplicate_or_shell_stage_names_fail(names):
    with pytest.raises(ValueError):
        conformance.select_stages(names)


def test_extended_expands_only_existing_supported_case_sets():
    for stage in conformance.STAGES:
        command = stage.command(sys.executable, Path("/tmp/build"), Path("/tmp/evidence"), extended=True)
        assert ("--extended" in command) == (stage.module in conformance.EXTENDED_MODULES)


def host_feature_report():
    source = conformance.ROOT / "tools/host_features_acceptance.py"
    module = ast.parse(source.read_text())
    cases = next(node.value for node in module.body if isinstance(node, ast.Assign) and
                 any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets))
    names = [case.args[0].value for case in cases.elts]
    return {"passed": True, "fpga_execution": False, "reference_only": False, "complete_default_suite": True,
            "requested_cases": names,
            "cases": [{"case": name, "passed": True, "status": "passed", "native_gl_readback": True,
                       "shader_abi_version": 4, "capture_version": 4, "unit_test_only": True} for name in names]}


def test_host_features_stage_selects_defaults_without_optional_or_preflight_flags():
    stage, = conformance.select_stages(("gallium-host-features",))
    assert stage.module == "tools.host_features_acceptance" and stage.evidence == "report.json"
    assert stage in conformance.select_stages(suite="all")
    assert stage not in conformance.select_stages(suite="smoke")
    assert stage not in conformance.select_stages(suite="core")
    command = stage.command(sys.executable, Path("/tmp/build"), Path("/tmp/evidence"), extended=True)
    assert command == [sys.executable, "-m", "tools.host_features_acceptance",
                       "--build", "/tmp/build", "--output", "/tmp/evidence"]
    assert stage.sources == ("tests/fixtures/host-features-gl/harness.c",)
    report = host_feature_report()
    assert len(report["requested_cases"]) == conformance.HOST_FEATURE_CASE_COUNT == 13
    assert not set(report["requested_cases"]) & set(conformance.OPTIONAL_HOST_FEATURE_CASES)


def test_host_features_source_snapshot_hashes_module_and_private_harness_without_execution(tmp_path, monkeypatch):
    stage = conformance.STAGE_BY_NAME["gallium-host-features"]
    module = tmp_path / "tools/host_features_acceptance.py"
    module.parent.mkdir()
    module.write_text('UNIT_TEST_ONLY = True\n')
    fixture = tmp_path / stage.sources[0]
    fixture.parent.mkdir(parents=True)
    fixture.write_text("int main(void) { return 0; }\n")
    commands = []

    def metadata_command(command, **options):
        commands.append(command)
        return {"command": command, "status": "passed", "returncode": 0}, b"unit-test git metadata", b""

    monkeypatch.setattr(conformance, "ROOT", tmp_path)
    monkeypatch.setattr(conformance, "run_command", metadata_command)
    snapshot = conformance.source_snapshot(tmp_path / "no-build", (stage,))
    sources = {entry["path"]: entry for entry in snapshot["tools"]}
    for path in (module, fixture):
        assert sources[str(path)]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(command[0] == "git" for command in commands)
    assert snapshot["binaries"] == []


def test_host_features_evidence_accepts_complete_mandatory_execution_report(tmp_path):
    stage = conformance.STAGE_BY_NAME["gallium-host-features"]
    path = tmp_path / stage.evidence
    report = host_feature_report()
    path.write_text(json.dumps(report))
    evidence = conformance._evidence(path, stage)
    assert evidence["report"] == report
    assert evidence["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ("reference-only", "incomplete", "missing-case", "missing-native",
                                      "wrong-abi", "wrong-capture", "frontend-blocked", "optional-vertexid"))
def test_host_features_preflight_partial_or_frontend_gates_never_become_acceptance(tmp_path, mutation):
    report = copy.deepcopy(host_feature_report())
    if mutation == "reference-only":
        report["reference_only"] = True
    elif mutation == "incomplete":
        report["complete_default_suite"] = False
    elif mutation == "missing-case":
        report["cases"].pop()
    elif mutation == "missing-native":
        report["cases"][0]["native_gl_readback"] = False
    elif mutation == "wrong-abi":
        report["cases"][0]["shader_abi_version"] = 1
    elif mutation == "wrong-capture":
        report["cases"][0]["capture_version"] = 1
    elif mutation == "frontend-blocked":
        report["cases"][0].update(status="frontend-blocked", native_gl_readback=False,
                                   frontend={"actual_glsl_version": 120, "required_glsl_version": 130,
                                             "shader_rejected": True, "vertexid_gl_execution": False},
                                   checks={"frontend_fail_closed": {"passed": True}})
    else:
        report["requested_cases"][0] = "vertex-local-instance-id"
        report["cases"][0]["case"] = "vertex-local-instance-id"
    stage = conformance.STAGE_BY_NAME["gallium-host-features"]
    path = tmp_path / stage.evidence
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="host feature"):
        conformance._evidence(path, stage)


def test_real_subprocess_result_keeps_exact_argv_streams_and_exit(tmp_path):
    command = [sys.executable, "-c", "import sys; print('progress'); print('diagnostic', file=sys.stderr); sys.exit(7)"]
    result, stdout, stderr = conformance.run_command(command, cwd=tmp_path, timeout=2)
    assert result["command"] == command and result["cwd"] == str(tmp_path)
    assert result["status"] == "failed" and result["returncode"] == 7
    assert stdout == b"progress\n" and stderr == b"diagnostic\n"
    assert result["started_at"] and result["finished_at"] and result["elapsed_seconds"] < 2


def test_real_subprocess_timeout_preserves_progress_and_kills_descendants(tmp_path):
    escaped = tmp_path / "escaped"
    child = f"import time; time.sleep(0.7); open({str(escaped)!r}, 'w').write('escaped')"
    parent = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); print('started', flush=True); time.sleep(10)"
    started = time.monotonic()
    result, stdout, _ = conformance.run_command([sys.executable, "-c", parent], cwd=tmp_path, timeout=0.2)
    assert result["status"] == "timed_out" and result["returncode"] != 0
    assert stdout == b"started\n" and time.monotonic() - started < 3
    time.sleep(0.75)
    assert not escaped.exists()


def test_closed_pipes_do_not_remove_process_deadline(tmp_path):
    command = [sys.executable, "-c", "import os, time; os.close(1); os.close(2); time.sleep(10)"]
    result, stdout, stderr = conformance.run_command(command, cwd=tmp_path, timeout=0.1)
    assert result["status"] == "timed_out" and stdout == stderr == b""


def test_subprocess_output_is_bounded_and_failure_propagates(tmp_path):
    command = [sys.executable, "-c", "import os, time; os.write(1, b'x' * 4096); time.sleep(10)"]
    result, stdout, stderr = conformance.run_command(command, cwd=tmp_path, timeout=2, max_output=64)
    assert result["status"] == "output_limit"
    assert len(stdout) + len(stderr) == 64
    assert result["returncode"] != 0


def test_missing_executable_is_structured_failure(tmp_path):
    result, stdout, stderr = conformance.run_command([str(tmp_path / "absent")], cwd=tmp_path, timeout=1)
    assert result["status"] == "error" and result["returncode"] is None
    assert result["error"] and stdout == stderr == b""


@pytest.mark.parametrize("command", ("echo unsafe", [], [""], ["unsafe\0argv"], ["x"] * 65))
def test_subprocess_rejects_unbounded_or_shell_argv(command, tmp_path):
    with pytest.raises(ValueError, match="bounded argv"):
        conformance.run_command(command, cwd=tmp_path, timeout=1)


@pytest.fixture
def mocked_acceptance(monkeypatch):
    calls = []
    failures = {}

    def snapshot(*args, **options):
        return {"unit_test_only": True}

    def run(command, **options):
        calls.append((command, options))
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir()
        status, evidence = failures.get(command[2], ("passed", {"passed": True, "fpga_execution": False,
                                                               "cases": [{"passed": True, "unit_test_only": True}]}))
        if evidence is not None:
            (directory / "summary.json").write_text(json.dumps(evidence))
        return {"command": command, "status": status, "returncode": 0 if status == "passed" else 2,
                "error": None if status == "passed" else "unit-test failure", "timeout_seconds": options["timeout"]}, b"unit-test stdout", b""

    monkeypatch.setattr(conformance, "source_snapshot", snapshot)
    monkeypatch.setattr(conformance, "run_command", run)
    return calls, failures


def test_manifest_retains_command_result_source_and_scope(tmp_path, mocked_acceptance):
    calls, _ = mocked_acceptance
    output = tmp_path / "new-run"
    manifest = conformance.run_conformance(output, names=("blend", "fragment"), timeout=5, total_timeout=20)
    assert manifest["passed"] is True and manifest["status"] == "passed"
    assert manifest["scope"] == "bounded host, not universal Khronos/FPGA certification"
    assert manifest["fpga_execution"] is False and manifest["khronos_certification"] is False
    assert manifest["source_provenance"]["unit_test_only"] is True
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert len(calls) == 2
    for stage in manifest["stages"]:
        assert stage["command"][2].startswith("tools.")
        assert stage["evidence"]["report"]["passed"] is True
        assert stage["stdout"]["sha256"] == hashlib.sha256(b"unit-test stdout").hexdigest()
        evidence = Path(stage["evidence"]["path"]).read_bytes()
        assert stage["evidence"]["sha256"] == hashlib.sha256(evidence).hexdigest()
    assert all(options["environment"]["LIBGL_ALWAYS_SOFTWARE"] == "1" for _, options in calls)


@pytest.mark.parametrize("evidence", (None, {"passed": False, "fpga_execution": False},
                                     {"passed": True}, {"passed": True, "fpga_execution": True},
                                     {"passed": True, "fpga_execution": False, "physical_gpu_execution": True},
                                     {"passed": True, "fpga_execution": False, "status": "running"},
                                     {"passed": True, "fpga_execution": False, "cases": [{"passed": False}]}))
def test_zero_exit_without_successful_fresh_evidence_fails(tmp_path, mocked_acceptance, evidence):
    _, failures = mocked_acceptance
    failures["tools.blend_acceptance"] = ("passed", evidence)
    manifest = conformance.run_conformance(tmp_path / "run", names=("blend",))
    assert manifest["passed"] is False
    assert manifest["stages"][0]["status"] == "failed" and manifest["stages"][0]["returncode"] == 0
    assert "fresh evidence" in manifest["stages"][0]["error"]


def test_nonzero_exit_cannot_be_overridden_by_a_passing_summary(tmp_path, mocked_acceptance):
    _, failures = mocked_acceptance
    failures["tools.blend_acceptance"] = ("failed", {"passed": True, "fpga_execution": False})
    manifest = conformance.run_conformance(tmp_path / "run", names=("blend", "fragment"))
    assert manifest["passed"] is False
    assert [stage["status"] for stage in manifest["stages"]] == ["failed", "passed"]
    assert manifest["stages"][0]["evidence"] is None


def test_stop_on_failure_records_not_run_commands(tmp_path, mocked_acceptance):
    calls, failures = mocked_acceptance
    failures["tools.blend_acceptance"] = ("timed_out", None)
    manifest = conformance.run_conformance(tmp_path / "run", names=("blend", "fragment"), stop_on_failure=True)
    assert manifest["passed"] is False and len(calls) == 1
    assert manifest["stages"][1]["status"] == "not_run"
    assert manifest["stages"][1]["command"][2] == "tools.fragment_acceptance"


def test_existing_directory_and_artifacts_are_never_reused(tmp_path, mocked_acceptance):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text('{"passed": true}')
    with pytest.raises(FileExistsError):
        conformance.run_conformance(output, names=("blend",))
    assert sentinel.read_text() == '{"passed": true}'
    assert mocked_acceptance[0] == []


@pytest.mark.parametrize("options", ({"timeout": 0}, {"timeout": 601}, {"timeout": True},
                                     {"total_timeout": float("inf")}, {"total_timeout": float("nan")},
                                     {"total_timeout": 3601}, {"stop_on_failure": 1}, {"extended": 1}))
def test_invalid_limits_do_not_create_output(tmp_path, options):
    output = tmp_path / "invalid-run"
    with pytest.raises(ValueError):
        conformance.run_conformance(output, names=("blend",), **options)
    assert not output.exists()


def test_global_deadline_records_unrun_subset(tmp_path, mocked_acceptance, monkeypatch):
    original = conformance.run_command

    def slow_command(*args, **options):
        result = original(*args, **options)
        time.sleep(0.05)
        return result

    monkeypatch.setattr(conformance, "run_command", slow_command)
    manifest = conformance.run_conformance(tmp_path / "run", names=("blend", "fragment"), total_timeout=0.02)
    assert manifest["passed"] is False
    assert manifest["stages"][1]["status"] == "not_run"
    assert mocked_acceptance[0][0][1]["timeout"] <= 0.02


def test_source_collection_failure_is_retained_in_manifest(tmp_path, monkeypatch):
    def failed_snapshot(*args, **options):
        raise TimeoutError("total deadline exhausted collecting sources")

    monkeypatch.setattr(conformance, "source_snapshot", failed_snapshot)
    manifest = conformance.run_conformance(tmp_path / "run", names=("blend",))
    assert manifest["status"] == "error" and manifest["passed"] is False
    assert "collecting sources" in manifest["error"]
    assert json.loads((tmp_path / "run/manifest.json").read_text())["passed"] is False


def test_render_evidence_checks_actual_rgba_digest(tmp_path):
    stage = conformance.STAGE_BY_NAME["render-gradient"]
    path = tmp_path / "render.json"
    raw = bytes((1, 2, 3, 4))
    path.write_text(json.dumps({"fpga_execution": False, "width": 1, "height": 1,
                               "rgba_sha256": hashlib.sha256(raw).hexdigest(),
                               "softpipe_comparison": {"passed": True}}))
    (tmp_path / "frame.rgba8").write_bytes(raw)
    assert conformance._evidence(path, stage)["report"]["softpipe_comparison"]["passed"] is True
    (tmp_path / "frame.rgba8").write_bytes(bytes(4))
    with pytest.raises(ValueError, match="raw pixels"):
        conformance._evidence(path, stage)


def test_cli_lists_without_running_and_propagates_manifest_failure(tmp_path, capsys, monkeypatch):
    assert conformance.main(["--list"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert len(listing["stages"]) == 17
    feature = next(stage for stage in listing["stages"] if stage["name"] == "gallium-host-features")
    assert feature["module"] == "tools.host_features_acceptance" and feature["evidence"] == "report.json"
    assert feature["sources"] == ["tests/fixtures/host-features-gl/harness.c"]
    monkeypatch.setattr(conformance, "run_conformance", lambda *args, **options: {"passed": False, "unit_test_only": True})
    assert conformance.main(["--stage", "blend", "--output", str(tmp_path / "run")]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
    assert conformance.main([]) == 1
    assert "new directory" in json.loads(capsys.readouterr().err)["error"]
