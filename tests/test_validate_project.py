"""Preserve real test progress when the integrated validator hits its deadline."""

import subprocess

import pytest

from tools.validate_project import source_hashes, timeout_log


@pytest.mark.parametrize("output,errors,expected", ((None, None, ""),
                                                  ("progress\n", "diagnostic\n", "progress\ndiagnostic\n"),
                                                  (b"progress\n", b"diagnostic\n", "progress\ndiagnostic\n"),
                                                  (b"progress\n", None, "progress\n"),
                                                  (None, "diagnostic\n", "diagnostic\n"),
                                                  (b"invalid\xff", "", "invalid\ufffd")))
def test_timeout_keeps_partial_progress_and_failure(output, errors, expected):
    error = subprocess.TimeoutExpired(["pytest", "-q"], 600, output=output, stderr=errors)
    actual = timeout_log(error, 600)
    assert actual.startswith(expected + "\n")
    assert "stage exceeded 600s" in actual
    assert "timed out" in actual


def test_source_hashes_include_nested_fixtures_and_untracked_mesa_helpers(tmp_path):
    inputs = ("tools/host.py", "tests/fixtures/gl/helper.c", "shaders/demo/frag.glsl",
              "docs/isa/isa.yaml", "mesa/src/gallium/drivers/mgc/new_helper.h",
              "interpreter/src/mgc_interpreter.c", "interpreter/src/mgc_fp16_math.c",
              "interpreter/include/mgc_interpreter.h", "interpreter/include/mgc_fp16_isa.h",
              "mesa/src/gallium/drivers/mgc/meson.build",
              "mesa/src/gallium/drivers/softpipe/sp_tex_sample.c")
    for name in inputs:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    cache = tmp_path / "tools/__pycache__/host.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"compiled")
    before = source_hashes(tmp_path)
    assert set(before) == set(inputs)
    assert source_hashes(tmp_path) == before
    (tmp_path / inputs[0]).write_text("changed")
    assert source_hashes(tmp_path)[inputs[0]] != before[inputs[0]]


def test_source_hashes_reject_linked_sources(tmp_path):
    directory = tmp_path / "tools"
    directory.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("source")
    (directory / "linked.py").symlink_to(outside)
    with pytest.raises(ValueError, match="ordinary file"):
        source_hashes(tmp_path)


@pytest.mark.parametrize("mutation", ("unchanged", "change", "add", "delete"))
def test_passing_commands_require_stable_sources(tmp_path, monkeypatch, mutation):
    import json
    from tools import validate_project

    source = tmp_path / "tools/source.py"
    source.parent.mkdir()
    source.write_text("before")
    monkeypatch.setattr(validate_project, "ROOT", tmp_path)

    def run(command, **kwargs):
        if mutation == "change":
            source.write_text("after")
        elif mutation == "add":
            source.with_name("new.py").write_text("new")
        elif mutation == "delete":
            source.unlink()
        return subprocess.CompletedProcess(command, 0, "1 passed\n", "")

    monkeypatch.setattr(validate_project.subprocess, "run", run)
    output = tmp_path / "validation"
    status = validate_project.main(["--output", str(output), "--no-build", "--tests-only"])
    report = json.loads((output / "summary.json").read_text())
    assert status == (mutation != "unchanged")
    assert report["all_passed"] == report["source_freeze"]["unchanged"] == (mutation == "unchanged")
    assert report["source_freeze"]["files_checked"] == 1
    assert all(stage["exit_code"] == 0 for stage in report["stages"])
