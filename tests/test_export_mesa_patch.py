"""Scratch-only patch reconstruction, plus bounded exports of the actual dirty Mesa."""

import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from tools import export_mesa_patch as exporter


HELPERS = tuple(f"{exporter.MGC}/{name}" for name in (
    "mgc_host_executor.c", "mgc_host_executor.h", "mgc_test_backend.c",
))
UNCHANGED = tuple(f"{exporter.MGC}/{name}" for name in ("mgc_test_arith.c", "mgc_test_trig.c"))


def make_archive(files):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as members:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            members.addfile(member, io.BytesIO(data))
    return archive.getvalue()


def modification_patch(relative, before, after):
    return (f"diff --git a/{relative} b/{relative}\n"
            f"--- a/{relative}\n+++ b/{relative}\n@@ -1 +1 @@\n"
            f"-{before}\n+{after}\n").encode("ascii")


@pytest.fixture
def fake_repositories(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    mesa = repo / "mesa"
    compiler = f"{exporter.MGC}/mgc_compiler.c"
    base = {f"{exporter.MGC}/meson.build": b"sources = []\n",
            compiler: b"old compiler\n", exporter.SOFTPIPE: b"old weight\n",
            **{path: b"unchanged\n" for path in UNCHANGED}}
    current = {**base, compiler: b"new compiler\n", exporter.SOFTPIPE: b"new weight\n",
               **{path: f"new helper {index}\n".encode("ascii")
                  for index, path in enumerate(HELPERS)}}
    for name, data in current.items():
        target = mesa / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for directory in (repo, mesa):
        (directory / ".git").mkdir()
        (directory / ".git/HEAD").write_bytes(b"ref: refs/heads/fixture\n")
        (directory / ".git/index").write_bytes(b"unchanged fixture index")
    head = "a" * 40
    original_read = exporter._Git.read

    def fake_read(git, cwd, arguments, allowed=(0,)):
        command = arguments[0]
        if command == "rev-parse":
            if "--verify" in arguments:
                return (head + "\n").encode("ascii")
            return "\n".join(str(cwd / suffix) for suffix in (
                ".git", ".git", ".git/index", ".git/HEAD",
            )).encode("utf-8") + b"\n"
        if command == "symbolic-ref":
            return b"mgc-driver\n" if cwd == mesa else b"master\n"
        if command == "status":
            return b" M dirty-file\0"
        if command == "ls-files":
            return b"\0".join(path.encode("ascii") for path in reversed(HELPERS)) + b"\0"
        if command == "diff":
            assert head in arguments
            assert "--binary" in arguments and "--full-index" in arguments
            if arguments[-1] == exporter.MGC:
                return modification_patch(compiler, "old compiler", "new compiler")
            return modification_patch(exporter.SOFTPIPE, "old weight", "new weight")
        if command == "archive":
            assert arguments == ["archive", "--format=tar", head, "--", exporter.MGC, exporter.SOFTPIPE]
            return make_archive(base)
        return original_read(git, cwd, arguments, allowed)

    monkeypatch.setattr(exporter._Git, "read", fake_read)
    return repo, mesa, current


def assert_proof(metadata, expected):
    proof = metadata["reconstruction"]
    assert proof["byte_for_byte"] and proof["file_modes_match"]
    assert proof["forward_apply_check"] and proof["forward_apply"]
    assert proof["files_checked"] == len(expected)
    assert set(metadata["sources"]) == set(expected)
    for relative, data in expected.items():
        assert metadata["sources"][relative]["sha256"] == exporter.sha256(data)
    assert proof["reconstructed_source_sha256"] == metadata["source_sha256"]
    originals = metadata["original_worktrees"]
    assert originals["unchanged"] and originals["before"] == originals["after"]
    assert originals["source_sha256_before"] == originals["source_sha256_after"]
    assert not metadata["safety"]["original_git_writes"]
    assert not metadata["safety"]["original_patch_applications"]
    for command in metadata["safety"]["commands"]:
        if command["argv"][0] == "apply":
            assert command["scope"] == "scratch"
            assert command["cwd"].startswith("<scratch>/")


def test_export_includes_helpers_and_unchanged_sources(fake_repositories, tmp_path):
    repo, mesa, current = fake_repositories
    output = tmp_path / "export"
    metadata = exporter.export_mesa_patch(repo, mesa, output, timeout=10)
    assert_proof(metadata, current)
    assert metadata["mesa_head"] == "a" * 40
    assert metadata["patches"][exporter.MGC_PATCH]["untracked_files"] == sorted(HELPERS)
    mgc_patch = (output / exporter.MGC_PATCH).read_bytes()
    softpipe_patch = (output / exporter.SOFTPIPE_PATCH).read_bytes()
    assert exporter.SOFTPIPE.encode("ascii") not in mgc_patch
    assert exporter.MGC.encode("ascii") not in softpipe_patch
    assert all(path.encode("ascii") in mgc_patch for path in HELPERS)
    assert all(path.encode("ascii") not in mgc_patch for path in UNCHANGED)
    assert json.loads((output / exporter.METADATA).read_text()) == metadata
    assert set(path.name for path in output.iterdir()) == {
        exporter.MGC_PATCH, exporter.SOFTPIPE_PATCH, exporter.METADATA,
    }


@pytest.mark.parametrize("data", [b"new file\n", b"no final newline", b"", b"\0binary\xff\n"])
@pytest.mark.parametrize("mode", [0o100644, 0o100755])
def test_untracked_diff_roundtrips_bytes_and_modes(tmp_path, data, mode):
    repo, mesa = tmp_path / "repo", tmp_path / "mesa"
    scratch = tmp_path / "scratch"
    additions, restored = scratch / "additions", scratch / "restored"
    additions.mkdir(parents=True)
    restored.mkdir()
    git = exporter._Git(repo, mesa, 10)
    git.scratch_root = scratch
    relative = f"{exporter.MGC}/space and \"quote\".c"
    source = exporter.SourceFile(data, mode)
    patch = exporter._untracked_patch(git, additions, relative, source)
    git.scratch(restored, ["apply", "--check", "--whitespace=nowarn", "-"], data=patch)
    git.scratch(restored, ["apply", "--whitespace=nowarn", "-"], data=patch)
    assert exporter._read_file(restored / relative) == source


@pytest.mark.parametrize("arguments", [
    ["apply", "--check", "-"], ["apply", "--reverse", "-"], ["add", "."],
    ["commit", "-m", "forbidden"], ["checkout", "HEAD"], ["reset", "--hard"],
    ["symbolic-ref", "HEAD", "refs/heads/forbidden"],
    ["diff", *exporter.DIFF_OPTIONS, "--output=/tmp/forbidden"],
    ["archive", "HEAD", "--output=/tmp/forbidden"],
])
def test_original_git_write_guard(tmp_path, arguments, monkeypatch):
    repo, mesa = tmp_path / "repo", tmp_path / "mesa"
    git = exporter._Git(repo, mesa, 10)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("forbidden commands must never reach subprocess.run")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    with pytest.raises(exporter.ExportError):
        git.read(mesa, arguments)
    with pytest.raises(exporter.ExportError):
        git.scratch(mesa, ["apply", "-"])


def test_git_environment_cannot_redirect_or_lock_originals(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/unrelated/git")
    monkeypatch.setenv("GIT_WORK_TREE", "/unrelated/tree")
    monkeypatch.setenv("GIT_INDEX_FILE", "/unrelated/index")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    git = exporter._Git(tmp_path / "repo", tmp_path / "mesa", 10)
    assert not {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG_COUNT"} & git.environment.keys()
    assert git.environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert git.environment["GIT_NO_LAZY_FETCH"] == "1"


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "outside", "duplicate"])
def test_archive_extraction_rejects_unsafe_members(tmp_path, kind):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as members:
        member = tarfile.TarInfo(f"{exporter.MGC}/safe.c")
        if kind == "traversal":
            member.name = f"{exporter.MGC}/../../escape.c"
        elif kind == "absolute":
            member.name = "/tmp/escape.c"
        elif kind in {"symlink", "hardlink"}:
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "/tmp/escape.c"
        elif kind == "outside":
            member.name = "unrelated.c"
        members.addfile(member)
        if kind == "duplicate":
            members.addfile(member)
    with pytest.raises(exporter.ExportError):
        exporter._extract_archive(archive.getvalue(), tmp_path / "scratch")


def test_missing_untracked_helper_fails_without_output(fake_repositories, tmp_path, monkeypatch):
    repo, mesa, _ = fake_repositories
    original = exporter._untracked_patch

    def drop_helper(git, scratch, relative, source):
        return b"" if relative == HELPERS[0] else original(git, scratch, relative, source)

    monkeypatch.setattr(exporter, "_untracked_patch", drop_helper)
    output = tmp_path / "failed-export"
    with pytest.raises(exporter.ExportError, match="reconstruction mismatch.*mgc_host_executor.c"):
        exporter.export_mesa_patch(repo, mesa, output, timeout=10)
    assert not output.exists()


def test_ignored_source_cannot_be_silently_dropped(fake_repositories, tmp_path):
    repo, mesa, _ = fake_repositories
    (mesa / exporter.MGC / "ignored-helper.c").write_bytes(b"must not vanish\n")
    output = tmp_path / "failed-export"
    with pytest.raises(exporter.ExportError, match="reconstruction mismatch.*ignored-helper.c"):
        exporter.export_mesa_patch(repo, mesa, output, timeout=10)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["source", "status", "head", "index"])
def test_concurrent_original_changes_fail_without_output(fake_repositories, tmp_path, monkeypatch, mutation):
    repo, mesa, _ = fake_repositories
    original = exporter._repository_state
    counts = {repo: 0, mesa: 0}

    def changed_state(git, directory):
        counts[directory] += 1
        state = original(git, directory)
        if directory == mesa and counts[directory] == 2:
            if mutation == "source":
                (mesa / HELPERS[0]).write_bytes(b"parent changed capture\n")
            else:
                key = {"status": "status_sha256", "head": "head", "index": "index_sha256"}[mutation]
                state[key] = "changed concurrently"
        return state

    if mutation == "source":
        original_snapshot = exporter._snapshot_sources
        snapshot_counts = {mesa: 0}

        def changed_snapshot(base):
            if base == mesa:
                snapshot_counts[mesa] += 1
                if snapshot_counts[mesa] == 2:
                    (mesa / HELPERS[0]).write_bytes(b"parent changed capture\n")
            return original_snapshot(base)

        monkeypatch.setattr(exporter, "_snapshot_sources", changed_snapshot)
    else:
        monkeypatch.setattr(exporter, "_repository_state", changed_state)
    output = tmp_path / "failed-export"
    with pytest.raises(exporter.ExportError, match="changed during export"):
        exporter.export_mesa_patch(repo, mesa, output, timeout=10)
    assert not output.exists()


def test_output_rejects_originals_and_existing_artifacts(fake_repositories, tmp_path):
    repo, mesa, _ = fake_repositories
    existing = tmp_path / "existing"
    existing.mkdir()
    artifact = existing / exporter.MGC_PATCH
    artifact.write_bytes(b"preserve me")
    for output in (repo / "exports", mesa / "exports", repo / ".git/export", existing):
        with pytest.raises(exporter.ExportError):
            exporter.export_mesa_patch(repo, mesa, output, timeout=10)
    assert artifact.read_bytes() == b"preserve me"
    assert not (repo / "exports").exists() and not (mesa / "exports").exists()


def test_symlink_sources_are_rejected(fake_repositories, tmp_path):
    _, mesa, _ = fake_repositories
    outside = tmp_path / "outside.c"
    outside.write_bytes(b"outside source\n")
    (mesa / exporter.MGC / "symlink.c").symlink_to(outside)
    with pytest.raises(exporter.ExportError, match="symlink"):
        exporter._snapshot_sources(mesa)


def test_timeout_is_bounded_and_leaves_no_output(fake_repositories, tmp_path, monkeypatch):
    repo, mesa, _ = fake_repositories

    def timed_out(*args, **kwargs):
        assert 0 < kwargs["timeout"] <= 2
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)
    output = tmp_path / "failed-export"
    with pytest.raises(exporter.ExportError, match="timeout budget"):
        exporter.export_mesa_patch(repo, mesa, output, timeout=2)
    assert not output.exists()


def test_cli_requires_explicit_output(capsys):
    with pytest.raises(SystemExit) as error:
        exporter.main([])
    assert error.value.code == 2
    assert "--output" in capsys.readouterr().err


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "bad"])
def test_cli_rejects_invalid_timeouts(timeout, capsys):
    with pytest.raises(SystemExit) as error:
        exporter.main(["--output", "/tmp/unused-export", "--timeout", timeout])
    assert error.value.code == 2
    assert "finite and positive" in capsys.readouterr().err


def test_current_dirty_mesa_export_is_reproducible_and_read_only(tmp_path, monkeypatch):
    repo, mesa = exporter.ROOT, exporter.ROOT / "mesa"
    if not (mesa / ".git").exists() or not shutil.which("git"):
        pytest.skip("actual Mesa checkout and Git required for bounded integration export")
    sources = exporter._snapshot_sources(mesa)
    expected = {path: source.data for path, source in sources.items()}
    canonical = {name: (repo / "mesa-patches" / name).read_bytes()
                 for name in (exporter.MGC_PATCH, exporter.SOFTPIPE_PATCH)}
    recorded = []
    original_run = subprocess.run

    def audited_run(command, **kwargs):
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        cwd = Path(kwargs["cwd"]).resolve()
        arguments = command[command.index("color.ui=false") + 1:]
        if cwd in {repo, mesa}:
            assert arguments[0] in {"rev-parse", "symbolic-ref", "status", "diff", "ls-files", "archive"}
        else:
            assert not cwd.is_relative_to(repo)
            assert arguments[0] in {"apply", "diff"}
            if arguments[0] == "diff":
                assert "--no-index" in arguments
        assert 0 < kwargs["timeout"] <= 20
        recorded.append((cwd, arguments))
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", audited_run)
    first = tmp_path / "first"
    second = tmp_path / "second"
    metadata = exporter.export_mesa_patch(repo, mesa, first, timeout=20)
    repeated = exporter.export_mesa_patch(repo, mesa, second, timeout=20)
    assert_proof(metadata, expected)
    assert repeated == metadata
    assert exporter._snapshot_sources(mesa) == sources
    assert all(path in metadata["sources"] for path in (*HELPERS, *UNCHANGED))
    for name in (exporter.MGC_PATCH, exporter.SOFTPIPE_PATCH, exporter.METADATA):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for name, data in canonical.items():
        assert (repo / "mesa-patches" / name).read_bytes() == data
        assert metadata["patches"][name]["sha256"] == exporter.sha256((first / name).read_bytes())
    assert any(arguments[0] == "archive" for _, arguments in recorded)
    assert sum(arguments[0] == "apply" and "--check" not in arguments
               for _, arguments in recorded) == 2
