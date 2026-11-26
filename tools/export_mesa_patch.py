#!/usr/bin/env python3
"""Export dirty Mesa sources, proving reconstruction without writing either Git repo."""

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
MGC = "src/gallium/drivers/mgc"
SOFTPIPE = "src/gallium/drivers/softpipe/sp_tex_sample.c"
MGC_PATCH = "0003-mgc-bounded-runtime-compiler-and-host-artifacts.patch"
SOFTPIPE_PATCH = "0004-softpipe-mirrored-linear-center-weight.patch"
METADATA = "mesa-export.json"
DIFF_OPTIONS = (
    "--binary", "--full-index", "--no-ext-diff", "--no-textconv",
    "--no-renames", "--no-color", "--diff-algorithm=myers",
    "--no-indent-heuristic", "--unified=3", "--src-prefix=a/", "--dst-prefix=b/",
)


class ExportError(RuntimeError):
    """The export cannot be proven safe or reconstructible."""


@dataclass(frozen=True)
class SourceFile:
    data: bytes
    mode: int


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _outside_originals(path, protected):
    if any(path.is_relative_to(original) for original in protected):
        raise ExportError(f"output/scratch must be outside original worktrees and Git directories: {path}")


class _Git:
    """Only read commands in originals; apply and no-index diff only in scratch."""

    def __init__(self, repo, mesa, timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ExportError("timeout must be finite and positive")
        self.repo = repo
        self.mesa = mesa
        self.originals = {repo, mesa}
        self.scratch_root = None
        self.deadline = time.monotonic() + timeout
        self.commands = []
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith("GIT_")}
        self.environment.update({
            "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1",
            "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull, "LC_ALL": "C", "TZ": "UTC",
        })

    def read(self, cwd, arguments, allowed=(0,)):
        if cwd not in self.originals or not arguments:
            raise ExportError("read-only Git commands require an original repository root")
        request = tuple(arguments)
        permitted = {
            ("rev-parse", "--verify", "HEAD"),
            ("rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir",
             "--git-path", "index", "--git-path", "HEAD"),
            ("symbolic-ref", "--quiet", "--short", "HEAD"),
            ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
            ("ls-files", "--others", "--exclude-standard", "-z", "--", MGC),
        }
        head_position = 1 + len(DIFF_OPTIONS) if arguments[0] == "diff" else 2
        if arguments[0] in {"diff", "archive"} and len(arguments) > head_position:
            head = arguments[head_position]
            if len(head) in {40, 64} and all(character in "0123456789abcdef" for character in head):
                permitted.update(("diff", *DIFF_OPTIONS, head, "--", scope)
                                 for scope in (MGC, SOFTPIPE))
                permitted.add(("archive", "--format=tar", head, "--", MGC, SOFTPIPE))
        if request not in permitted:
            raise ExportError("only exact read-only Git requests are permitted in original repositories")
        return self._run(cwd, arguments, allowed, None, "original/read-only")

    def scratch(self, cwd, arguments, data=None, allowed=(0,)):
        cwd = cwd.resolve()
        _outside_originals(cwd, self.originals)
        if self.scratch_root is None or not cwd.is_relative_to(self.scratch_root):
            raise ExportError("patch application requires this export's scratch directory")
        if not arguments or arguments[0] not in {"apply", "diff"}:
            raise ExportError("scratch Git commands are restricted to apply and no-index diff")
        if arguments[0] == "diff" and "--no-index" not in arguments:
            raise ExportError("scratch diff must use --no-index")
        return self._run(cwd, arguments, allowed, data, "scratch")

    def _run(self, cwd, arguments, allowed, data, scope):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ExportError("export exceeded its Git timeout budget")
        environment = self.environment.copy()
        if scope == "scratch":
            environment["GIT_CEILING_DIRECTORIES"] = str(self.scratch_root.parent)
            label = "<scratch>/" + cwd.relative_to(self.scratch_root).as_posix()
        else:
            label = "<mesa>" if cwd == self.mesa else "<repo>"
        command = ["git", "--no-pager", "-c", "core.fsmonitor=false",
                   "-c", "core.quotePath=true", "-c", "color.ui=false", *arguments]
        try:
            result = subprocess.run(command, cwd=cwd, env=environment, input=data,
                                    capture_output=True, timeout=remaining, check=False)
        except subprocess.TimeoutExpired as error:
            raise ExportError("export exceeded its Git timeout budget") from error
        self.commands.append({"cwd": label, "argv": list(arguments),
                              "scope": scope, "returncode": result.returncode})
        if result.returncode not in allowed:
            detail = result.stderr.decode("utf-8", errors="replace")[:2000].strip()
            raise ExportError(f"git {arguments[0]} failed in {label}: {detail}")
        return result.stdout


def _repository_state(git, repository):
    head = git.read(repository, ["rev-parse", "--verify", "HEAD"]).decode("ascii").strip()
    branch = git.read(repository, ["symbolic-ref", "--quiet", "--short", "HEAD"],
                      allowed=(0, 1))
    status = git.read(repository, ["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    locations = git.read(repository, ["rev-parse", "--path-format=absolute", "--git-dir",
                                     "--git-common-dir", "--git-path", "index", "--git-path", "HEAD"])
    git_dir, common_dir, index_path, head_path = map(Path, os.fsdecode(locations).splitlines())
    return {
        "head": head, "branch": os.fsdecode(branch).strip() or None,
        "status_porcelain_v1_z": os.fsdecode(status), "status_sha256": sha256(status),
        "git_dir": str(git_dir), "git_common_dir": str(common_dir),
        "index_sha256": sha256(index_path.read_bytes()) if index_path.exists() else None,
        "head_file_sha256": sha256(head_path.read_bytes()),
    }


def _read_file(path):
    if path.is_symlink() or path.resolve() != path:
        raise ExportError(f"symlink sources are not supported: {path}")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ExportError(f"source must be a regular file: {path}")
    return SourceFile(path.read_bytes(), 0o100755 if mode & 0o111 else 0o100644)


def _snapshot_sources(base):
    """Inventory every MGC file, not just changed files, plus the softpipe source."""
    driver = base / MGC
    if not driver.is_dir() or driver.resolve() != driver:
        raise ExportError(f"missing or symlink MGC directory: {driver}")
    sources = {}

    def walk_error(error):
        raise ExportError(f"cannot inventory MGC sources: {error}") from error

    for directory, directories, filenames in os.walk(driver, onerror=walk_error, followlinks=False):
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise ExportError(f"symlink source directory: {Path(directory) / name}")
        for name in filenames:
            path = Path(directory) / name
            sources[path.relative_to(base).as_posix()] = _read_file(path)
    sources[SOFTPIPE] = _read_file(base / SOFTPIPE)
    return dict(sorted(sources.items()))


def _manifest(sources):
    return {path: {"sha256": sha256(source.data), "size_bytes": len(source.data),
                   "mode": format(source.mode, "o")}
            for path, source in sorted(sources.items())}


def _manifest_hash(manifest):
    return sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii"))


def _compare_sources(expected, actual):
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(path for path in expected.keys() & actual.keys() if expected[path] != actual[path])
    if missing or unexpected or changed:
        raise ExportError(f"reconstruction mismatch: missing={missing}, unexpected={unexpected}, changed={changed}")


def _extract_archive(archive, destination):
    """Extract only regular files/directories in the requested scope, without tar links."""
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as members:
        for member in members:
            relative = PurePosixPath(member.name)
            name = relative.as_posix()
            if relative.is_absolute() or ".." in relative.parts or name in seen:
                raise ExportError(f"unsafe archive path: {member.name}")
            in_scope = name == SOFTPIPE or name == MGC or name.startswith(MGC + "/")
            ancestor = MGC.startswith(name + "/") or SOFTPIPE.startswith(name + "/")
            if not in_scope and not (member.isdir() and ancestor):
                raise ExportError(f"archive entry outside requested sources: {member.name}")
            if not member.isdir() and not member.isfile():
                raise ExportError(f"archive links/special files are forbidden: {member.name}")
            seen.add(name)
            target = destination / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with members.extractfile(member) as contents:
                    target.write_bytes(contents.read())
                target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _untracked_patch(git, scratch, relative, source):
    if not relative.startswith(MGC + "/") or ".." in PurePosixPath(relative).parts:
        raise ExportError(f"untracked file outside MGC: {relative}")
    target = scratch / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.data)
    target.chmod(0o755 if source.mode == 0o100755 else 0o644)
    return git.scratch(scratch, ["diff", "--no-index", *DIFF_OPTIONS,
                                 "--", os.devnull, relative], allowed=(0, 1))


def _check_output(output, protected):
    _outside_originals(output, protected)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ExportError(f"output must be a new or empty directory; refusing to overwrite: {output}")


def _publish(output, patches, metadata, protected):
    _check_output(output, protected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mesa-export-publish-", dir=output.parent) as temporary:
        staged = Path(temporary) / "export"
        staged.mkdir()
        for name, data in patches.items():
            (staged / name).write_bytes(data)
        (staged / METADATA).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
        _check_output(output, protected)
        staged.rename(output)


def export_mesa_patch(repo, mesa, output, *, timeout=30.0):
    """Return metadata and publish patches only after a scratch reconstruction and stable proofs."""
    repo, mesa, output = (Path(path).resolve() for path in (repo, mesa, output))
    git = _Git(repo, mesa, timeout)
    before = {"repo": _repository_state(git, repo), "mesa": _repository_state(git, mesa)}
    protected = {repo, mesa}
    for state in before.values():
        protected.update((Path(state["git_dir"]).resolve(), Path(state["git_common_dir"]).resolve()))
    _check_output(output, protected)
    _outside_originals(Path(tempfile.gettempdir()).resolve(), protected)
    sources = _snapshot_sources(mesa)
    manifest = _manifest(sources)
    head = before["mesa"]["head"]
    untracked = sorted(os.fsdecode(path) for path in git.read(
        mesa, ["ls-files", "--others", "--exclude-standard", "-z", "--", MGC]).split(b"\0") if path)
    patches = {
        MGC_PATCH: git.read(mesa, ["diff", *DIFF_OPTIONS, head, "--", MGC]),
        SOFTPIPE_PATCH: git.read(mesa, ["diff", *DIFF_OPTIONS, head, "--", SOFTPIPE]),
    }
    archive = git.read(mesa, ["archive", "--format=tar", head, "--", MGC, SOFTPIPE])
    with tempfile.TemporaryDirectory(prefix="mesa-export-reconstruct-") as temporary:
        git.scratch_root = Path(temporary).resolve()
        reconstructed = git.scratch_root / "reconstructed"
        additions = git.scratch_root / "new-files"
        reconstructed.mkdir()
        additions.mkdir()
        _extract_archive(archive, reconstructed)
        for relative in untracked:
            if relative not in sources:
                raise ExportError(f"untracked source disappeared from snapshot: {relative}")
            patches[MGC_PATCH] += _untracked_patch(git, additions, relative, sources[relative])
        combined = b"".join(patch for patch in patches.values() if patch)
        if combined:
            git.scratch(reconstructed, ["apply", "--check", "--whitespace=nowarn", "-"], data=combined)
            git.scratch(reconstructed, ["apply", "--whitespace=nowarn", "-"], data=combined)
        restored = _snapshot_sources(reconstructed)
        _compare_sources(sources, restored)
    sources_after = _snapshot_sources(mesa)
    after = {"repo": _repository_state(git, repo), "mesa": _repository_state(git, mesa)}
    if sources_after != sources or after != before:
        raise ExportError("original HEAD/status/index/source snapshot changed during export; no output published")
    metadata = {
        "schema_version": 1, "mesa_head": head,
        "patches": {name: {"sha256": sha256(data), "size_bytes": len(data),
                            "lines": data.count(b"\n"),
                            "scope": MGC if name == MGC_PATCH else SOFTPIPE,
                            "untracked_files": untracked if name == MGC_PATCH else []}
                    for name, data in patches.items()},
        "sources": manifest, "source_sha256": _manifest_hash(manifest),
        "reconstruction": {
            "method": "read-only git archive HEAD; forward git apply in TemporaryDirectory only",
            "archive_head": head, "archive_sha256": sha256(archive),
            "forward_apply_check": bool(combined), "forward_apply": bool(combined),
            "byte_for_byte": True, "file_modes_match": True,
            "files_checked": len(sources), "mgc_files_checked": len(sources) - 1,
            "all_mgc_files": sorted(path for path in sources if path.startswith(MGC + "/")),
            "reconstructed_source_sha256": _manifest_hash(_manifest(restored)),
        },
        "original_worktrees": {"before": before, "after": after, "unchanged": True,
                               "source_sha256_before": _manifest_hash(manifest),
                               "source_sha256_after": _manifest_hash(_manifest(sources_after))},
        "safety": {"original_git_writes": False, "original_patch_applications": False,
                   "scratch_only_patch_application": True, "optional_git_locks": False,
                   "commands": git.commands},
    }
    _publish(output, patches, metadata, protected)
    return metadata


def _positive_timeout(value):
    try:
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
        return timeout
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be finite and positive") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT, help="root project checkout (read only)")
    parser.add_argument("--mesa", type=Path, help="Mesa checkout (default: REPO/mesa; read only)")
    parser.add_argument("--output", type=Path, required=True,
                        help="new or empty directory outside both original worktrees")
    parser.add_argument("--timeout", type=_positive_timeout, default=30.0,
                        help="total Git subprocess timeout budget in seconds (default: 30)")
    args = parser.parse_args(argv)
    try:
        metadata = export_mesa_patch(args.repo, args.mesa or args.repo / "mesa", args.output,
                                     timeout=args.timeout)
    except (ExportError, OSError, tarfile.TarError) as error:
        print(f"Mesa export failed: {error}", file=sys.stderr)
        return 1
    print(f"Exported HEAD {metadata['mesa_head']}: "
          f"{metadata['reconstruction']['files_checked']} files match byte-for-byte; "
          f"original Git worktrees unchanged. Proof: {args.output / METADATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
