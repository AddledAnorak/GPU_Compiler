"""Run selected real CPU-host acceptance commands with bounded, fresh provenance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from tools.host_present import bounded_seconds, file_source, parse_json, read_bounded


ROOT = Path(__file__).resolve().parents[1]
SCOPE = "bounded host, not universal Khronos/FPGA certification"
MAX_LOG_BYTES = 1 << 20
MAX_REPORT_BYTES = 4 << 20
EXTENDED_MODULES = ("tools.instancing_acceptance", "tools.mixed_texture_acceptance", "tools.mipmap_texture_acceptance")
HOST_FEATURE_CASE_COUNT = 13
OPTIONAL_HOST_FEATURE_CASES = ("vertex-local-instance-id", "base-instance")


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    arguments: tuple = ()
    evidence: str = "summary.json"
    contract: str = "fresh bounded managed/native CPU-host acceptance"
    sources: tuple = ()

    def command(self, python, build, output, *, extended=False):
        extra = ["--extended"] if extended and self.module in EXTENDED_MODULES else []
        return [str(python), "-m", self.module, *self.arguments, *extra,
                "--build", str(build), "--output", str(output)]


STAGES = (
    Stage("render-flat-hardware", "tools.run_render",
          ("shaders/01_flat_color", "--target", "fp16-hardware-v1", "--compare-softpipe", "--max-channel-error", "0"),
          "render.json", "fresh GL capture -> managed HardwareCPU -> CPU raster; same-draw softpipe comparison"),
    Stage("render-gradient", "tools.run_render",
          ("tests/fixtures/host-gradient", "--compare-softpipe", "--max-channel-error", "0"), "render.json"),
    Stage("render-uniform", "tools.run_render",
          ("shaders/03_uniform", "--compare-softpipe", "--max-channel-error", "0"), "render.json"),
    Stage("render-phong", "tools.run_render",
          ("shaders/07_lighting_phong", "--compare-softpipe", "--max-channel-error", "8"), "render.json"),
    Stage("native-host", "tools.native_host_acceptance"),
    Stage("fragment", "tools.fragment_acceptance"),
    Stage("blend", "tools.blend_acceptance"),
    Stage("blend-equations", "tools.blend_equation_acceptance"),
    Stage("texture", "tools.texture_acceptance"),
    Stage("texture-flow", "tools.texture_flow_acceptance"),
    Stage("linear-texture", "tools.linear_texture_acceptance"),
    Stage("repeat-texture", "tools.repeat_texture_acceptance"),
    Stage("mirrored-texture", "tools.mirrored_texture_acceptance"),
    Stage("mixed-texture", "tools.mixed_texture_acceptance"),
    Stage("mipmap-texture", "tools.mipmap_texture_acceptance"),
    Stage("instancing", "tools.instancing_acceptance"),
    Stage("gallium-host-features", "tools.host_features_acceptance", evidence="report.json",
          contract="13 mandatory ABI4 GL/compiler/managed/native CPU-host cases; optional GLSL 130 VertexID/base-instance gates excluded",
          sources=("tests/fixtures/host-features-gl/harness.c",)),
)
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
SUITES = {"smoke": ("render-flat-hardware", "render-gradient", "native-host"),
          "core": ("render-flat-hardware", "render-gradient", "native-host", "fragment", "blend", "texture", "texture-flow"),
          "all": tuple(STAGE_BY_NAME)}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def select_stages(names=None, *, suite="core"):
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite}")
    selected = SUITES[suite] if names is None else tuple(names)
    if not selected or len(selected) > len(STAGES) or len(set(selected)) != len(selected):
        raise ValueError("select a nonempty, unique bounded set of stages")
    if any(name not in STAGE_BY_NAME for name in selected):
        raise ValueError("unknown host acceptance stage")
    return tuple(STAGE_BY_NAME[name] for name in selected)


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_command(command, *, cwd, timeout, environment=None, max_output=MAX_LOG_BYTES):
    bounded_seconds(timeout, "subprocess timeout", 600)
    if (type(command) not in (list, tuple) or not 1 <= len(command) <= 64 or
            any(type(argument) is not str or not argument or len(argument) > 4096 or "\0" in argument for argument in command)):
        raise ValueError("command must be a bounded argv, never a shell string")
    if type(max_output) is not int or not 1 <= max_output <= MAX_LOG_BYTES:
        raise ValueError("output byte budget must be in 1..1048576")
    started = time.monotonic()
    result = {"command": list(command), "cwd": str(Path(cwd).absolute()), "started_at": timestamp(),
              "timeout_seconds": timeout, "output_limit_bytes": max_output, "returncode": None,
              "status": "error", "error": None}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    process = None
    try:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        deadline = started + timeout
        with selectors.DefaultSelector() as selector:
            for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result.update(status="timed_out", error=f"command exceeded {timeout:g}s")
                    break
                for key, _ in selector.select(min(remaining, 0.05)):
                    contents = os.read(key.fileobj.fileno(), 65536)
                    if not contents:
                        selector.unregister(key.fileobj)
                        continue
                    available = max_output - sum(len(buffer) for buffer in buffers.values())
                    buffers[key.data].extend(contents[:available])
                    if len(contents) > available:
                        result.update(status="output_limit", error="command exceeded aggregate stdout/stderr byte budget")
                        break
                if result["status"] in ("timed_out", "output_limit"):
                    break
            if result["status"] not in ("timed_out", "output_limit"):
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    result["status"] = "passed" if process.returncode == 0 else "failed"
                    if process.returncode:
                        result["error"] = f"command exited {process.returncode}"
                except subprocess.TimeoutExpired:
                    result.update(status="timed_out", error=f"command exceeded {timeout:g}s")
    except OSError as error:
        result["error"] = str(error)
    finally:
        if process is not None:
            _kill_group(process)
            process.wait(timeout=2)
            for stream in (process.stdout, process.stderr):
                stream.close()
            result["returncode"] = process.returncode
    result.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - started)
    return result, bytes(buffers["stdout"]), bytes(buffers["stderr"])


def source_snapshot(build, selected, *, deadline=None):
    def remaining():
        seconds = 5 if deadline is None else min(5, deadline - time.monotonic())
        if seconds <= 0:
            raise TimeoutError("total deadline exhausted during source provenance collection")
        return seconds

    sources = set(ROOT.joinpath("tools").rglob("*.py"))
    sources.update(ROOT.joinpath("tools").glob("*.c"))
    for stage in selected:
        sources.update(ROOT / path for path in stage.sources)
        if stage.module == "tools.run_render":
            sources.update((ROOT / stage.arguments[0]).glob("*.glsl"))
    records = []
    for path in sorted(sources):
        remaining()
        contents = read_bounded(path, 4 << 20)
        records.append(file_source(path, contents))
    binaries = []
    for path in (ROOT / "tools/mgc_harness", build / "src/egl/libEGL.so.1.0.0",
                 *sorted(build.glob("src/gallium/targets/dri/libgallium-*.so"))):
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for contents in iter(lambda: source.read(1 << 20), b""):
                    remaining()
                    digest.update(contents)
            binaries.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    head, head_output, _ = run_command(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=remaining())
    status, status_output, _ = run_command(["git", "--no-optional-locks", "status", "--porcelain=v1"], cwd=ROOT, timeout=remaining())
    return {"tools": records, "binaries": binaries,
            "git_head_command": head, "git_status_command": status,
            "git_head": head_output.decode("utf-8", errors="replace").strip() if head["status"] == "passed" else None,
            "git_status_porcelain": status_output.decode("utf-8", errors="replace"),
            "limitation": "hashes identify host tools/selected GLSL/fixtures/binaries, not an exhaustive Mesa source or toolchain attestation"}


def _host_features_evidence(report, cases):
    if report.get("reference_only") is not False or report.get("complete_default_suite") is not True:
        raise ValueError("host features require complete mandatory execution, never reference-only preflight")
    requested = report.get("requested_cases")
    if (type(requested) is not list or len(requested) != HOST_FEATURE_CASE_COUNT or
            any(type(name) is not str or not name or name in OPTIONAL_HOST_FEATURE_CASES for name in requested) or
            len(set(requested)) != HOST_FEATURE_CASE_COUNT):
        raise ValueError("host features require the 13 mandatory cases; optional VertexID/frontend gates are not execution acceptance")
    if len(cases) != len(requested) or [case.get("case") for case in cases] != requested:
        raise ValueError("host feature case results must match the complete requested mandatory suite")
    if any(case.get("passed") is not True or case.get("status") != "passed" or
           case.get("native_gl_readback") is not True or case.get("shader_abi_version") != 4 or
           case.get("capture_version") != 4 for case in cases):
        raise ValueError("host features require actual passed ABI4 capture/native execution; frontend-blocked probes never qualify")


def _evidence(path, stage):
    contents = read_bounded(path, MAX_REPORT_BYTES)
    report = parse_json(contents)
    if type(report) is not dict or report.get("fpga_execution") is not False:
        raise ValueError("acceptance evidence must explicitly report CPU-host, not FPGA execution")
    if (any(report.get(marker) is True for marker in
            ("physical_or_rtl_validation", "physical_gpu_execution", "rtl_timing_validated", "khronos_certification")) or
            report.get("status") in ("running", "error", "failed")):
        raise ValueError("acceptance evidence is incomplete or exceeds CPU-host scope")
    if stage.evidence == "render.json":
        comparison = report.get("softpipe_comparison")
        if type(comparison) is not dict or comparison.get("passed") is not True:
            raise ValueError("fresh render requires a passing same-draw softpipe comparison")
        width, height = report.get("width"), report.get("height")
        if type(width) is not int or type(height) is not int or not 1 <= width <= 128 or not 1 <= height <= 128:
            raise ValueError("render evidence requires bounded RGBA8 dimensions")
        raw = read_bounded(path.parent / "frame.rgba8", width * height * 4)
        if len(raw) != width * height * 4 or hashlib.sha256(raw).hexdigest() != report.get("rgba_sha256"):
            raise ValueError("fresh render raw pixels do not match report digest")
    elif report.get("passed") is not True:
        raise ValueError("acceptance summary did not pass")
    cases = report.get("cases", [])
    if type(cases) is not list or any(type(case) is not dict or case.get("passed") is False for case in cases):
        raise ValueError("acceptance summary contains invalid or failed cases")
    if stage.name == "gallium-host-features":
        _host_features_evidence(report, cases)
    return {**file_source(path, contents), "report": report}


def _write_manifest(path, manifest):
    temporary = path.with_suffix(".pending")
    with temporary.open("x") as destination:
        destination.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_conformance(output, *, build=None, names=None, suite="core", timeout=120,
                    total_timeout=900, stop_on_failure=False, extended=False):
    selected = select_stages(names, suite=suite)
    bounded_seconds(timeout, "stage timeout", 600)
    bounded_seconds(total_timeout, "total timeout", 3600)
    if type(stop_on_failure) is not bool or type(extended) is not bool:
        raise ValueError("stop_on_failure and extended must be explicitly boolean")
    output = Path(output).absolute()
    build = Path(build if build is not None else os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")).absolute()
    output.mkdir(mode=0o700)
    started = time.monotonic()
    environment = dict(os.environ)
    environment.update(LIBGL_ALWAYS_SOFTWARE="1", EGL_PLATFORM="surfaceless")
    manifest = {"schema": "mgc-host-conformance-v1", "scope": SCOPE, "passed": False, "status": "running",
                "fpga_execution": False, "physical_gpu_execution": False, "khronos_certification": False,
                "started_at": timestamp(), "root": str(ROOT), "build": str(build), "python": sys.executable,
                "selected_stages": [stage.name for stage in selected], "stage_timeout_seconds": timeout,
                "total_timeout_seconds": total_timeout, "stop_on_failure": stop_on_failure,
                "extended_cases": extended,
                "environment": {name: environment.get(name) for name in
                                ("DISPLAY", "EGL_PLATFORM", "LIBGL_ALWAYS_SOFTWARE", "MGC_BUILD_DIR")},
                "stages": [], "source_provenance": None,
                "limitations": ["only selected bounded host contracts; no universal GL/Vulkan/window-system coverage",
                                "software timings are wall time, not device cycles or GPU speedup",
                                "no board commands and no dependency installation; window presentation is tested separately"]}
    manifest_path = output / "manifest.json"
    _write_manifest(manifest_path, manifest)
    try:
        manifest["source_provenance"] = source_snapshot(build, selected, deadline=started + total_timeout)
        failed = False
        for stage in selected:
            directory = output / stage.name
            evidence_directory = directory / "evidence"
            command = stage.command(sys.executable, build, evidence_directory, extended=extended)
            remaining = total_timeout - (time.monotonic() - started)
            if remaining <= 0 or (failed and stop_on_failure):
                manifest["stages"].append({"name": stage.name, "command": command, "status": "not_run",
                                           "passed": False, "error": "total deadline exhausted" if remaining <= 0 else "earlier stage failed"})
                continue
            directory.mkdir()
            result, stdout, stderr = run_command(command, cwd=ROOT, timeout=min(timeout, remaining), environment=environment)
            result.update(name=stage.name, contract=stage.contract, evidence=None)
            for name, contents in (("stdout", stdout), ("stderr", stderr)):
                path = directory / f"{name}.log"
                path.write_bytes(contents)
                result[name] = file_source(path, contents)
            if result["status"] == "passed":
                try:
                    result["evidence"] = _evidence(evidence_directory / stage.evidence, stage)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    result.update(status="failed", error=f"invalid or missing fresh evidence: {error}")
            result["passed"] = result["status"] == "passed"
            failed |= not result["passed"]
            manifest["stages"].append(result)
            _write_manifest(manifest_path, manifest)
        manifest["passed"] = all(stage["passed"] for stage in manifest["stages"])
        manifest["status"] = "passed" if manifest["passed"] else "failed"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        manifest.update(status="error", passed=False, error=str(error))
    manifest.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - started,
                    manifest_path=str(manifest_path))
    _write_manifest(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new output directory; existing files/artifacts are never reused")
    parser.add_argument("--build", type=Path)
    parser.add_argument("--suite", choices=tuple(SUITES), default="core")
    parser.add_argument("--stage", action="append", choices=tuple(STAGE_BY_NAME), help="explicit subset, replacing --suite")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--total-timeout", type=float, default=900)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--extended", action="store_true", help="expand existing instancing/mixed/mipmap case sets")
    parser.add_argument("--list", action="store_true", help="list allowlisted commands without executing")
    args = parser.parse_args(argv)
    try:
        if args.list:
            print(json.dumps({"scope": SCOPE, "suites": SUITES,
                              "stages": [{"name": stage.name, "module": stage.module,
                                          "arguments": stage.arguments, "contract": stage.contract,
                                          "evidence": stage.evidence, "sources": stage.sources} for stage in STAGES]}, indent=2))
            return 0
        if args.output is None:
            raise ValueError("--output must name a new directory for retained fresh provenance")
        manifest = run_conformance(args.output, build=args.build, names=args.stage, suite=args.suite,
                                   timeout=args.timeout, total_timeout=args.total_timeout, stop_on_failure=args.stop_on_failure,
                                   extended=args.extended)
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return 0 if manifest["passed"] else 1
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"schema": "mgc-host-conformance-v1", "passed": False, "scope": SCOPE,
                          "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
