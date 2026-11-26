"""Rebuild and validate host deliverables without opening any physical device."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def timeout_log(error, timeout):
    captured = "".join(part.decode("utf-8", errors="replace") if isinstance(part, bytes) else part or ""
                       for part in (error.stdout, error.stderr))
    return f"{captured}\nstage exceeded {timeout}s: {error}\n"


def source_hashes(root: Path) -> dict[str, str]:
    suffixes = {".py", ".c", ".h", ".glsl", ".sh", ".yaml", ".json"}
    directories = ("tools", "tests", "shaders", "docs/isa", "interpreter", "mesa/src/gallium/drivers/mgc")
    paths = {path for directory in directories for path in (root / directory).rglob("*")
             if path.is_file() and path.suffix in suffixes}
    softpipe = root / "mesa/src/gallium/drivers/softpipe/sp_tex_sample.c"
    if softpipe.is_file():
        paths.add(softpipe)
    meson = root / "mesa/src/gallium/drivers/mgc/meson.build"
    if meson.is_file():
        paths.add(meson)
    result = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError(f"validation source must be an ordinary file: {path}")
        with path.open("rb") as source:
            result[path.relative_to(root).as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/validation")
    parser.add_argument("--no-build", action="store_true", help="explicitly reuse a previously rebuilt Mesa")
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    environment["MPLBACKEND"] = "Agg"
    python = sys.executable
    build = Path(environment.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
    stages = []
    if not args.no_build:
        stages.append(("build", ["bash", "tools/build_project.sh"], 900))
    stages.append(("tests", [python, "-m", "pytest", "-q", "-rs"], 900))
    if not args.tests_only:
        stages.extend([
            ("current-c-phong", [python, "-m", "tools.run_c_shader", "shaders/07_lighting_phong",
                                  "--build", str(build), "--output", str(output / "current-c-phong"),
                                  "--phong"], 90),
            ("frontend", [python, "-m", "tools.validate_frontend", "--trace-dir", str(output / "frontend")], 120),
            ("board-preflight-template", [python, "-c",
                "import json; from pathlib import Path; from tools.mgcdriver.preflight import assess_json; "
                "report = assess_json(Path('docs/driver/board-manifest.template.json').read_bytes()); "
                "print(json.dumps(report, indent=2)); "
                "assert report['status'] == 'missing' and report['plan'] is None "
                "and report['hardware_execution'] is False and report['readiness_unverified'] is True "
                "and report['references_verified'] is False"], 30),
            ("hardware-encoding", [python, "-m", "tools.mgchw", "verify", "--report",
                                   str(output / "hardware-encoding.json")], 60),
            ("glsl-host-pipeline", [python, "-m", "tools.run_shader", "shaders/01_flat_color", "--output",
                                    str(output / "shader-flat-color")], 60),
            ("glsl-paired-pipeline", [python, "-m", "tools.run_shader", "shaders/12_runtime_arithmetic",
                                      "--varying-map", "tests/fixtures/shader-pair-varyings.json",
                                      "--vertex-values", "tests/fixtures/shader-pair-12-vertex.json",
                                      "--output", str(output / "shader-pair-arithmetic")], 60),
            ("glsl-hardware-target", [python, "-m", "tools.run_shader", "shaders/01_flat_color",
                                      "--target", "fp16-hardware-v1", "--output",
                                      str(output / "shader-hardware-flat-color")], 60),
            ("glsl-vector-phong", [python, "-m", "tools.run_vector_shader", "shaders/07_lighting_phong",
                                     "--allow-approximate-fp16", "--output",
                                     str(output / "shader-vector-phong")], 60),
            ("glsl-hardware-copy", [python, "-m", "tools.run_shader", "shaders/02_varying",
                                    "--target", "fp16-hardware-v1",
                                    "--varying-map", "tests/fixtures/shader-pair-varyings.json",
                                    "--vertex-values", "tests/fixtures/hardware-varying-vertex.json",
                                    "--output", str(output / "shader-hardware-varying")], 60),
            ("glsl-reflected-link", [python, "-m", "tools.run_shader", "shaders/12_runtime_arithmetic", "--auto-link",
                                      "--vertex-values", "tests/fixtures/shader-pair-12-vertex.json",
                                      "--output", str(output / "shader-reflected-arithmetic")], 60),
            ("hardware-reflected-link", [python, "-m", "tools.run_shader", "shaders/02_varying", "--auto-link",
                                          "--target", "fp16-hardware-v1", "--vertex-values",
                                          "tests/fixtures/hardware-varying-vertex.json",
                                          "--output", str(output / "shader-reflected-hardware")], 60),
            ("glsl-hardware-add", [python, "-m", "tools.run_shader", "tests/fixtures/hardware-add", "--auto-link",
                                   "--target", "fp16-hardware-v1", "--vertex-values",
                                   "tests/fixtures/hardware-varying-vertex.json",
                                   "--output", str(output / "shader-hardware-add")], 60),
            ("gallium-hardware-add", [python, "-m", "tools.run_render", "tests/fixtures/hardware-add", "--target",
                                      "fp16-hardware-v1", "--compare-softpipe", "--max-channel-error", "1",
                                      "--output", str(output / "render-hardware-add")], 90),
            ("gallium-host-flat", [python, "-m", "tools.run_render", "shaders/01_flat_color", "--target",
                                   "fp16-hardware-v1", "--compare-softpipe", "--max-channel-error", "0",
                                   "--output", str(output / "render-flat-hardware")], 90),
            ("gallium-host-gradient", [python, "-m", "tools.run_render", "tests/fixtures/host-gradient",
                                       "--compare-softpipe", "--max-channel-error", "0",
                                       "--output", str(output / "render-gradient")], 90),
            ("gallium-host-uniform", [python, "-m", "tools.run_render", "shaders/03_uniform", "--compare-softpipe",
                                      "--max-channel-error", "1", "--output", str(output / "render-uniform")], 90),
            ("gallium-host-phong", [python, "-m", "tools.run_render", "shaders/07_lighting_phong", "--compare-softpipe",
                                    "--max-channel-error", "8", "--output", str(output / "render-phong")], 90),
            ("gallium-host-fragment-commit", [python, "-m", "tools.fragment_acceptance", "--build", str(build),
                                             "--output", str(output / "fragment-commit")], 120),
            ("gallium-host-texture", [python, "-m", "tools.texture_acceptance", "--build", str(build),
                                      "--output", str(output / "host-texture")], 120),
            ("gallium-host-texture-flow", [python, "-m", "tools.texture_flow_acceptance", "--build", str(build),
                                           "--output", str(output / "host-texture-flow")], 120),
            ("gallium-host-linear-texture", [python, "-m", "tools.linear_texture_acceptance", "--build", str(build),
                                             "--output", str(output / "linear-texture")], 180),
            ("gallium-host-repeat-texture", [python, "-m", "tools.repeat_texture_acceptance", "--build", str(build),
                                             "--output", str(output / "repeat-texture")], 180),
            ("gallium-host-mirrored-texture", [python, "-m", "tools.mirrored_texture_acceptance", "--build", str(build),
                                               "--output", str(output / "mirrored-texture")], 180),
            ("gallium-host-mixed-texture", [python, "-m", "tools.mixed_texture_acceptance", "--build", str(build),
                                            "--output", str(output / "mixed-texture")], 180),
            ("gallium-host-mipmap-texture", [python, "-m", "tools.mipmap_texture_acceptance", "--build", str(build),
                                             "--output", str(output / "mipmap-texture")], 180),
            ("gallium-host-blend", [python, "-m", "tools.blend_acceptance", "--build", str(build),
                                    "--output", str(output / "host-blend")], 120),
            ("gallium-host-blend-equations", [python, "-m", "tools.blend_equation_acceptance", "--build", str(build),
                                             "--output", str(output / "host-blend-equations")], 180),
            ("gallium-host-instancing", [python, "-m", "tools.instancing_acceptance", "--build", str(build),
                                         "--output", str(output / "host-instancing")], 180),
            ("gallium-host-features", [python, "-m", "tools.host_features_acceptance", "--build", str(build),
                                        "--output", str(output / "host-features")], 180),
            ("gallium-host-anisotropy", [python, "-m", "tools.host_features_acceptance", "--build", str(build),
                                          "--suite", "anisotropy-v14",
                                          "--output", str(output / "host-anisotropy")], 180),
            ("gallium-native-host-readback", [python, "-m", "tools.native_host_acceptance", "--build", str(build),
                                              "--output", str(output / "native-host")], 180),
            ("teapot-mesh", [python, "-m", "tools.teapot_mesh", "--output", str(output / "teapot-vertices.bin"),
                             "--metadata", str(output / "teapot-mesh.json"), "--subdivisions", "3"], 60),
            ("mesa-teapot-host", [python, "-m", "tools.run_render", "shaders/07_lighting_phong",
                                  "--mesh", str(output / "teapot-vertices.bin"), "--compare-softpipe",
                                  "--max-channel-error", "16", "--output", str(output / "render-teapot")], 120),
            ("managed-bucky", [python, "-m", "tools.run_hardware_scene", "--scene", "bucky12", "--output",
                                str(output / "driver-bucky")], 90),
            ("managed-utah", [python, "-m", "tools.run_hardware_scene", "--scene", "utah",
                               "--allow-approximate-fp16", "--output", str(output / "driver-utah")], 90),
            ("utah-reference-demo", [python, "-m", "tools.reference_demo", "--scene", "utah", "--frames", "all",
                                      "--output", str(output / "reference-utah")], 90),
            ("host-benchmark", [python, "-m", "tools.benchmark_reference", "--warmup", "2", "--repeats", "5",
                                 "--output", str(output / "host-benchmark.json")], 90),
        ])
    report = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "host tests/reference models only; no physical FPGA access or GPU performance claims",
        "mesa_rebuilt": False, "build_reuse_requested": args.no_build, "stages": [],
    }
    before = source_hashes(ROOT)
    (output / "validated-source-hashes.json").write_text(json.dumps(before, indent=2, sort_keys=True) + "\n")
    failed = False
    for name, command, timeout in stages:
        started = time.perf_counter()
        print(f"RUN {name}", flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                       text=True, timeout=timeout, check=False)
            status = completed.returncode
            log = completed.stdout + completed.stderr
        except subprocess.TimeoutExpired as error:
            status = 124
            log = timeout_log(error, timeout)
        (output / f"{name}.log").write_text(log)
        if name == "build":
            report["mesa_rebuilt"] = status == 0
        if name == "tests":
            report["tests"] = {label: int(match.group(1)) if match else 0
                               for label in ("passed", "failed", "skipped", "errors")
                               for match in [re.search(rf"(\d+) {label.rstrip('s')}s?\b", log)]}
        report["stages"].append({"name": name, "command": command, "exit_code": status,
                                  "seconds": time.perf_counter() - started,
                                  "log": str(output / f"{name}.log")})
        failed |= status != 0
        print(f"{'PASS' if status == 0 else 'FAIL'} {name}; see {output / (name + '.log')}", flush=True)
        if name == "build" and status:
            break
    after = source_hashes(ROOT)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    report["source_freeze"] = {
        "unchanged": not changed, "files_checked": len(before), "changed_paths": changed,
        "manifest": str(output / "validated-source-hashes.json"),
        "scope": "host tools/tests/GLSL/ISA contracts, MGC source and changed softpipe source; not all Mesa dependencies",
    }
    if changed:
        failed = True
        print(f"FAIL source freeze; changed: {', '.join(changed)}", flush=True)
    report["all_passed"] = not failed
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Project validation", "", report["scope"], "",
             f"Run UTC: {report['started_utc']}", f"All executed stages passed: {not failed}", "",
             "| Stage | Result | Log |", "|---|---|---|"]
    if "tests" in report:
        lines[6:6] = [f"Tests: {report['tests']['passed']} passed, {report['tests']['failed']} failed, "
                      f"{report['tests']['skipped']} skipped, {report['tests']['errors']} errors.", ""]
    lines.extend(f"| {entry['name']} | {'PASS' if entry['exit_code'] == 0 else 'FAIL'} | "
                 f"{Path(entry['log']).name} |" for entry in report["stages"])
    lines.extend(("", f"Validated source hashes unchanged: {not changed}; {len(before)} files checked."))
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
