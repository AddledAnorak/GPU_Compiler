#!/usr/bin/env python3
"""Regenerate and validate GLSL -> NIR -> TGSI traces, never cached results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .validate_week78_nir import has, require, validate_phong, validate_texture
else:
    from validate_week78_nir import has, require, validate_phong, validate_texture


ROOT = Path(__file__).resolve().parents[1]
STAGE = re.compile(r"^shader: MESA_SHADER_(VERTEX|FRAGMENT)$", re.MULTILINE)
DIAGNOSTIC = re.compile(
    r"shader\s+compile\s+failed|compilation\s+failed|link(?:ing)?\s+failed|"
    r"\berror:|assertion[^\n]*failed|segmentation fault|frontend timeout",
    re.IGNORECASE,
)
ALU = re.compile(r"^\s*\d+(?:x\d+)?\s+%\d+ = ([a-z]\w*)\b", re.MULTILINE)


@dataclass(frozen=True)
class Snapshot:
    stage: str
    label: str
    line: int
    text: str


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_snapshots(text: str) -> list[Snapshot]:
    markers = list(STAGE.finditer(text))
    result = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        section = text[marker.start():end].split("TGSI after translation from NIR:", 1)[0]
        closing = list(re.finditer(r"^}$", section, re.MULTILINE))
        require(bool(closing), f"incomplete {marker[1]} snapshot")
        section = section[:closing[-1].end()]
        require("impl main {" in section, f"missing {marker[1]} main implementation")
        require(section.count("{") == section.count("}"),
                f"unbalanced {marker[1]} snapshot")
        prefix = text[:marker.start()].rstrip()
        label = prefix.rsplit("\n", 1)[-1] if prefix else ""
        result.append(Snapshot(marker[1], label, text.count("\n", 0, marker.start()) + 1,
                               section))
    return result


def selected(snapshots: list[Snapshot], stage: str, label: str) -> Snapshot:
    matches = [snapshot for snapshot in snapshots
               if snapshot.stage == stage and snapshot.label == label]
    require(bool(matches), f"missing {stage.lower()} {label} snapshot")
    return matches[-1]


def operations(text: str) -> dict[str, int]:
    return dict(sorted(Counter(ALU.findall(text)).items()))


def tgsi_program(text: str, stage: str) -> str:
    header = "VERT" if stage == "VERTEX" else "FRAG"
    matches = list(re.finditer(
        rf"^TGSI after translation from NIR:\n{header}\n", text, re.MULTILINE))
    require(bool(matches), f"missing {header} TGSI translation")
    tail = text[matches[-1].end():]
    tail = re.split(r"^(?:shader:|glsl_to_nir|NIR before translation to TGSI:|"
                    r"TGSI after translation from NIR:)", tail, maxsplit=1,
                    flags=re.MULTILINE)[0]
    ending = re.search(r"^\s*\d+:\s+END\s*$", tail, re.MULTILINE)
    require(ending is not None, f"incomplete {header} TGSI program")
    return tail[:ending.end()]


def validate_constructs(name: str, initial: dict[str, str], final: dict[str, str],
                        tgsi: dict[str, str]) -> str:
    fragment = final["FRAGMENT"]
    vertex = final["VERTEX"]
    early = initial["FRAGMENT"]
    if name == "00_arith":
        for opcode in ("fmul", "fadd", "fdiv"):
            has(early, rf"= {opcode}\b", f"initial {opcode}")
        has(fragment, r"3\.250000", "folded arithmetic result 3.25")
        require(not any(opcode.startswith("f") for opcode in operations(fragment)),
                "literal arithmetic was not folded")
        return "literal arithmetic folds to 3.25"
    if name == "01_flat_color":
        has(fragment, r"\(1\.000000, 0\.000000, 0\.000000, 1\.000000\)",
            "red constant")
        require(operations(fragment) == {"load_const": 2}, "unexpected flat-color ALU")
        return "constant red -> color output"
    if name in ("02_varying", "03_uniform"):
        has(vertex, r"@load_input[^\n]*a_color", "vertex color input")
        has(fragment, r"@load_interpolated_input", "interpolated color")
        has(fragment, r"@load_barycentric_pixel", "barycentric input")
        if name == "03_uniform":
            has(fragment, r"@load_ubo_vec4", "tint uniform")
            has(fragment, r"= fmul\b", "tint multiply")
        return "varying interpolation" + (" and uniform tint" if name == "03_uniform" else "")
    if name == "04_conditional":
        has(early, r"^\s*if %", "initial structured if")
        has(fragment, r"= flt32\b", "threshold comparison")
        has(fragment, r"= b32csel\b", "color select")
        require(not re.search(r"^\s*if %", fragment, re.MULTILINE), "if survived select lowering")
        has(vertex, r"= ffma\b", "luma moved to vertex")
        return "value-only if -> select; luma moved to vertex"
    if name == "05_loop":
        has(early, r"^\s*loop \{", "initial fixed loop")
        require(not re.search(r"^\s*loop \{", fragment, re.MULTILINE), "fixed loop survives")
        has(fragment, r"2\.083333", "folded harmonic coefficient")
        has(fragment, r"= fmul\b", "live varying multiply")
        has(fragment, r"@load_interpolated_input", "live loop input")
        return "fixed loop disappears; varying * 2.083333 remains"
    if name == "06_builtin_math":
        for opcode in ("fsin", "frsq", "fdot3_replicated", "ffma"):
            has(fragment, rf"= {opcode}\b", opcode)
        return "sin plus dot/rsqrt normalization"
    if name == "07_lighting_phong":
        return validate_phong(fragment_trace(initial, final))
    if name == "08_texture":
        has(tgsi["FRAGMENT"], r"^\s*\d+:\s+TEX\b", "TGSI texture sampling")
        return validate_texture(fragment_trace(initial, final))
    if name == "09_runtime_loop":
        for pattern, description in ((r"^\s*loop \{", "runtime loop"),
                                     (r"^\s*break$", "loop break"),
                                     (r"= ige32\b", "integer bound comparison"),
                                     (r"= iadd\b", "loop increment"),
                                     (r"= i2f32\b", "integer-to-float conversion"),
                                     (r"= ffma\b", "live accumulation")):
            has(fragment, pattern, description)
        has(tgsi["FRAGMENT"], r"\bBGNLOOP\b", "TGSI loop")
        has(tgsi["FRAGMENT"], r"\bBRK\b", "TGSI break")
        return "runtime loop/break survive -> TGSI BGNLOOP/BRK"
    if name == "10_branch_discard":
        has(early, r"^\s*if %", "initial discard branch")
        has(fragment, r"@terminate_if", "conditional fragment termination")
        has(tgsi["FRAGMENT"], r"\bKILL_IF\b", "TGSI fragment kill")
        return "side-effecting if -> terminate_if -> KILL_IF"
    if name == "11_transcendentals":
        for opcode in ("fsin", "fcos", "frcp", "ffma", "iand", "ior", "b32csel"):
            has(fragment, rf"= {opcode}\b", opcode)
        require(not any("atan" in opcode for opcode in operations(early)),
                "unexpected primitive atan in initial NIR")
        has(vertex, r"@load_ubo_vec4", "phase uniform moved to vertex")
        has(vertex, r"= fadd\b", "phase addition moved to vertex")
        return "sin/cos remain; atan expanded; phase addition moved to vertex"
    if name == "12_runtime_arithmetic":
        has(early, r"= fdiv\b", "live initial division")
        has(fragment, r"= fdiv\b", "live final division")
        has(tgsi["FRAGMENT"], r"^\s*\d+:\s+DIV\b", "TGSI division")
        has(vertex + fragment, r"= fadd\b", "live addition")
        return "live fdiv -> TGSI DIV; additions moved to vertex"
    return "stage outputs and complete NIR/TGSI checked; no construct-specific contract"


def fragment_trace(initial: dict[str, str], final: dict[str, str]) -> str:
    return "\n".join(initial[stage] for stage in ("VERTEX", "FRAGMENT")) + "\n" + "\n".join(
        final[stage] for stage in ("VERTEX", "FRAGMENT"))


def inspect_trace(name: str, text: str, returncode: int | None) -> dict:
    require(returncode == 0, f"harness exited {returncode}")
    diagnostic = DIAGNOSTIC.search(text)
    require(diagnostic is None, f"compiler/runtime diagnostic: {diagnostic[0] if diagnostic else ''}")
    snapshots = parse_snapshots(text)
    initial = {}
    final = {}
    evidence = {}
    tgsi = {}
    for stage in ("VERTEX", "FRAGMENT"):
        first = selected(snapshots, stage, "glsl_to_nir")
        last = selected(snapshots, stage, "NIR before translation to TGSI:")
        initial[stage] = first.text
        final[stage] = last.text
        tgsi[stage] = tgsi_program(text, stage)
        output = "VARYING_SLOT_POS|gl_Position" if stage == "VERTEX" else "FRAG_RESULT_COLOR"
        has(last.text, rf"@store_output[^\n]*(?:{output})", f"{stage.lower()} stage output")
        evidence[stage] = {
            "snapshots": sum(snapshot.stage == stage for snapshot in snapshots),
            "initial_line": first.line,
            "final_line": last.line,
            "initial_ops": operations(first.text),
            "final_ops": operations(last.text),
            "final_intrinsics": dict(sorted(Counter(re.findall(r"@(\w+)", last.text)).items())),
            "loops": len(re.findall(r"^\s*loop \{", last.text, re.MULTILINE)),
            "ifs": len(re.findall(r"^\s*if %", last.text, re.MULTILINE)),
            "textures": len(re.findall(r"= \(float32\)tex\b", last.text)),
        }
    return {"evidence": evidence, "observation": validate_constructs(name, initial, final, tgsi)}


def capture(command: list[str], environment: dict[str, str], timeout: float) -> tuple[int | None, str]:
    try:
        result = subprocess.run(command, env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return result.returncode, result.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        return None, output.decode("utf-8", errors="replace") + "\nfrontend timeout\n"
    except OSError as error:
        return None, f"harness launch failed: {error}\n"


def environment(build: Path) -> dict[str, str]:
    result = dict(os.environ)
    result["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + result.get("PATH", "")
    libraries = [build / "src/egl", build / "src/mapi/es2api", build / "src/gallium/targets/dri"]
    result["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries)) + os.pathsep + result.get(
        "LD_LIBRARY_PATH", "")
    result.update({"NIR_DEBUG": "print,tgsi", "MESA_SHADER_CACHE_DISABLE": "true",
                   "GALLIUM_DRIVER": "softpipe", "EGL_PLATFORM": "surfaceless",
                   "LIBGL_ALWAYS_SOFTWARE": "1", "MESA_LOADER_DRIVER_OVERRIDE": "swrast",
                   "LIBGL_DRIVERS_PATH": str(libraries[-1])})
    return result


def build_harness(build: Path, destination: Path, env: dict[str, str]) -> dict:
    libraries = [build / "src/egl/libEGL.so.1", build / "src/mapi/es2api/libGLESv2.so.2"]
    gallium = sorted((build / "src/gallium/targets/dri").glob("libgallium*.so"))
    require(all(path.is_file() for path in libraries) and bool(gallium),
            f"missing local EGL/GLES/Gallium libraries in {build}; run PATH=\"$PWD/.venv/bin:$PATH\" "
            ".venv/bin/ninja -C mesa/build")
    compiler = shutil.which("gcc", path=env["PATH"])
    pkg_config = shutil.which("pkg-config", path=env["PATH"])
    require(compiler is not None and pkg_config is not None, "gcc and pkg-config are required")
    pkg_env = dict(env, PKG_CONFIG_PATH=str(build / "meson-uninstalled"))
    require((build / "meson-uninstalled/egl-uninstalled.pc").is_file(),
            "missing local Mesa uninstalled pkg-config metadata")
    flags = subprocess.run([pkg_config, "--cflags", "--libs", "egl", "glesv2"],
                           env=pkg_env, capture_output=True, text=True, timeout=30, check=False)
    require(flags.returncode == 0, f"pkg-config failed: {flags.stderr.strip()}")
    command = [compiler, str(ROOT / "tools/nir_dump_harness.c"), "-o", str(destination),
               *shlex.split(flags.stdout)]
    compiled = subprocess.run(command, env=env, capture_output=True, text=True,
                              timeout=30, check=False)
    require(compiled.returncode == 0, f"harness build failed: {compiled.stderr.strip()}")
    ldd = subprocess.run(["ldd", str(destination)], env=env, capture_output=True, text=True,
                         timeout=30, check=False)
    require(ldd.returncode == 0, f"ldd failed: {ldd.stderr.strip()}")
    for library in libraries:
        match = re.search(rf"{re.escape(library.name)} => (\S+)", ldd.stdout)
        require(match is not None and Path(match[1]).resolve() == library.resolve(),
                f"harness does not resolve local {library.name}")
    metadata = json.loads((build / "meson-info/meson-info.json").read_text())
    mesa = Path(metadata["directories"]["source"])
    revision = subprocess.run(["git", "-C", str(mesa), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30, check=False)
    status = subprocess.run(["git", "-C", str(mesa), "status", "--short"],
                            capture_output=True, text=True, timeout=30, check=False)
    options = json.loads((build / "meson-info/intro-buildoptions.json").read_text())
    tracked_options = {"buildtype", "b_ndebug", "gallium-drivers", "llvm", "platforms", "glx", "egl"}
    return {"command": shlex.join(command), "ldd": ldd.stdout.strip(),
            "harness_source_sha256": digest(ROOT / "tools/nir_dump_harness.c"),
            "libraries": {str(path.resolve()): digest(path) for path in libraries + gallium},
            "mesa_source_root": str(mesa), "mesa_version": (mesa / "VERSION").read_text().strip(),
            "mesa_revision": revision.stdout.strip() if revision.returncode == 0 else "unknown",
            "mesa_git_status": status.stdout.strip() if status.returncode == 0 else "unknown",
            "mesa_build_options": {option["name"]: option["value"] for option in options
                                   if option["name"] in tracked_options},
            "source_binary_note": "Checkout revision/status are not a claim that Mesa was rebuilt; "
                                  "the actual libraries are independently hashed."}


def write_reports(output: Path, manifest: dict) -> None:
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = ["# Fresh frontend validation", "", f"Run (UTC): {manifest['started_utc']}", "",
             f"Successful validated pairs: **{manifest['successful']}/{manifest['discovered']}**.",
             f"Fresh harness attempts: {manifest['attempted']}. Failed pairs: {manifest['failed']}.",
             "", "Scope: GLSL compile/link and NIR-to-TGSI translation, not pixels or FPGA execution.",
             "Durations include context setup, compilation, linking, draw/glFinish and debug printing;",
             "they are not shader execution time, throughput, GPU utilization or fps.", "",
             "| Pair | Result | VS/FS snapshots | Initial VS/FS lines | Final VS/FS lines | Observation |",
             "|---|---|---|---|---|---|"]
    for result in manifest["results"]:
        evidence = result.get("evidence", {})
        def stages(field: str) -> str:
            return "/".join(str(evidence[stage][field]) for stage in ("VERTEX", "FRAGMENT")) if evidence else "—"
        note = result.get("observation", result.get("error", ""))
        lines.append(f"| {result['name']} | {'PASS' if result['ok'] else 'FAIL'} | "
                     f"{stages('snapshots')} | {stages('initial_line')} | {stages('final_line')} | {note} |")
    if "error" in manifest:
        lines.extend(["", "Setup error: " + manifest["error"]])
    lines.extend(["", "See `manifest.json` for commands, exit codes, timings, hashes and per-stage opcode counts.",
                  "Traces and reports are overwritten from a new run; existing files are never treated as success."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "mesa/build")
    parser.add_argument("--shader-dir", type=Path, default=ROOT / "shaders")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "docs/glsl-to-nir/traces")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not args.timeout > 0 or not args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    build, shaders, output = (path.resolve() for path in (args.build_dir, args.shader_dir, args.trace_dir))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                "build_dir": str(build), "shader_dir": str(shaders), "driver": "softpipe",
                "nir_debug": "print,tgsi", "discovered": 0, "attempted": 0,
                "successful": 0, "failed": 0, "results": []}
    try:
        require(shaders.is_dir(), f"shader directory not found: {shaders}")
        pairs = sorted(path for path in shaders.iterdir() if path.is_dir() and not path.name.startswith("."))
        manifest["discovered"] = len(pairs)
        require(bool(pairs), "no shader pairs discovered")
        for pair in pairs:
            (output / f"{pair.name}.nir.txt").unlink(missing_ok=True)
        env = environment(build)
        manifest["environment"] = {key: env[key] for key in (
            "NIR_DEBUG", "MESA_SHADER_CACHE_DISABLE", "GALLIUM_DRIVER", "EGL_PLATFORM",
            "LIBGL_ALWAYS_SOFTWARE", "MESA_LOADER_DRIVER_OVERRIDE", "LIBGL_DRIVERS_PATH", "LD_LIBRARY_PATH")}
        with tempfile.TemporaryDirectory(prefix=".frontend-", dir=output) as work:
            harness = Path(work) / "nir_dump_harness"
            manifest["build"] = build_harness(build, harness, env)
            manifest["mesa_revision"] = manifest["build"].get("mesa_revision", "unknown")
            manifest["mesa_version"] = manifest["build"].get("mesa_version", "unknown")
            for pair in pairs:
                result = {"name": pair.name, "ok": False}
                path = output / f"{pair.name}.nir.txt"
                try:
                    sources = [pair / "vert.glsl", pair / "frag.glsl"]
                    require(all(source.is_file() for source in sources), "missing vert.glsl or frag.glsl")
                    result["source_sha256"] = {str(source): digest(source) for source in sources}
                    command = [str(harness), *map(str, sources)]
                    result["command"] = shlex.join(command)
                    manifest["attempted"] += 1
                    started = time.perf_counter()
                    returncode, text = capture(command, env, args.timeout)
                    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
                    result["returncode"] = returncode
                    path.write_text(text)
                    result["trace_sha256"] = digest(path)
                    result.update(inspect_trace(pair.name, text, returncode))
                    require(all(digest(source) == result["source_sha256"][str(source)] for source in sources),
                            "shader source changed during capture")
                    result["ok"] = True
                    manifest["successful"] += 1
                    print(f"PASS {pair.name}: {result['observation']}")
                except (ValueError, OSError) as error:
                    result["error"] = str(error)
                    print(f"FAIL {pair.name}: {error}", file=sys.stderr)
                manifest["results"].append(result)
            library_hashes = manifest["build"].get("libraries", {})
            if any(not Path(path).is_file() or digest(Path(path)) != expected
                   for path, expected in library_hashes.items()):
                for result in manifest["results"]:
                    result["ok"] = False
                    result["error"] = "Mesa libraries changed during capture; rerun validation"
                manifest["successful"] = 0
                raise ValueError("Mesa libraries changed during capture; all results invalidated")
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        manifest["error"] = str(error)
        print(f"FAIL setup: {error}", file=sys.stderr)
    manifest["failed"] = manifest["discovered"] - manifest["successful"]
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_reports(output, manifest)
    print(f"Successful fresh traces: {manifest['successful']}/{manifest['discovered']}; "
          f"attempted {manifest['attempted']}; failed {manifest['failed']}")
    print(f"Evidence: {output / 'summary.md'}")
    return int("error" in manifest or manifest["failed"] > 0 or not manifest["successful"])


if __name__ == "__main__":
    raise SystemExit(main())
