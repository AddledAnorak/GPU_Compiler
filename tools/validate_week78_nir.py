#!/usr/bin/env python3
"""Check the Week 7-8 Phong and texture claims against fresh softpipe NIR dumps.

Run tools/dump_nir.sh first. Its full dumps are intentionally Git-ignored;
this checker makes the small, documented observations reproducible.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACES = ROOT / "docs/glsl-to-nir/traces"
STAGE = re.compile(r"^shader: MESA_SHADER_(VERTEX|FRAGMENT)$", re.MULTILINE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def snapshots(text: str, stage: str) -> list[str]:
    markers = list(STAGE.finditer(text))
    result = []
    for i, marker in enumerate(markers):
        if marker.group(1) == stage:
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            result.append(text[marker.start():end])
    require(result, f"no {stage.lower()} NIR snapshots")
    return result


def has(text: str, pattern: str, description: str) -> None:
    require(re.search(pattern, text, re.MULTILINE) is not None,
            f"missing {description}")


def count(text: str, op: str) -> int:
    return len(re.findall(r"= " + re.escape(op) + r"\b", text))


def validate_phong(text: str) -> str:
    vertex = snapshots(text, "VERTEX")[-1]
    fragments = snapshots(text, "FRAGMENT")
    fragment = fragments[-1]

    for name in ("a_position", "a_normal"):
        has(vertex, rf"@load_input[^\n]*// {name}$", f"vertex input {name}")
    has(vertex, r"@load_ubo_vec4", "vertex matrix loads")
    for name in ("gl_Position", "VARYING_SLOT_VAR0", "VARYING_SLOT_VAR1"):
        has(vertex, rf"@store_output[^\n]*{name}", f"vertex output {name}")

    for op, expected in (("fdot3_replicated", 6), ("frsq", 3),
                         ("fmax", 2), ("fpow", 1)):
        actual = count(fragment, op)
        require(actual == expected, f"final fragment has {actual} {op}, expected {expected}")
    require(count(fragment, "ffma") == 2, "expected reflect and final-color ffma")
    require(len(re.findall(r"@load_ubo_vec4", fragment)) == 3,
            "expected light, view, and object-color uniform loads")
    for name in ("VARYING_SLOT_VAR0", "VARYING_SLOT_VAR1"):
        has(fragment, rf"@load_interpolated_input[^\n]*{name}",
            f"interpolated {name}")
    has(fragment, r"= fpow[^\n]*32\.000000", "specular exponent 32")
    has(fragment, r"@store_output[^\n]*FRAG_RESULT_COLOR", "fragment color output")
    has(fragments[0], r"= fdot3\b", "initial fragment dot product")

    return "Phong: vertex matrices/packed varyings -> interpolated light math -> color; 3 frsq, 1 fpow"


def validate_texture(text: str) -> str:
    vertex = snapshots(text, "VERTEX")[-1]
    fragments = snapshots(text, "FRAGMENT")
    fragment = fragments[-1]

    for name in ("a_position", "a_texcoord"):
        has(vertex, rf"@load_input[^\n]*// {name}$", f"vertex input {name}")
    has(vertex, r"@load_ubo_vec4", "MVP matrix load")
    has(vertex, r"@store_output[^\n]*gl_Position", "position output")
    has(vertex, r"@store_output[^\n]*v_texcoord", "texture-coordinate output")
    has(fragments[0], r"\btex\b[^\n]*sampler_deref_intrinsic",
        "initial texture instruction with sampler reference")

    coord = re.search(
        r"^\s*32x2\s+(%\d+) = @load_interpolated_input[^\n]*VARYING_SLOT_VAR0",
        fragment, re.MULTILINE)
    tex = re.search(
        r"^\s*32x4\s+(%\d+) = \(float32\)tex\s+(%\d+) \(backend1\), "
        r"0 \(texture\), 0 \(sampler\)", fragment, re.MULTILINE)
    store = re.search(r"@store_output \((%\d+),[^\n]*FRAG_RESULT_COLOR", fragment)
    require(coord is not None and tex is not None and store is not None,
            "missing final fragment coordinate, texture sample, or color store")
    require(len(re.findall(r"= \(float32\)tex\b", fragment)) == 1,
            "expected exactly one texture instruction")
    require(coord.group(1) == tex.group(2) and tex.group(1) == store.group(1),
            "texture coordinate -> sample -> color SSA chain is broken")

    return "Texture: vertex UV -> interpolated UV -> one tex(texture 0, sampler 0) -> color"


def main(trace_dir: Path) -> int:
    for name, validate in (("07_lighting_phong", validate_phong),
                           ("08_texture", validate_texture)):
        path = trace_dir / f"{name}.nir.txt"
        if not path.is_file():
            print(f"FAIL {name}: missing {path}; run tools/dump_nir.sh", file=sys.stderr)
            return 1
        text = path.read_text(errors="replace")
        try:
            require("shader compile failed" not in text and "link failed" not in text,
                    "GLSL compilation or linking failed")
            print(f"PASS {name}: {validate(text)}")
        except ValueError as error:
            print(f"FAIL {name}: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_TRACES))
