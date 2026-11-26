"""Capture an actual Gallium draw and render it with managed CPU shader execution."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

from tools.host_blend import BLEND_MODES
from tools.host_render import render_artifact
from tools.mgcreference.artifacts import write_png
from tools.mgcdriver import ProgramFormat
from tools.run_shader import ROOT, mesa_environment


INSTANCE_HEADER = struct.Struct("<8s5I")
INSTANCE_MAGIC = b"MGCINS01"
INSTANCE_MAX_BYTES = INSTANCE_HEADER.size + 16 * 8 * 4
INSTANCING_ENV = ("MGC_HOST_INSTANCING", "MGC_HOST_INSTANCE_DATA", "MGC_HOST_INSTANCE_COUNT",
                  "MGC_HOST_INSTANCE_OFFSET_DIVISOR", "MGC_HOST_INSTANCE_COLOR_DIVISOR", "MGC_HOST_BASE_INSTANCE")


def _validate_instance_data(instance_data):
    """MGCINS01 + LE u32 count/S-offset divisor/color divisor/vec widths, then LE FP32 records.

    Both divisors are nonzero. Each stream has ceil(count/divisor) tightly packed vec3/vec4
    records; offset records precede color records. The GL path always has base instance zero.
    """
    if instance_data is None:
        return None
    if not isinstance(instance_data, Path):
        raise ValueError("instance data must be an explicit Path")
    descriptor = os.open(instance_data, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        storage = os.fstat(source.fileno())
        if not stat.S_ISREG(storage.st_mode) or not INSTANCE_HEADER.size < storage.st_size <= INSTANCE_MAX_BYTES:
            raise ValueError("instance data must be an ordinary bounded binary file")
        contents = source.read(INSTANCE_MAX_BYTES + 1)
    if len(contents) != storage.st_size:
        raise ValueError("instance data changed size during bounded read")
    magic, count, offset_divisor, color_divisor, offset_components, color_components = INSTANCE_HEADER.unpack_from(contents)
    if (magic != INSTANCE_MAGIC or not 1 <= count <= 16 or
            not 1 <= offset_divisor <= 1048576 or not 1 <= color_divisor <= 1048576 or
            offset_components not in (3, 4) or color_components not in (3, 4)):
        raise ValueError("instance data requires MGCINS01, count 1..16, divisors 1..1048576 and vec3/vec4 records")
    scalar_count = ((count - 1) // offset_divisor + 1) * offset_components + ((count - 1) // color_divisor + 1) * color_components
    if len(contents) != INSTANCE_HEADER.size + scalar_count * 4:
        raise ValueError("instance data must contain exactly the divisor-indexed FP32 records")
    if not np.all(np.isfinite(np.frombuffer(contents, dtype="<f4", offset=INSTANCE_HEADER.size))):
        raise ValueError("instance data records must be finite native FP32 values")
    return {"instance_count": count, "offset_divisor": offset_divisor, "color_divisor": color_divisor,
            "offset_components": offset_components, "color_components": color_components, "start_instance": 0}


def _configure_instancing(environment, instance_data):
    if instance_data is not None:
        environment["MGC_HOST_INSTANCING"] = "1"
        environment["MGC_HOST_INSTANCE_DATA"] = str(instance_data.absolute())


def _validate_clear_color(clear_color):
    if (type(clear_color) is not tuple or len(clear_color) != 4 or
            any(type(channel) not in (int, float) or not 0 <= channel <= 1 or
                not math.isfinite(channel) for channel in clear_color)):
        raise ValueError("clear color must be an exact four-component tuple of finite int/float values in [0, 1]")


def _configure_clear_color(environment, clear_color):
    if any(clear_color):
        environment["MGC_HOST_CLEAR_RGBA"] = ",".join(format(channel, ".17g") for channel in clear_color)


def _validate_texture_filter(texture, texture_filter):
    if type(texture_filter) is not str or texture_filter not in ("nearest", "linear"):
        raise ValueError("texture filter must be explicitly nearest or linear")
    if texture_filter == "linear" and texture is None:
        raise ValueError("linear texture filter requires an actual texture file")


def _validate_texture_wrap(texture, texture_wrap, texture_wrap_s=None, texture_wrap_t=None):
    wraps = ("clamp-to-edge", "repeat", "mirrored-repeat")
    wrap_s = texture_wrap if texture_wrap_s is None else texture_wrap_s
    wrap_t = texture_wrap if texture_wrap_t is None else texture_wrap_t
    for value in (texture_wrap, wrap_s, wrap_t):
        if type(value) is not str or value not in wraps:
            raise ValueError("texture wrap must be explicitly clamp-to-edge or repeat or mirrored-repeat")
    for value in (wrap_s, wrap_t):
        if value != "clamp-to-edge" and texture is None:
            raise ValueError(f"{value} texture wrap requires an actual texture file")
    return wrap_s, wrap_t


def _validate_mipmaps(texture, mipmaps):
    if type(mipmaps) is not bool:
        raise ValueError("mipmaps mode must be explicitly boolean")
    if mipmaps and texture is None:
        raise ValueError("mipmaps mode requires an actual texture file")


def _host_texture_version(texture_filter, texture_wrap, texture_wrap_t=None, *, mipmaps=False):
    if mipmaps:
        return "10" if texture_filter == "linear" else "9"
    if texture_wrap_t is not None and texture_wrap != texture_wrap_t:
        return "8" if texture_filter == "linear" else "7"
    return str((2 if texture_filter == "linear" else 1) +
               {"clamp-to-edge": 0, "repeat": 2, "mirrored-repeat": 4}[texture_wrap])


def _configure_texture(environment, texture, texture_filter, wrap_s, wrap_t, *, versioned=True, mipmaps=False):
    if texture is None:
        return
    environment["MGC_HOST_TEXTURE_DATA"] = str(Path(texture).absolute())
    environment["MGC_HOST_TEXTURE_FILTER"] = texture_filter
    if versioned or wrap_s != wrap_t or mipmaps:
        environment["MGC_HOST_TEXTURE"] = _host_texture_version(texture_filter, wrap_s, wrap_t, mipmaps=mipmaps)
    if mipmaps:
        environment["MGC_HOST_GLSL_PROFILE"] = "desktop-120"
    if wrap_s == wrap_t and not mipmaps:
        environment["MGC_HOST_TEXTURE_WRAP"] = wrap_s
    else:
        environment["MGC_HOST_TEXTURE_WRAP_S"] = wrap_s
        environment["MGC_HOST_TEXTURE_WRAP_T"] = wrap_t


def capture_draw(shader_directory, build, output, target=ProgramFormat.LEGACY, *, mesh=None,
                 depth_test=False, fragment_status=True, host_texture=False, texture=None,
                 blend="disabled", texture_filter="nearest", texture_wrap="clamp-to-edge",
                 clear_color=(0, 0, 0, 0), texture_wrap_s=None, texture_wrap_t=None, instance_data=None,
                 mipmaps=False, host_features=False):
    target = ProgramFormat(target)
    if type(host_features) is not bool:
        raise ValueError("host features must be explicitly boolean")
    if host_features and (target != ProgramFormat.LEGACY or not fragment_status):
        raise ValueError("ABI4 host features require legacy compilation and explicit fragment status")
    if host_features and (host_texture or texture is not None or instance_data is not None):
        raise ValueError("ABI4 extended textures/instancing require the explicit host_features_acceptance frontend")
    if not isinstance(depth_test, bool):
        raise ValueError("depth mode must be explicitly boolean")
    if not isinstance(fragment_status, bool):
        raise ValueError("fragment status mode must be explicitly boolean")
    if type(host_texture) is not bool:
        raise ValueError("host texture mode must be explicitly boolean")
    if (host_texture or texture is not None) and target != ProgramFormat.LEGACY:
        raise ValueError("host texture is unsupported on hardware")
    if instance_data is not None and target != ProgramFormat.LEGACY:
        raise ValueError("host instancing is unsupported on hardware")
    _validate_instance_data(instance_data)
    if host_texture != (texture is not None):
        raise ValueError("host texture mode requires an explicit texture file, and vice versa")
    _validate_texture_filter(texture, texture_filter)
    _validate_mipmaps(texture, mipmaps)
    wrap_s, wrap_t = _validate_texture_wrap(texture, texture_wrap, texture_wrap_s, texture_wrap_t)
    if type(blend) is not str or blend not in BLEND_MODES:
        raise ValueError("unsupported explicit host blend mode")
    if blend != "disabled" and target != ProgramFormat.LEGACY:
        raise ValueError("host blend is unsupported on hardware")
    _validate_clear_color(clear_color)
    with tempfile.TemporaryDirectory(prefix="btp-draw-") as temporary:
        directory = Path(temporary)
        captures = directory / "capture"
        captures.mkdir()
        dump = directory / "shader"
        dump.mkdir()
        environment = mesa_environment(build, dump, target)
        environment["MGC_HOST_CAPTURE_DIR"] = str(captures)
        for name in ("MGC_HOST_MESH", "MGC_HOST_DEPTH_TEST", "MGC_FRAGMENT_STATUS",
                     "MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_DATA", "MGC_HOST_TEXTURE_FILTER",
                     "MGC_HOST_TEXTURE_WRAP", "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T", "MGC_HOST_MIPMAPS",
                     "MGC_HOST_GLSL_PROFILE",
                     "MGC_HOST_BLEND", "MGC_HOST_CLEAR_RGBA",
                     "MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT",
                     "MGC_HOST_DRAW_REPEATS") + INSTANCING_ENV:
            environment.pop(name, None)
        _configure_clear_color(environment, clear_color)
        if host_features:
            environment["MGC_HOST_FEATURES"] = "1"
        _configure_instancing(environment, instance_data)
        if blend != "disabled":
            environment["MGC_HOST_BLEND"] = blend
        _configure_texture(environment, texture, texture_filter, wrap_s, wrap_t, mipmaps=mipmaps)
        if depth_test:
            environment["MGC_HOST_DEPTH_TEST"] = "1"
        if fragment_status and target == ProgramFormat.LEGACY:
            environment["MGC_FRAGMENT_STATUS"] = "1"
        if mesh is not None:
            environment["MGC_HOST_MESH"] = str(Path(mesh).resolve())
        completed = subprocess.run([str(ROOT / "tools/mgc_harness"), str(shader_directory / "vert.glsl"),
                                    str(shader_directory / "frag.glsl"), "--draw"],
                                   env=environment, capture_output=True, text=True, timeout=60)
        output.mkdir(parents=True, exist_ok=True)
        (output / "capture.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise ValueError(f"fresh Gallium draw failed; see {output / 'capture.log'}")
        artifacts = list(captures.glob("*.json"))
        if len(artifacts) != 1:
            raise ValueError("fresh single-draw harness must produce exactly one artifact")
        if artifacts[0].stat().st_size > 64 << 20:
            raise ValueError("capture exceeds bounded wire budget")
        contents = artifacts[0].read_text()
        (output / "draw.json").write_text(contents)
        for path in dump.iterdir():
            if path.is_file() and path.suffix in (".asm", ".json"):
                (output / path.name).write_text(path.read_text())
        return json.loads(contents)


def softpipe_reference(shader_directory, build, output, *, mesh=None, depth_test=False,
                       texture=None, blend="disabled", repeats=1, texture_filter="nearest",
                       texture_wrap="clamp-to-edge", clear_color=(0, 0, 0, 0),
                       texture_wrap_s=None, texture_wrap_t=None, instance_data=None, mipmaps=False):
    _validate_texture_filter(texture, texture_filter)
    _validate_mipmaps(texture, mipmaps)
    wrap_s, wrap_t = _validate_texture_wrap(texture, texture_wrap, texture_wrap_s, texture_wrap_t)
    _validate_instance_data(instance_data)
    if not isinstance(depth_test, bool):
        raise ValueError("depth mode must be explicitly boolean")
    if type(blend) is not str or blend not in BLEND_MODES:
        raise ValueError("unsupported explicit host blend mode")
    _validate_clear_color(clear_color)
    if type(repeats) is not int or not 1 <= repeats <= 4:
        raise ValueError("draw repeats must be an explicit integer in [1, 4]")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "softpipe.rgba8"
    environment = mesa_environment(build, output, ProgramFormat.LEGACY)
    environment["GALLIUM_DRIVER"] = "softpipe"
    for name in ("MGC_HOST_CAPTURE_DIR", "MGC_ASM_DUMP", "MGC_HOST_MESH",
                 "MGC_HOST_DEPTH_TEST", "MGC_FRAGMENT_STATUS", "MGC_HOST_TEXTURE",
                 "MGC_HOST_TEXTURE_DATA", "MGC_HOST_TEXTURE_FILTER", "MGC_HOST_TEXTURE_WRAP",
                 "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T", "MGC_HOST_MIPMAPS", "MGC_HOST_GLSL_PROFILE",
                 "MGC_HOST_BLEND", "MGC_HOST_CLEAR_RGBA", "MGC_HOST_EXECUTOR_SOCKET",
                 "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT", "MGC_HOST_DRAW_REPEATS") + INSTANCING_ENV:
        environment.pop(name, None)
    environment["MGC_HOST_DRAW_REPEATS"] = str(repeats)
    _configure_clear_color(environment, clear_color)
    _configure_instancing(environment, instance_data)
    if blend != "disabled":
        environment["MGC_HOST_BLEND"] = blend
    _configure_texture(environment, texture, texture_filter, wrap_s, wrap_t, versioned=False, mipmaps=mipmaps)
    if depth_test:
        environment["MGC_HOST_DEPTH_TEST"] = "1"
    if mesh is not None:
        environment["MGC_HOST_MESH"] = str(Path(mesh).resolve())
    completed = subprocess.run([str(ROOT / "tools/mgc_harness"), str(shader_directory / "vert.glsl"),
                                str(shader_directory / "frag.glsl"), "--reference", str(path)],
                               env=environment, capture_output=True, text=True, timeout=60)
    (output / "softpipe.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode or not path.is_file() or path.stat().st_size != 16 * 16 * 4:
        raise ValueError(f"matched softpipe reference failed; see {output / 'softpipe.log'}")
    return np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(16, 16, 4)[::-1].copy()


def native_host_readback(shader_directory, build, output, *, socket_path, mesh=None,
                         texture=None, blend="disabled", repeats=1, width=16, height=16,
                         depth_test=False, texture_filter="nearest",
                         texture_wrap="clamp-to-edge", clear_color=(0, 0, 0, 0),
                         texture_wrap_s=None, texture_wrap_t=None, instance_data=None, mipmaps=False,
                         host_features=False) -> np.ndarray:
    if type(host_features) is not bool:
        raise ValueError("host features must be explicitly boolean")
    if host_features and (texture is not None or instance_data is not None):
        raise ValueError("ABI4 extended textures/instancing require the explicit host_features_acceptance frontend")
    _validate_texture_filter(texture, texture_filter)
    _validate_mipmaps(texture, mipmaps)
    wrap_s, wrap_t = _validate_texture_wrap(texture, texture_wrap, texture_wrap_s, texture_wrap_t)
    instance_options = _validate_instance_data(instance_data)
    if type(depth_test) is not bool:
        raise ValueError("native depth mode must be explicitly boolean")
    if type(blend) is not str or blend not in BLEND_MODES:
        raise ValueError("unsupported explicit host blend mode")
    _validate_clear_color(clear_color)
    for name, extent in (("width", width), ("height", height)):
        if type(extent) is not int or not 1 <= extent <= 128:
            raise ValueError(f"native framebuffer {name} must be an explicit integer in [1, 128]")
    if type(repeats) is not int or not 1 <= repeats <= 4:
        raise ValueError("draw repeats must be an explicit integer in [1, 4]")
    socket_path = Path(socket_path)
    socket_bytes = os.fsencode(socket_path)
    if not socket_path.is_absolute() or not socket_bytes or len(socket_bytes) > 107 or b"\0" in socket_bytes:
        raise ValueError("native host executor socket must be an explicit bounded absolute path")
    shader_directory = Path(shader_directory).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "native.rgba8"
    log_path = output / "native.log"
    report_path = output / "native-report.json"
    if raw_path.is_symlink() or (raw_path.exists() and not raw_path.is_file()):
        raise ValueError("native GL output must be an ordinary file")
    raw_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="btp-native-") as temporary:
        directory = Path(temporary)
        captures = directory / "capture"
        captures.mkdir()
        dump = directory / "shader"
        dump.mkdir()
        environment = mesa_environment(Path(build), dump, ProgramFormat.LEGACY)
        for name in ("MGC_HOST_CAPTURE_DIR", "MGC_ASM_DUMP", "MGC_HOST_MESH",
                     "MGC_HOST_DEPTH_TEST", "MGC_FRAGMENT_STATUS", "MGC_HOST_TEXTURE",
                     "MGC_HOST_TEXTURE_DATA", "MGC_HOST_TEXTURE_FILTER", "MGC_HOST_TEXTURE_WRAP",
                     "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T", "MGC_HOST_MIPMAPS", "MGC_HOST_GLSL_PROFILE",
                     "MGC_HOST_BLEND", "MGC_HOST_CLEAR_RGBA", "MGC_HOST_EXECUTOR_SOCKET",
                     "MGC_HOST_WIDTH", "MGC_HOST_HEIGHT", "MGC_HOST_DRAW_REPEATS") + INSTANCING_ENV:
            environment.pop(name, None)
        environment.update({"MGC_HOST_EXECUTOR_SOCKET": str(socket_path),
                            "MGC_HOST_CAPTURE_DIR": str(captures), "MGC_ASM_DUMP": str(dump),
                            "MGC_FRAGMENT_STATUS": "1", "MGC_HOST_WIDTH": str(width),
                            "MGC_HOST_HEIGHT": str(height), "MGC_HOST_DRAW_REPEATS": str(repeats)})
        if host_features:
            environment["MGC_HOST_FEATURES"] = "1"
        _configure_clear_color(environment, clear_color)
        _configure_instancing(environment, instance_data)
        if blend != "disabled":
            environment["MGC_HOST_BLEND"] = blend
        if depth_test:
            environment["MGC_HOST_DEPTH_TEST"] = "1"
        if mesh is not None:
            environment["MGC_HOST_MESH"] = str(Path(mesh).resolve())
        _configure_texture(environment, texture, texture_filter, wrap_s, wrap_t, mipmaps=mipmaps)
        try:
            completed = subprocess.run(
                [str(ROOT / "tools/mgc_harness"), str(shader_directory / "vert.glsl"),
                 str(shader_directory / "frag.glsl"), "--host-execute", str(raw_path)],
                env=environment, capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired as error:
            contents = [part.decode("utf-8", errors="replace") if isinstance(part, bytes) else part or ""
                        for part in (error.stdout, error.stderr)]
            log_path.write_text("".join(contents))
            raise ValueError(f"native GL readback timed out; see {log_path}") from error
        log_path.write_text(completed.stdout + completed.stderr)
        capture_paths = []
        for path in sorted(captures.glob("*.json")):
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 << 20:
                raise ValueError(f"native capture exceeds bounded wire budget; see {log_path}")
            destination = output / path.name
            destination.write_bytes(path.read_bytes())
            capture_paths.append(str(destination))
        for path in dump.iterdir():
            if path.is_file() and path.suffix in (".asm", ".json"):
                (output / path.name).write_bytes(path.read_bytes())
        if completed.returncode or "MGC_HOST_EXECUTOR_FAILURE" in completed.stderr:
            raise ValueError(f"native GL host execution failed; see {log_path}")
        commits = []
        for line in completed.stderr.splitlines():
            if not line.startswith("MGC_HOST_EXECUTOR_COMMIT"):
                continue
            match = re.fullmatch(r"MGC_HOST_EXECUTOR_COMMIT sequence=([1-9][0-9]*) width=([1-9][0-9]*) height=([1-9][0-9]*)", line)
            if match is None:
                raise ValueError(f"malformed native executor commit diagnostic; see {log_path}")
            sequence, committed_width, committed_height = map(int, match.groups())
            if sequence > (1 << 64) - 1 or (committed_width, committed_height) != (width, height):
                raise ValueError(f"native executor commit sequence or dimensions mismatch; see {log_path}")
            commits.append({"sequence": sequence, "width": committed_width, "height": committed_height})
        sequences = [commit["sequence"] for commit in commits]
        if len(commits) != repeats or len(set(sequences)) != repeats:
            raise ValueError(f"native host execution requires exactly {repeats} positive distinct commits; see {log_path}")
        try:
            proof = json.loads(completed.stdout)
            artifacts = [json.loads(Path(path).read_text()) for path in capture_paths]
        except (ValueError, OSError) as error:
            raise ValueError(f"invalid native GL readback or capture JSON; see {log_path}") from error
        expected = {"native_gl_readback": True, "fpga_execution": False, "width": width,
                    "height": height, "format": "R8G8B8A8_UNORM", "pixel_origin": "bottom-left",
                    "repeats": repeats}
        if type(proof) is not dict or proof != expected or any(type(proof[name]) is not type(value) for name, value in expected.items()):
            raise ValueError(f"missing exact native GL readback proof; see {log_path}")
        captured_sequences = [artifact.get("sequence") if type(artifact) is dict else None for artifact in artifacts]
        if any(type(sequence) is not int for sequence in captured_sequences) or sorted(captured_sequences) != sorted(sequences):
            raise ValueError(f"native capture sequences must match accepted commits; see {log_path}")
        paths_by_sequence = dict(zip(captured_sequences, capture_paths))
        capture_paths = [paths_by_sequence[sequence] for sequence in sequences]
        if not raw_path.is_file() or raw_path.stat().st_size != width * height * 4:
            raise ValueError(f"native GL readback must contain exact bounded RGBA8 bytes; see {log_path}")
        image = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8).reshape(height, width, 4)[::-1].copy()
        report = {**expected, "pixel_origin": "top-left", "raw_pixel_origin": "bottom-left",
                  "host_executor_committed": True, "commits": commits, "capture_paths": capture_paths,
                  "raw_path": str(raw_path), "log_path": str(log_path), "blend": blend}
        if depth_test:
            report["depth_test"] = True
        if any(clear_color):
            report["clear_color"] = list(clear_color)
        if texture is not None:
            report["texture_filter"] = texture_filter
            report["texture_wrap"] = wrap_s if wrap_s == wrap_t else "mixed"
            report["texture_wrap_s"] = wrap_s
            report["texture_wrap_t"] = wrap_t
            if mipmaps:
                report["mipmaps"] = True
                report["glsl_profile"] = "desktop-120"
        if instance_options is not None:
            report["instancing"] = {**instance_options, "input_path": str(instance_data.absolute()),
                                    "draw": "actual extension glDrawArraysInstanced; no GL base-instance or indexed-draw claim"}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return image


def compare_images(image, reference, tolerance=8):
    if image.shape != reference.shape or image.dtype != np.uint8 or reference.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4 or not image.shape[0] or not image.shape[1]:
        raise ValueError("reference images must have identical RGBA8 geometry")
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or not 0 <= tolerance <= 255:
        raise ValueError("pixel tolerance must be a channel-byte bound")
    differences = np.abs(image.astype(np.int16) - reference.astype(np.int16))
    coverage_mismatch = int(np.count_nonzero((image[:, :, 3] != 0) != (reference[:, :, 3] != 0)))
    maximum = int(differences.max())
    return {"oracle": "same fresh GLSL, attributes, constants and draw on CPU softpipe",
            "fpga_validation": False, "pixels": image.shape[0] * image.shape[1],
            "max_channel_error": maximum, "mismatch_pixels": int(np.count_nonzero(np.any(differences, axis=2))),
            "alpha_support_mismatch": coverage_mismatch, "channel_tolerance": tolerance,
            "passed": maximum <= tolerance and coverage_mismatch == 0,
            "limitation": "alpha-support is a coverage proxy only for tested positive-alpha shaders; not universal coverage conformance"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader_directory", type=Path, nargs="?")
    parser.add_argument("--capture", type=Path, help="render a saved actual Gallium artifact instead of drawing freshly")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--target", choices=[target.value for target in ProgramFormat], default=ProgramFormat.LEGACY.value)
    parser.add_argument("--host-features", action="store_true", help="explicit legacy ABI4 state/builtin capture; extended sampler fixtures use tools.host_features_acceptance")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--depth-test", action="store_true", help="explicit less-depth mode for the fresh draw, host renderer and reference")
    parser.add_argument("--blend", choices=BLEND_MODES, default="disabled",
                        help="explicit bounded RGBA blend equation for capture, CPU rendering and reference")
    parser.add_argument("--clear-color", type=float, nargs=4, default=(0, 0, 0, 0),
                        metavar=("R", "G", "B", "A"), help="explicit finite RGBA clear components in [0, 1]")
    parser.add_argument("--mesh", type=Path, help="explicit bounded position/normal triangle-list records for the fresh draw")
    parser.add_argument("--instance-data", type=Path,
                        help="explicit MGCINS01 bounded FP32 instance records; advertised GLES2 ANGLE/EXT instancing required")
    parser.add_argument("--host-texture", action="store_true", help="opt into bounded legacy host textures for a fresh draw")
    parser.add_argument("--texture", type=Path, help="little-endian u32 width/height followed by tightly packed RGBA8 texels")
    parser.add_argument("--mipmaps", action="store_true",
                        help="explicit complete MGCMIP1 mipchain and desktop GLSL120 texture2DLod; advertised ARB_shader_texture_lod required")
    parser.add_argument("--texture-filter", choices=("nearest", "linear"), default="nearest",
                        help="actual min/mag filter; clamp v1/v2, repeat v3/v4, mirrored-repeat v5/v6, mixed v7/v8, mipmapped v9/v10")
    parser.add_argument("--texture-wrap", choices=("clamp-to-edge", "repeat", "mirrored-repeat"),
                        default="clamp-to-edge",
                        help="actual texture S/T wrap; periodic modes require a fresh legacy draw and texture input")
    parser.add_argument("--texture-wrap-s", choices=("clamp-to-edge", "repeat", "mirrored-repeat"),
                        help="optional actual S wrap override; defaults to --texture-wrap")
    parser.add_argument("--texture-wrap-t", choices=("clamp-to-edge", "repeat", "mirrored-repeat"),
                        help="optional actual T wrap override; defaults to --texture-wrap")
    parser.add_argument("--compare-softpipe", action="store_true", help="same fresh 16x16 draw on CPU softpipe as pixel oracle")
    parser.add_argument("--max-channel-error", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/host-render")
    args = parser.parse_args(argv)
    try:
        clear_color = tuple(args.clear_color)
        _validate_clear_color(clear_color)
        if bool(args.shader_directory) == bool(args.capture):
            raise ValueError("provide either a shader directory or a saved draw capture")
        if args.capture and (args.mesh or args.compare_softpipe or args.host_texture or args.texture or
                             args.texture_filter != "nearest" or args.texture_wrap != "clamp-to-edge" or
                             args.texture_wrap_s is not None or args.texture_wrap_t is not None or
                             args.instance_data is not None or args.mipmaps):
            raise ValueError("mesh/reference/texture options require a fresh shader draw")
        if args.host_texture and ProgramFormat(args.target) != ProgramFormat.LEGACY:
            raise ValueError("host texture is unsupported on hardware")
        if args.compare_softpipe and (args.width != 16 or args.height != 16):
            raise ValueError("matched reference currently requires 16x16")
        capture_started = time.perf_counter()
        if args.capture:
            if args.capture.stat().st_size > 64 << 20:
                raise ValueError("capture exceeds bounded wire budget")
            artifact = json.loads(args.capture.read_text())
            if artifact.get("program_format") != args.target:
                raise ValueError("saved capture does not match explicitly selected target")
            if args.host_features and artifact.get("schema") != "mgc-host-draw-v4":
                raise ValueError("saved draw does not match explicitly requested ABI4 host features")
        else:
            artifact = capture_draw(args.shader_directory.resolve(), args.build.resolve(), args.output,
                                    ProgramFormat(args.target), mesh=args.mesh, depth_test=args.depth_test,
                                    host_texture=args.host_texture, texture=args.texture, blend=args.blend,
                                    texture_filter=args.texture_filter, texture_wrap=args.texture_wrap,
                                    clear_color=clear_color, texture_wrap_s=args.texture_wrap_s,
                                    texture_wrap_t=args.texture_wrap_t, instance_data=args.instance_data, mipmaps=args.mipmaps,
                                    host_features=args.host_features)
        capture_seconds = time.perf_counter() - capture_started
        render_started = time.perf_counter()
        render_options = {}
        if any(clear_color):
            render_options["initial_image"] = np.broadcast_to(
                np.rint(np.asarray(clear_color) * 255).astype(np.uint8), (args.height, args.width, 4)).copy()
        image, coverage, report = render_artifact(artifact, width=args.width, height=args.height,
                                                depth_test=args.depth_test if artifact.get("schema") != "mgc-host-draw-v4" else False,
                                                blend=args.blend if artifact.get("schema") != "mgc-host-draw-v4" else "disabled",
                                                **render_options)
        if any(clear_color):
            report["clear_color"] = list(clear_color)
        if args.texture is not None:
            wrap_s, wrap_t = _validate_texture_wrap(args.texture, args.texture_wrap,
                                                    args.texture_wrap_s, args.texture_wrap_t)
            report.update(texture_filter=args.texture_filter, texture_wrap_s=wrap_s, texture_wrap_t=wrap_t,
                          texture_wrap=wrap_s if wrap_s == wrap_t else "mixed")
        elif artifact.get("textures"):
            sampler = artifact["textures"][0]["sampler_state" if artifact.get("schema") == "mgc-host-draw-v4" else "sampler"]
            wrap_s, wrap_t = sampler["wrap_s"], sampler["wrap_t"]
            report.update(texture_filter=sampler["mag_filter"], texture_wrap_s=wrap_s, texture_wrap_t=wrap_t,
                          texture_wrap=wrap_s if wrap_s == wrap_t else "mixed")
            if "mipmaps" in artifact["textures"][0]:
                report["mipmaps"] = True
        if args.instance_data is not None:
            report["instancing"] = _validate_instance_data(args.instance_data)
        if args.mipmaps:
            report["mipmaps"] = True
            report["glsl_profile"] = "desktop-120"
        report["capture_wall_seconds"] = capture_seconds
        report["host_render_seconds"] = time.perf_counter() - render_started
        report["timing_scope"] = "CPU wall time only; capture includes compilation/upload/serialization, render includes managed setup/VS/raster/FS; not equal timing scopes or device cycles"
        args.output.mkdir(parents=True, exist_ok=True)
        if args.compare_softpipe:
            reference_started = time.perf_counter()
            reference = softpipe_reference(args.shader_directory.resolve(), args.build.resolve(), args.output,
                                           mesh=args.mesh, depth_test=args.depth_test, texture=args.texture,
                                           blend=args.blend, texture_filter=args.texture_filter,
                                           texture_wrap=args.texture_wrap, clear_color=clear_color,
                                           texture_wrap_s=args.texture_wrap_s, texture_wrap_t=args.texture_wrap_t,
                                           instance_data=args.instance_data, mipmaps=args.mipmaps)
            report["softpipe_reference_wall_seconds"] = time.perf_counter() - reference_started
            report["softpipe_comparison"] = compare_images(image, reference, args.max_channel_error)
            write_png(args.output / "softpipe.png", reference[:, :, :3])
        write_png(args.output / "frame.png", image[:, :, :3])
        (args.output / "frame.rgba8").write_bytes(image.tobytes())
        (args.output / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
        (args.output / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report.get("softpipe_comparison", {"passed": True})["passed"] else 1
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
