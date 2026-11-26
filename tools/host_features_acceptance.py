"""Fresh ABI4 GL/compiler/managed/native acceptance, never FPGA execution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import host_executor
from tools.host_features import DRAW_FEATURES, validate_draw_features
from tools.host_pipeline_state import PipelineState, StencilFace
from tools.host_quad import QuadSimulationBackend
from tools.host_render import render_artifact
from tools.host_texture_features import pack_texture_features
from tools.mgcasm import assemble
from tools.mgcdriver import Driver, Program, ProgramFormat
from tools.run_shader import ROOT, mesa_environment
from tools.shader_layout import ShaderLayout


WIDTH = HEIGHT = 8
MAX_STEPS = 20000000
MAX_FRAGMENTS = 1024
FIXTURE = ROOT / "tests/fixtures/host-features-gl/harness.c"
REQUEST_V4 = struct.Struct("<8sIQIIIIII")
FORMATS = {"rgba8": (1, 4), "bgra8": (2, 4), "rgb565": (3, 2), "r8": (4, 1)}
SOURCE_FORMATS = {0: "R8G8B8A8_UNORM", 1: "R8_UNORM", 2: "R8G8B8_UNORM", 3: "B8G8R8A8_UNORM"}


@dataclass(frozen=True)
class FeatureCase:
    name: str
    texture_version: int | None = None
    derivatives: bool = False
    builtins: tuple[str, ...] = ()
    tolerance: int = 2

    @property
    def state(self):
        inactive_stencil = StencilFace(comparison="never", read_mask=0, write_mask=0)
        inactive = {"depth_stencil_format": "none", "depth_write": False, "depth_comparison": "never",
                    "stencil_front": inactive_stencil, "stencil_back": inactive_stencil}
        if self.name == "separate-blend-factors":
            return PipelineState(**inactive, blend_enabled=True,
                                 src_rgb="constant-color", dst_rgb="one-minus-src-alpha",
                                 src_alpha="one", dst_alpha="one", alpha_equation="reverse-subtract",
                                 constant=(.5, .25, .75, .5))
        if self.name == "rgb565-color-mask":
            return PipelineState(**inactive, color_format="rgb565",
                                 color_mask=(True, False, True, True))
        if self.name == "r8-framebuffer":
            return PipelineState(**inactive, color_format="r8")
        if self.name == "bgra8-framebuffer":
            return PipelineState(**inactive, color_format="bgra8")
        if self.name == "stencil-z24s8":
            return PipelineState(depth_stencil_format="z24s8", depth_enabled=True,
                                 stencil_enabled=True,
                                 stencil_front=StencilFace(reference=6, fail="zero",
                                                           zfail="increment-wrap", zpass="replace"),
                                 stencil_back=StencilFace(reference=11, fail="zero",
                                                          zfail="invert", zpass="replace"))
        return PipelineState(**inactive)


CASES = (
    FeatureCase("runtime-lod-nearest", 11),
    FeatureCase("fractional-lod-linear", 12),
    FeatureCase("fractional-lod-trilinear", 13),
    FeatureCase("multiple-bindings-samples-sparse-32-r8", 11),
    FeatureCase("fragcoord-opposite-facing", builtins=("gl_FragCoord", "gl_FrontFacing")),
    FeatureCase("dfdx-dfdy-quad", derivatives=True),
    FeatureCase("implicit-lod-quad", 11, derivatives=True),
    FeatureCase("local-instance-id", builtins=("gl_InstanceID",)),
    FeatureCase("separate-blend-factors"),
    FeatureCase("stencil-z24s8", tolerance=1),
    FeatureCase("rgb565-color-mask", tolerance=4),
    FeatureCase("r8-framebuffer", tolerance=1),
    FeatureCase("bgra8-framebuffer", tolerance=1),
)
OPTIONAL_CASES = (FeatureCase("vertex-local-instance-id", builtins=("gl_VertexID", "gl_InstanceID")),
                  FeatureCase("base-instance", builtins=("gl_VertexID", "gl_InstanceID", "gl_BaseInstance")))
ANISOTROPY_CASES = (
    FeatureCase("anisotropy-v14-implicit-four-tap", 14, derivatives=True),
    FeatureCase("anisotropy-v14-grad-three-tap-txl-fallback", 14),
    FeatureCase("anisotropy-v14-constant-ewa-control", 14, derivatives=True),
)
ALL_CASES = {case.name: case for case in (*CASES, *OPTIONAL_CASES, *ANISOTROPY_CASES)}


class AcceptanceError(AssertionError):
    pass


class ReplyObserverSocket:
    def __init__(self, connection, headers):
        self.connection = connection
        self.headers = headers

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def sendall(self, data):
        result = self.connection.sendall(data)
        values = host_executor.REPLY_FEATURES.unpack(data[:host_executor.REPLY_FEATURES.size])
        self.headers.append({"magic": values[0].rstrip(b"\0").decode("ascii"), "bytes": 44,
                             "sequence": values[1], "status": values[2], "width": values[3], "height": values[4],
                             "color_format": values[5], "depth_format": values[6], "color_bytes": values[7],
                             "depth_bytes": values[8], "payload_bytes": len(data) - 44,
                             "payload_sha256": hashlib.sha256(data[44:]).hexdigest()})
        return result


class FrontendBlocked(AcceptanceError):
    def __init__(self, version, log, exit_code, gl_version, extensions):
        super().__init__(f"actual GLSL {version} frontend rejects required GLSL 130; see {log}")
        self.version = version
        self.log = log
        self.exit_code = exit_code
        self.gl_version = gl_version
        self.extensions = extensions


def _require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_private_harness(build, directory):
    build, directory = Path(build).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    libraries = (build / "src/egl/libEGL.so", build / "src/mapi/es2api/libGLESv2.so")
    _require(all(path.is_file() for path in libraries), f"actual Mesa EGL/GLES libraries required: {build}")
    environment = dict(os.environ)
    environment["PKG_CONFIG_PATH"] = os.pathsep.join(filter(None, (
        str(build / "meson-uninstalled"), environment.get("PKG_CONFIG_PATH"))))
    flags = subprocess.run(["pkg-config", "--cflags", "--libs", "egl", "glesv2"],
                           env=environment, capture_output=True, text=True, timeout=30)
    log = directory / "private-build.log"
    log.write_text(flags.stdout + flags.stderr)
    _require(flags.returncode == 0, f"real Mesa pkg-config failed: {log}")
    link_flags = shlex.split(flags.stdout)
    _require(all(f"-L{path.parent}" in link_flags for path in libraries),
             "private harness must link the requested Mesa build, not system libraries")
    binary = directory / "mgc_host_features_harness"
    sources = {str(path): _sha256(path) for path in (FIXTURE, ROOT / "tools/mgc_harness.c")}
    command = [*shlex.split(os.environ.get("CC", "cc")), "-std=c11", "-Wall", "-Wextra",
               "-Werror=implicit-function-declaration", "-I", str(ROOT), str(FIXTURE),
               "-o", str(binary), *link_flags, "-lm"]
    compiled = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    log.write_text(shlex.join(command) + "\n" + flags.stdout + flags.stderr + compiled.stdout + compiled.stderr)
    _require(compiled.returncode == 0, f"private real GL harness compile failed: {log}")
    _require(all(_sha256(path) == digest for path, digest in sources.items()),
             "harness source changed during private compilation; rebuild the private harness")
    (directory / "private-build.json").write_text(json.dumps({
        "command": command, "binary": str(binary), "binary_sha256": _sha256(binary),
        "source_sha256": sources, "build": str(build), "shared_library_rebuild": False}, indent=2) + "\n")
    return binary


def _quad(left=-1, right=1, depth=0, clockwise=False, color=(.75, .25, .5), bottom=-1, top=1):
    corners = ((left, bottom), (right, bottom), (right, top), (left, top))
    indices = (0, 2, 1, 0, 3, 2) if clockwise else (0, 1, 2, 0, 2, 3)
    return b"".join(struct.pack("<7f", *corners[index], depth, 1, *color) for index in indices)


def mesh_bytes(case):
    if case.texture_version == 14:
        return _quad(right=-.5, top=-.5)
    if case.name == "fragcoord-opposite-facing":
        return _quad(right=0, depth=-.5) + _quad(left=0, depth=.5, clockwise=True)
    if case.name == "stencil-z24s8":
        return _quad(depth=-.5) + _quad(depth=.5, clockwise=True, color=(0, 0, 1))
    return _quad()


def uploaded_vertex_bytes(mesh):
    return b"".join(struct.pack("<10f", *record, *record[4:]) for record in struct.iter_unpack("<7f", mesh))


def shader_sources(case):
    if case.name == "local-instance-id":
        vertex = ("#version 120\n#extension GL_ARB_draw_instanced : require\n"
                  "attribute vec4 a_position;\nvarying vec4 v_color;\n"
                  "void main() { gl_Position = a_position; "
                  "v_color = vec4(0.75, float(gl_InstanceIDARB) * 0.25, 0.5, 1.0); }\n")
        fragment = "#version 120\nvarying vec4 v_color;\nvoid main() { gl_FragColor = v_color; }\n"
        return vertex, fragment
    if case.name in ("vertex-local-instance-id", "base-instance"):
        base = case.name == "base-instance"
        extensions = "#extension GL_ARB_draw_instanced : require\n"
        if base:
            extensions += "#extension GL_ARB_shader_draw_parameters : require\n"
        blue = "float(gl_BaseInstanceARB) / 8.0" if base else "0.0"
        vertex = ("#version 130\n" + extensions + "in vec4 a_position;\nflat out vec4 v_color;\n"
                  "void main() { gl_Position = a_position; "
                  f"v_color = vec4(float(gl_VertexID) / 8.0, float(gl_InstanceID) / 4.0, {blue}, 1.0); }}\n")
        fragment = "#version 130\nflat in vec4 v_color;\nvoid main() { gl_FragColor = v_color; }\n"
        return vertex, fragment
    version = 100 if case.name == "bgra8-framebuffer" else 120
    vertex = (f"#version {version}\nattribute vec4 a_position;\nattribute vec3 a_color;\n"
              "uniform float u_lod;\nvarying vec2 v_uv;\nvarying float v_lod;\nvarying vec4 v_color;\n"
              "void main() {\ngl_Position = a_position;\nv_color = vec4(a_color, 0.25);\n")
    vertex += ("v_uv = a_position.xy * vec2(1.0, 0.0625) + vec2(1.25, 0.546875);\n"
               if case.texture_version == 14 else
               "v_uv = a_position.xy * 2.0 + vec2(2.0);\n" if case.name == "implicit-lod-quad" else
               "v_uv = a_position.xy * 0.5 + vec2(0.5);\n")
    vertex += ("v_lod = a_position.x + 1.25;\n" if case.name == "runtime-lod-nearest" else
               "v_lod = u_lod;\n") + "}\n"
    declarations = f"#version {version}\n"
    if version == 100:
        declarations += "precision mediump float;\n"
    if case.texture_version is not None:
        declarations += "#extension GL_ARB_shader_texture_lod : require\n"
    declarations += "varying vec2 v_uv;\nvarying float v_lod;\nvarying vec4 v_color;\n"
    if case.texture_version == 14:
        declarations += "uniform sampler2D u_texture;\n"
        expression = ("texture2DGradARB(u_texture, v_uv, vec2(0.0625, 0.0), vec2(0.0, 0.1875)) * 0.5 + "
                      "texture2DLod(u_texture, v_uv, 0.5) * 0.5"
                      if case.name == "anisotropy-v14-grad-three-tap-txl-fallback" else "texture2D(u_texture, v_uv)")
    elif case.name == "multiple-bindings-samples-sparse-32-r8":
        declarations += "uniform sampler2D u_textures[4];\n"
        expression = ("texture2DLod(u_textures[0], v_uv, v_lod) * 0.25 + "
                      "texture2DLod(u_textures[0], vec2(1.0) - v_uv, v_lod) * 0.25 + "
                      "texture2DLod(u_textures[3], v_uv, v_lod) * 0.5")
    elif case.texture_version is not None:
        declarations += "uniform sampler2D u_texture;\n"
        expression = ("texture2D(u_texture, v_uv)" if case.name == "implicit-lod-quad" else
                      "texture2DLod(u_texture, v_uv, v_lod)")
    elif case.name == "fragcoord-opposite-facing":
        expression = "vec4(gl_FragCoord.xy / 8.0, gl_FragCoord.z, gl_FrontFacing ? 1.0 : 0.25)"
    elif case.name == "dfdx-dfdy-quad":
        expression = "vec4(dFdx(v_uv.x * v_uv.x) * 4.0, dFdy(v_uv.y * v_uv.y) * 4.0, dFdx(v_uv.y) + dFdy(v_uv.x), 1.0)"
    else:
        expression = "v_color"
    fragment = declarations + f"void main() {{ gl_FragColor = {expression}; }}\n"
    return vertex, fragment


def texture_inputs(case):
    if case.texture_version is None:
        return ()
    dimensions = 32 if case.name == "multiple-bindings-samples-sparse-32-r8" or case.texture_version == 14 else 4
    resources = []
    bindings = ((0, 0), (3, 1)) if case.name == "multiple-bindings-samples-sparse-32-r8" else ((0, 0),)
    for binding, source_format in bindings:
        width = height = dimensions
        levels = []
        level = 0
        while True:
            channels = 1 if source_format == 1 else 4
            data = np.empty((height, width, channels), np.uint8)
            for row in range(height):
                for column in range(width):
                    if case.name == "anisotropy-v14-constant-ewa-control":
                        data[row, column] = (19, 63, 127, 255)
                    elif source_format == 1:
                        data[row, column] = ((column * 7 + row * 11 + level * 47 + 19) % 256,)
                    else:
                        data[row, column] = ((column * 41 + level * 67 + 13) % 256,
                                             (row * 53 + level * 29 + 31) % 256,
                                             (column * 17 + row * 23 + level * 73 + 43) % 256, 255)
            levels.append(data)
            if width == height == 1:
                break
            width, height, level = max(1, width // 2), max(1, height // 2), level + 1
        resources.append((binding, source_format, tuple(levels)))
    return tuple(resources)


def encode_textures(resources):
    contents = bytearray(struct.pack("<8sI", b"MGCTX04\0", len(resources)))
    for binding, source_format, levels in resources:
        contents.extend(struct.pack("<IIIII", binding, levels[0].shape[1], levels[0].shape[0],
                                    source_format, len(levels)))
        for level in levels:
            contents.extend(level.tobytes())
    return bytes(contents)


def _texel(level, column, row):
    row = min(max(row, 0), level.shape[0] - 1)
    column = min(max(column, 0), level.shape[1] - 1)
    channels = tuple(Fraction(int(value), 255) for value in level[row, column])
    return (channels[0], Fraction(0), Fraction(0), Fraction(1)) if len(channels) == 1 else channels


def _sample_level(level, uv, linear):
    if not linear:
        return _texel(level, int(uv[0] * level.shape[1] // 1), int(uv[1] * level.shape[0] // 1))
    horizontal = uv[0] * level.shape[1] - Fraction(1, 2)
    vertical = uv[1] * level.shape[0] - Fraction(1, 2)
    column, row = int(horizontal // 1), int(vertical // 1)
    weight_x, weight_y = horizontal - column, vertical - row
    samples = tuple(_texel(level, column + offset_x, row + offset_y)
                    for offset_y in (0, 1) for offset_x in (0, 1))
    weights = ((1 - weight_x) * (1 - weight_y), weight_x * (1 - weight_y),
               (1 - weight_x) * weight_y, weight_x * weight_y)
    return tuple(sum(sample[channel] * weight for sample, weight in zip(samples, weights)) for channel in range(4))


def rational_sample(levels, uv, lod, version):
    lod = min(max(lod, Fraction(0)), Fraction(len(levels) - 1))
    if version not in (13, 14):
        selected = int((lod + Fraction(1, 2)) // 1)
        return _sample_level(levels[selected], uv, version == 12)
    selected = int(lod // 1)
    weight = lod - selected
    lower = _sample_level(levels[selected], uv, True)
    upper = _sample_level(levels[min(selected + 1, len(levels) - 1)], uv, True)
    return tuple((1 - weight) * first + weight * second for first, second in zip(lower, upper))


def anisotropy_footprint(case, width, height, last_level):
    if case.name == "anisotropy-v14-grad-three-tap-txl-fallback":
        gradient_x, gradient_y = (Fraction(1, 16), Fraction(0)), (Fraction(0), Fraction(3, 16))
    else:
        gradient_x, gradient_y = (Fraction(1, 4), Fraction(0)), (Fraction(0), Fraction(1, 64))
    squared = tuple((gradient[0] * width) ** 2 + (gradient[1] * height) ** 2
                    for gradient in (gradient_x, gradient_y))
    major = gradient_x if squared[0] >= squared[1] else gradient_y
    maximum, minimum = max(squared), min(squared)
    length = math.sqrt(max(maximum, 1))
    minor = math.sqrt(max(minimum, maximum / 16, 1))
    taps = min(max(math.ceil(length / minor), 1), 4)
    lod = Fraction(min(max(math.log2(min(minor, 32)), 0), last_level))
    return major, length, minor, taps, lod


def rational_anisotropic_sample(case, levels, uv):
    major, _, _, taps, lod = anisotropy_footprint(case, levels[0].shape[1], levels[0].shape[0], len(levels) - 1)
    samples = []
    for tap in range(taps):
        offset = Fraction(2 * tap + 1, 2 * taps) - Fraction(1, 2)
        coordinate = tuple(center + offset * gradient for center, gradient in zip(uv, major))
        samples.append(rational_sample(levels, coordinate, lod, 13))
    return tuple(sum(sample[channel] for sample in samples) / taps for channel in range(4))


def expected_coverage(case):
    mask = np.ones((HEIGHT, WIDTH), np.bool_)
    if case.texture_version == 14:
        mask[:] = False
        mask[-2:, :2] = True
    return mask


def _round_channel(value, maximum=255):
    return min(max(int(value * maximum + Fraction(1, 2)), 0), maximum)


def rational_oracle(case, resources=None):
    resources = texture_inputs(case) if resources is None else resources
    image = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    mask = expected_coverage(case)
    depths = None
    if case.name == "stencil-z24s8":
        depth_code = int(Fraction(1, 4) * 16777215 + Fraction(1, 2))
        depths = np.full((HEIGHT, WIDTH), depth_code | (249 << 24), dtype="<u4")
    for image_row in range(HEIGHT):
        row = HEIGHT - image_row - 1
        for column in range(WIDTH):
            if not mask[image_row, column]:
                continue
            uv = (Fraction(2 * column + 1, 2 * WIDTH), Fraction(2 * row + 1, 2 * HEIGHT))
            if case.texture_version == 14:
                uv = (2 * uv[0] + Fraction(1, 4), uv[1] / 8 + Fraction(31, 64))
                rgba = rational_anisotropic_sample(case, resources[0][2], uv)
                if case.name == "anisotropy-v14-grad-three-tap-txl-fallback":
                    isotropic = rational_sample(resources[0][2], uv, Fraction(1, 2), 13)
                    rgba = tuple((line + single) / 2 for line, single in zip(rgba, isotropic))
            elif case.texture_version is not None:
                lod = (uv[0] * 2 + Fraction(1, 4) if case.name == "runtime-lod-nearest" else
                       Fraction(1, 2) if case.texture_version == 13 else Fraction(1, 4))
                if case.name == "implicit-lod-quad":
                    uv, lod = tuple(coordinate * 4 for coordinate in uv), Fraction(1)
                rgba = rational_sample(resources[0][2], uv, lod, case.texture_version)
                if len(resources) > 1:
                    mirrored = rational_sample(resources[0][2], tuple(1 - coordinate for coordinate in uv), lod, 11)
                    second = rational_sample(resources[1][2], uv, lod, 11)
                    rgba = tuple(first / 4 + other / 4 + red / 2
                                 for first, other, red in zip(rgba, mirrored, second))
            elif case.name == "fragcoord-opposite-facing":
                rgba = (*uv, Fraction(1, 4) if column < WIDTH // 2 else Fraction(3, 4),
                        Fraction(1) if column < WIDTH // 2 else Fraction(1, 4))
            elif case.name == "dfdx-dfdy-quad":
                lower_x, upper_x = Fraction(4 * (column // 2) + 1, 2 * WIDTH), Fraction(4 * (column // 2) + 3, 2 * WIDTH)
                lower_y, upper_y = Fraction(4 * (row // 2) + 1, 2 * HEIGHT), Fraction(4 * (row // 2) + 3, 2 * HEIGHT)
                rgba = (4 * (upper_x ** 2 - lower_x ** 2), 4 * (upper_y ** 2 - lower_y ** 2), Fraction(0), Fraction(1))
            elif case.name in ("vertex-local-instance-id", "base-instance"):
                vertex_id = 2 if row < column else 5
                rgba = (Fraction(vertex_id, 8), Fraction(1, 4),
                        Fraction(3, 8) if case.name == "base-instance" else Fraction(0), Fraction(1))
            elif case.name == "local-instance-id":
                rgba = (Fraction(3, 4), Fraction(1, 4), Fraction(1, 2), Fraction(1))
            elif case.name == "separate-blend-factors":
                source = (Fraction(3, 4), Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))
                destination = tuple(Fraction(value, 255) for value in (64, 128, 191, 191))
                constant = (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4))
                rgba = tuple(source[channel] * constant[channel] + destination[channel] * (1 - source[3])
                             for channel in range(3)) + (destination[3] - source[3],)
            else:
                rgba = (Fraction(3, 4), Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))
                if case.name == "rgb565-color-mask":
                    rgba = (Fraction(_round_channel(rgba[0], 31), 31),
                            Fraction(_round_channel(Fraction(1, 2), 63), 63),
                            Fraction(_round_channel(rgba[2], 31), 31), Fraction(1))
                elif case.name == "r8-framebuffer":
                    rgba = (rgba[0], Fraction(0), Fraction(0), Fraction(1))
            image[image_row, column] = tuple(_round_channel(channel) for channel in rgba)
    return image, depths


def _compare(actual, expected, tolerance, label, *, enforce=True):
    _require(actual.shape == expected.shape and actual.dtype == expected.dtype, f"{label}: typed shape mismatch")
    delta = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
    maximum = int(delta.max())
    result = {"passed": maximum <= tolerance, "max_error": maximum,
              "tolerance": tolerance, "different_channels": int(np.count_nonzero(delta))}
    if enforce:
        _require(result["passed"], f"{label}: max error {maximum} exceeds {tolerance}")
    else:
        result["matches_within_tolerance"] = result.pop("passed")
        result.update(required_for_acceptance=False, kernel_equivalence_claimed=False)
    return result


def _environment(build, directory, case, mesh, textures):
    environment = mesa_environment(build, directory / "compiler", ProgramFormat.LEGACY)
    for name in tuple(environment):
        if name.startswith(("MGC_HOST_", "MGC_ACCEPT_", "MGC_TEST_")) or name in (
                "MESA_GL_VERSION_OVERRIDE", "MESA_GLES_VERSION_OVERRIDE", "MESA_GLSL_VERSION_OVERRIDE",
                "MESA_EXTENSION_OVERRIDE"):
            environment.pop(name)
    environment.update(MGC_HOST_FEATURES="1", MGC_FRAGMENT_STATUS="1", MGC_ACCEPT_CASE=case.name,
                       MGC_HOST_MESH=str(mesh), MGC_HOST_WIDTH=str(WIDTH), MGC_HOST_HEIGHT=str(HEIGHT))
    if textures is not None:
        environment.update(MGC_HOST_TEXTURE=str(case.texture_version), MGC_HOST_TEXTURE_WRAP_S="clamp-to-edge",
                           MGC_HOST_TEXTURE_WRAP_T="clamp-to-edge", MGC_ACCEPT_TEXTURES=str(textures))
    return environment


def _invoke(binary, build, output, shader, case, mesh, textures, action, socket_path=None):
    output.mkdir(parents=True, exist_ok=True)
    (output / "compiler").mkdir()
    captures = output / "captures"
    captures.mkdir()
    environment = _environment(build, output, case, mesh, textures)
    environment["GALLIUM_DRIVER"] = "softpipe" if action == "reference" else "mgc"
    raw = output / "readback.rgba8"
    depth = output / "readback.z24s8"
    command = [str(binary), str(shader / "vert.glsl"), str(shader / "frag.glsl")]
    if action == "capture":
        command.append("--draw")
    else:
        command.extend(("--reference" if action == "reference" else "--host-execute", str(raw)))
    if action != "reference":
        environment["MGC_HOST_CAPTURE_DIR"] = str(captures)
    if action == "native":
        environment["MGC_HOST_EXECUTOR_SOCKET"] = str(socket_path)
    if case.state.depth_stencil_format == "z24s8":
        environment["MGC_ACCEPT_DEPTH_OUTPUT"] = str(depth)
    (output / "invocation.json").write_text(json.dumps({"command": command, "environment": {
        name: value for name, value in environment.items() if name.startswith(("MGC_", "MESA_", "GALLIUM_", "EGL_"))}}, indent=2) + "\n")
    try:
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        text = b"".join(part for part in (error.stdout, error.stderr) if isinstance(part, bytes)).decode(errors="replace")
        (output / "gl.log").write_text(text + "\nbounded subprocess timeout\n")
        raise
    log = completed.stdout + completed.stderr
    (output / "gl.log").write_text(log)
    if action == "capture" and case in OPTIONAL_CASES and completed.returncode != 0:
        language = re.search(r"^GL_SHADING_LANGUAGE_VERSION: (\d+)\.(\d+)", log, re.MULTILINE)
        if language is not None:
            actual_version = int(language[1]) * 100 + int(language[2].ljust(2, "0"))
            if actual_version < 130 and "GLSL 1.30 is not supported" in log:
                _require(completed.returncode > 0, "unsupported GLSL must fail compilation without aborting")
                _require("GL_RENDERER: mgc\n" in log, "frontend gate requires an actual MGC GL context")
                _require(not list(captures.glob("*.json")), "blocked GLSL frontend must not emit a pretend capture")
                gl_version = re.search(r"^GL_VERSION: (.+)$", log, re.MULTILINE)
                extensions = {name: value == "1" for name, value in re.findall(
                    r"^MGC_ACCEPT_CAPABILITY (GL_\w+)=(0|1)$", log, re.MULTILINE)}
                _require(gl_version is not None and len(extensions) == 4,
                         "frontend gate requires actual GL version and advertised extension evidence")
                raise FrontendBlocked(actual_version, output / "gl.log", completed.returncode,
                                      gl_version[1], extensions)
    _require(completed.returncode == 0, f"fresh {action} GL failed (exit {completed.returncode}); see {output / 'gl.log'}")
    _require(f"GL_RENDERER: {environment['GALLIUM_DRIVER']}\n" in log, f"{action}: actual renderer proof missing")
    _require("MGC_ACCEPT_DRAW" in log, f"{action}: actual draw proof missing")
    if case.texture_version == 14:
        anisotropy = re.search(r"^MGC_ACCEPT_ANISOTROPY maximum=(\d+) sampler=(\d+) arb=(0|1)$", log, re.MULTILINE)
        _require(anisotropy is not None and int(anisotropy[2]) == 4,
                 f"{action}: genuine EXT max_anisotropy4 sampler proof missing")
        if action != "reference":
            _require(anisotropy[1] == "4" and anisotropy[3] == "0",
                     "actual bounded MGC EXT must advertise max4 without pretending ARB min16")
    artifacts = []
    for path in sorted(captures.glob("*.json")):
        _require(path.stat().st_size <= 12 << 20, "fresh capture exceeds native JSON budget")
        artifacts.append(json.loads(path.read_text()))
    if action == "reference":
        _require(not artifacts, "softpipe must not masquerade as an MGC capture")
    else:
        _require(len(artifacts) == 1, f"{action}: expected one actual capture, received {len(artifacts)}")
        for stage in ("vs", "fs"):
            compiler_layout = json.loads((output / "compiler" / f"{stage}.layout.json").read_text())
            compiler_assembly = (output / "compiler" / f"{stage}.asm").read_text()
            _require(artifacts[0]["shaders"][stage]["layout"] == compiler_layout,
                     f"{action}/{stage}: captured reflection differs from the fresh actual compiler dump")
            _require(artifacts[0]["shaders"][stage]["assembly"] == compiler_assembly,
                     f"{action}/{stage}: captured assembly differs from the fresh actual compiler dump")
    image = depth_image = None
    if action != "capture":
        _require("MGC_ACCEPT_READBACK actual_glReadPixels=1" in log, "actual native/reference glReadPixels proof missing")
        _require(raw.stat().st_size == WIDTH * HEIGHT * 4, "real GL readback size mismatch")
        image = np.frombuffer(raw.read_bytes(), np.uint8).reshape(HEIGHT, WIDTH, 4)[::-1].copy()
        if case.state.depth_stencil_format == "z24s8":
            _require(depth.stat().st_size == WIDTH * HEIGHT * 4, "real GL depth/stencil readback size mismatch")
            depth_image = np.frombuffer(depth.read_bytes(), "<u4").reshape(HEIGHT, WIDTH)[::-1].copy()
    return artifacts, image, depth_image


def assert_capture(artifact, case, resources, mesh):
    _require(artifact.get("schema") == "mgc-host-draw-v4", "actual draw requires capturev4, not legacy capture")
    _require(artifact.get("program_format") == ProgramFormat.LEGACY.value and
             type(artifact.get("shader_abi_version")) is int and artifact["shader_abi_version"] == 4,
             "actual draw must be explicitly legacy ABI4")
    _require(artifact.get("host_features") == DRAW_FEATURES, "actual capture host_features contract mismatch")
    _require(validate_draw_features(artifact), "actual capture failed the strict parent-owned ABI4 validator")
    _require(PipelineState.from_dict(artifact["render_state"]) == case.state,
             "actual GL state types differ from the validated pipeline contract")
    _require(artifact.get("render_state") == case.state.to_dict(), "actual GL state differs from exact requested render_state")
    draw = artifact["draw"]
    _require(draw["mode"] == "triangles" and draw["count"] == len(mesh) // 28 and draw["index_size"] == 0,
             "capture does not contain the real fixture triangles")
    _require(draw["instance_count"] == (2 if case.name in ("local-instance-id", "vertex-local-instance-id", "base-instance") else 1),
             "actual GL instance count mismatch")
    _require(draw["start_instance"] == (3 if case.name == "base-instance" else 0), "actual GL base-instance mismatch")
    buffers = artifact["vertex_buffers"]
    _require(any(buffer["stride"] == 40 and bytes.fromhex(buffer["data_hex"]) == uploaded_vertex_bytes(mesh)
                 for buffer in buffers), "captured mesh differs from actual uploaded FP32 vertex-buffer bytes")
    layouts = {stage: ShaderLayout(artifact["shaders"][stage]["layout"]) for stage in ("vs", "fs")}
    for stage, layout in layouts.items():
        _require(layout.target is ProgramFormat.LEGACY and layout.source["version"] == 4 and
                 layout.source["shader_abi_version"] == 4, f"{stage}: fresh compiler layout4/ABI4 required")
        _require(artifact["shaders"][stage]["assembly"].startswith("; mgc legacy-simulator backend, ABI=4,"),
                 f"{stage}: fresh compiler assembly ABI4 header required")
        _require(layout.source["supported"] is True and layout.host_features is not None,
                 f"{stage}: actual supported compiler reflection required")
        expected_builtins = set(case.builtins) & ({"gl_VertexID", "gl_InstanceID", "gl_BaseInstance"} if stage == "vs" else
                                                 {"gl_FragCoord", "gl_FrontFacing"})
        _require({entry["name"] for entry in layout.host_features["builtins"]} == expected_builtins,
                 f"{stage}: compiler builtin reflection does not match actual GLSL use")
        if "gl_FragCoord" in expected_builtins:
            declaration = next(entry for entry in layout.host_features["builtins"] if entry["name"] == "gl_FragCoord")
            _require(declaration["word_offsets"] == [65524, 65525, 65526],
                     "actual FragCoord xyz reflection must preserve the used scalar subset, not invent w")
        _require(layout.host_features["uses_derivatives"] is (stage == "fs" and case.derivatives),
                 f"{stage}: actual derivative/implicit-LOD reflection mismatch")
    _require(layouts["fs"].fragment_status == {"version": 1, "word_offset": 65535, "pending": 0,
                                             "discarded": 1, "committed": 2}, "ABI4 FS completion contract required")
    if resources:
        texture_contract = layouts["fs"].host_texture
        _require(texture_contract["version"] == case.texture_version, "wrong compiled ABI4 texture mode")
        pairs = [{"texture": resource[0], "sampler": resource[0]} for resource in resources]
        _require(texture_contract["bindings"] == pairs, "actual static/sparse GLSL binding pairs not preserved")
        expected_lod_mode = ("mixed" if case.name == "anisotropy-v14-grad-three-tap-txl-fallback" else
                             "implicit" if case.derivatives else "explicit")
        _require(texture_contract["lod_mode"] == expected_lod_mode,
                 "runtime shader LOD mode missing from compiler reflection")
        if case.texture_version == 14:
            _require(type(texture_contract.get("max_anisotropy")) is int and texture_contract["max_anisotropy"] == 4,
                     "v14 fresh compiler must reflect the exact bounded anisotropy4 contract")
        captures = artifact.get("textures")
        _require(type(captures) is list and len(captures) == len(resources), "all used real texture resources must be captured")
        for capture, (binding, source_format, levels) in zip(captures, resources):
            if case.texture_version == 14:
                _require(type(capture["sampler_state"].get("max_anisotropy")) is int and
                         capture["sampler_state"]["max_anisotropy"] == 4,
                         "actual v14 sampler capture must preserve exact integer max_anisotropy4")
            _require((capture["texture"], capture["sampler"]) == (binding, binding), "actual texture pair mismatch")
            _require(capture["binding"] == binding, "captured binding must identify its actual sparse texture plane")
            _require(capture["format"] == SOURCE_FORMATS[source_format], "actual source texture format was not preserved")
            _require(capture["data_hex"] == levels[0].tobytes().hex(), "capture changed uploaded base-level bytes")
            _require((capture["width"], capture["height"]) == (levels[0].shape[1], levels[0].shape[0]),
                     "capture changed uploaded dimensions")
            _require(len(capture["mipmaps"]) == len(levels) - 1, "capture dropped actual uploaded mip levels")
            for index, (captured, level) in enumerate(zip(capture["mipmaps"], levels[1:]), 1):
                _require(captured == {"level": index, "width": level.shape[1], "height": level.shape[0],
                                      "format": SOURCE_FORMATS[source_format], "data_hex": level.tobytes().hex()},
                         "captured typed mip differs from uploaded bytes")
        payload = pack_texture_features(layouts["fs"], captures)
        _require(bool(payload) and all(type(word) is int for word in payload.values()), "actual captured texture input packing failed")
    else:
        _require(layouts["fs"].host_texture is None and not artifact.get("textures"), "untextured shader captured spurious resources")
    return layouts


def _build_identity(build):
    drivers = sorted((build / "src/gallium/targets/dri").glob("libgallium-*.so"))
    _require(len(drivers) == 1, "acceptance requires one unambiguous actual Mesa Gallium shared library")
    paths = list(drivers)
    paths.extend((build / "src/egl/libEGL.so", build / "src/mapi/es2api/libGLESv2.so"))
    return {str(path.resolve()): _sha256(path) for path in paths if path.is_file()}


def run_case(case, build, binary, output, *, reference_only=False):
    build, binary, output = Path(build).resolve(), Path(binary).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"case": case.name, "passed": False, "status": "running",
                "scope": "fresh real GL -> actual compiler/reflection -> managed legacy CPU -> native C IPC -> glReadPixels",
                "fpga_execution": False, "not_fpga": True, "physical_opengl_conformance": False,
                "reference_only": reference_only, "native_gl_readback": False,
                "required_versions": {"shader_abi": 4, "capture": 4, "layout": 4},
                "width": WIDTH, "height": HEIGHT, "build": str(build),
                "harness": str(binary), "harness_sha256": _sha256(binary), "fixture_sha256": _sha256(FIXTURE),
                "source_derived_from": str(ROOT / "tools/mgc_harness.c"),
                "oracle": "independent Fraction texel/mip interpolation, pixel centers, numeric blend and packed stencil",
                "limits": {"subprocess_seconds": 30, "max_steps": MAX_STEPS, "max_fragments": MAX_FRAGMENTS},
                "checks": {}, "comparisons": {}, "submissions": [], "native_request_headers": [], "native_reply_headers": [],
                "quad_derivative_barriers": {"managed": {"dx": 0, "dy": 0}, "native": {"dx": 0, "dy": 0}}}
    report_path = output / "acceptance.json"
    def save():
        report_path.write_text(json.dumps(evidence, indent=2) + "\n")
    save()
    started = time.monotonic()
    stage = "fixture"
    server = None
    original_submit = Driver.submit
    original_receive = host_executor._receive
    original_handle = host_executor.HostExecutorServer._handle
    original_resolve = QuadSimulationBackend._resolve
    request_prefixes = {}
    request_json = {}
    def observed_handle(executor, connection, pid):
        return original_handle(executor, ReplyObserverSocket(connection, evidence["native_reply_headers"]), pid)
    program_hashes = {}
    def observed_submit(driver, program, **options):
        if id(program) not in program_hashes:
            program_hashes[id(program)] = (program, hashlib.sha256(program.to_bytes()).hexdigest())
        submission = {"phase": stage, "format": program.format.value,
                      "backend": type(driver.backend).__name__, "buffers": len(options.get("buffers", ())),
                      "program_sha256": program_hashes[id(program)][1]}
        if submission["backend"] == "QuadSimulationBackend":
            submission["quad_lanes"] = len(driver.backend._regions or ())
            _require(submission["quad_lanes"] == 4, "actual derivative submission requires four configured managed lanes")
        evidence["submissions"].append(submission)
        _require(program.format is ProgramFormat.LEGACY, "actual ABI4 shader submitted to nonlegacy backend")
        return original_submit(driver, program, **options)
    def observed_receive(connection, size, deadline):
        data = original_receive(connection, size, deadline)
        if connection in request_json:
            request = request_json.pop(connection)
            _require(size == request["json_bytes"] and len(data) == size, "actual native request JSON byte count mismatch")
            request["json_sha256"] = hashlib.sha256(data).hexdigest()
        header = None
        if size == 8 and data == b"MGCHE04\0":
            request_prefixes[connection] = data
        elif connection in request_prefixes:
            _require(size == REQUEST_V4.size - 8, "actual native ABI4 header length mismatch")
            header = request_prefixes.pop(connection) + data
        elif size == REQUEST_V4.size and data.startswith(b"MGCHE04\0"):
            header = data
        if header is not None:
            values = REQUEST_V4.unpack(header)
            evidence["native_request_headers"].append({"magic": "MGCHE04", "bytes": size,
                "json_bytes": values[1], "sequence": values[2], "width": values[3], "height": values[4],
                "color_format": values[5], "depth_format": values[6], "color_bytes": values[7], "depth_bytes": values[8]})
            evidence["native_request_headers"][-1]["bytes"] = len(header)
            request_json[connection] = evidence["native_request_headers"][-1]
        return data
    def observed_resolve(backend):
        if all(backend._waiting):
            _require(len(backend._cpus) == 4 and len(set(backend._waiting)) == 1,
                     "actual derivative barrier requires four synchronized CPU lanes")
            operation = backend._waiting[0][1]
            _require(operation in (1, 2), "actual derivative mailbox operation must be dx or dy")
            evidence["quad_derivative_barriers"][stage]["dx" if operation == 1 else "dy"] += 1
        return original_resolve(backend)
    try:
        identity = _build_identity(build)
        _require(len(identity) == 3, "actual frozen shared Mesa libraries required; acceptance never rebuilds them")
        evidence["build_identity"] = identity
        manifest = json.loads((binary.parent / "private-build.json").read_text())
        _require(manifest["binary"] == str(binary) and manifest["binary_sha256"] == evidence["harness_sha256"],
                 "private harness binary differs from its actual compilation manifest")
        _require(manifest["build"] == str(build) and manifest["shared_library_rebuild"] is False,
                 "private harness must be compiled against this build without a shared rebuild")
        expected_sources = {str(path): _sha256(path) for path in (FIXTURE, ROOT / "tools/mgc_harness.c")}
        _require(manifest["source_sha256"] == expected_sources,
                 "private harness includes stale source; privately rebuild it before actual acceptance")
        evidence["private_compile_manifest"] = manifest
        evidence["checks"]["private_source_binary_provenance"] = {"passed": True}
        shader = output / "shader"
        shader.mkdir()
        vertex, fragment = shader_sources(case)
        (shader / "vert.glsl").write_text(vertex)
        (shader / "frag.glsl").write_text(fragment)
        evidence["shader_sha256"] = {name: _sha256(shader / name) for name in ("vert.glsl", "frag.glsl")}
        mesh = output / "mesh.fp32"
        mesh.write_bytes(mesh_bytes(case))
        resources = texture_inputs(case)
        if case.texture_version == 14:
            major, length, minor, taps, lod = anisotropy_footprint(case, 32, 32, len(resources[0][2]) - 1)
            evidence.update(scope="bounded v14 max4 uniformly spaced line taps with trilinear mip filtering; NOT EWA",
                            kernel={"name": "bounded-line-tap-trilinear", "max_anisotropy": 4, "taps": taps,
                                    "major_uv": [str(component) for component in major], "major_length": length,
                                    "effective_minor_length": minor, "lod": str(lod), "not_softpipe_ewa": True,
                                    "txl_fallback": "single isotropic v13 trilinear sample"})
        textures = output / "textures.bin" if resources else None
        if textures is not None:
            textures.write_bytes(encode_textures(resources))
        expected, expected_depth = rational_oracle(case, resources)
        expected.tofile(output / "rational.rgba8")
        if expected_depth is not None:
            expected_depth.tofile(output / "rational.z24s8")
        artifact = None
        if case in OPTIONAL_CASES and not reference_only:
            stage = "capture"
            artifacts, _, _ = _invoke(binary, build, output / stage, shader, case, mesh, textures, "capture")
            artifact = artifacts[0]
        stage = "softpipe"
        _, reference, reference_depth = _invoke(binary, build, output / stage, shader, case, mesh, textures, "reference")
        if case.texture_version == 14:
            evidence["comparisons"]["softpipe_ewa_vs_line_tap_oracle"] = _compare(reference, expected, case.tolerance,
                                                                                 stage, enforce=False)
            evidence["checks"]["actual_softpipe_ewa_readback"] = {"passed": True, "universal_acceptance_oracle": False}
            if case.name == "anisotropy-v14-constant-ewa-control":
                evidence["checks"]["constant_fixture_softpipe_vs_oracle"] = _compare(reference, expected, 1, stage)
        else:
            evidence["checks"]["softpipe_vs_independent_oracle"] = _compare(reference, expected, case.tolerance, stage)
        if expected_depth is not None:
            evidence["checks"]["softpipe_depth_stencil"] = _compare(reference_depth, expected_depth, 1, "softpipe packed Z24S8")
        save()
        if reference_only:
            evidence.update(status="reference-preflight-only", scope="actual softpipe fixture/oracle preflight; ABI4 MGC acceptance NOT executed")
            if case.texture_version == 14:
                evidence["scope"] = "actual softpipe EWA observation; v14 line-tap MGC acceptance NOT executed"
            return evidence
        stage = "capture"
        if artifact is None:
            artifacts, _, _ = _invoke(binary, build, output / stage, shader, case, mesh, textures, "capture")
            artifact = artifacts[0]
        assert_capture(artifact, case, resources, mesh.read_bytes())
        evidence["compiled_program_sha256"] = {name: hashlib.sha256(Program.from_words(
            assemble(shader_data["assembly"]), format=ProgramFormat.LEGACY).to_bytes()).hexdigest()
            for name, shader_data in artifact["shaders"].items()}
        evidence["checks"]["fresh_compiler_capture_reflection"] = {"passed": True, "schema": artifact["schema"],
                                                                 "actual_compiler_dump_match": True}
        initial = np.zeros_like(expected)
        if case.name in ("separate-blend-factors", "rgb565-color-mask"):
            initial[:] = (64, 128, 191, 191)
        depth_output = bytearray(WIDTH * HEIGHT * 4) if expected_depth is not None else None
        stage = "managed"
        with patch.object(Driver, "submit", observed_submit), patch.object(QuadSimulationBackend, "_resolve", observed_resolve):
            managed, coverage, managed_report = render_artifact(artifact, width=WIDTH, height=HEIGHT,
                initial_image=initial, max_steps=MAX_STEPS, max_fragments=MAX_FRAGMENTS,
                execution_timeout=30, depth_stencil_output=depth_output)
        managed.tofile(output / "managed.rgba8")
        coverage.tofile(output / "managed.coverage.u32")
        (output / "managed-report.json").write_text(json.dumps(managed_report, indent=2) + "\n")
        _require(managed_report["fpga_execution"] is False and managed_report["shader_abi_version"] == 4,
                 "actual managed CPU report must identify ABI4, never FPGA")
        _require(np.array_equal(coverage > 0, expected_coverage(case)), "actual managed coverage differs from the independent bounded geometry")
        evidence["checks"]["managed_vs_oracle"] = _compare(managed, expected, case.tolerance, "managed")
        if case.texture_version == 14:
            evidence["comparisons"]["managed_line_tap_vs_softpipe_ewa"] = _compare(managed, reference, case.tolerance,
                                                                                 "managed versus EWA", enforce=False)
        else:
            evidence["checks"]["managed_vs_softpipe"] = _compare(managed, reference, case.tolerance, "managed versus real softpipe")
        if expected_depth is not None:
            managed_depth = np.frombuffer(depth_output, "<u4").reshape(HEIGHT, WIDTH)
            evidence["checks"]["managed_depth_stencil"] = _compare(managed_depth, expected_depth, 1, "managed Z24S8")
        save()
        stage = "native"
        _require(host_executor.REQUEST_FEATURES_MAGIC == b"MGCHE04\0" and
                 host_executor.REPLY_FEATURES_MAGIC == b"MGCRG04\0" and
                 host_executor.REQUEST_FEATURES.size == host_executor.REPLY_FEATURES.size == REQUEST_V4.size == 44,
                 "parent-owned native executor must expose the exact v4 request/reply protocol")
        with tempfile.TemporaryDirectory(prefix="mgc-abi4-", dir="/tmp") as socket_directory, \
                patch.object(Driver, "submit", observed_submit), patch.object(host_executor, "_receive", observed_receive), \
                patch.object(QuadSimulationBackend, "_resolve", observed_resolve), \
                patch.object(host_executor.HostExecutorServer, "_handle", observed_handle):
            socket_path = Path(socket_directory) / "executor.sock"
            evidence["native_socket_path"] = str(socket_path)
            with host_executor.HostExecutorServer(socket_path, max_requests=1, max_steps=MAX_STEPS, max_fragments=MAX_FRAGMENTS) as server:
                native_artifacts, native, native_depth = _invoke(binary, build, output / stage, shader, case, mesh, textures,
                                                                 "native", socket_path=server.path)
        _require(not server.errors and len(server.records) == 1, f"actual native executor failed: {server.errors}")
        record = server.records[0]
        assert_capture(record["artifact"], case, resources, mesh.read_bytes())
        _require(native_artifacts[0] == record["artifact"], "native captured JSON differs from actual IPC artifact")
        for name in ("shaders", "vertex_buffers", "vertex_elements", "draw", "render_state", "textures"):
            _require(artifact.get(name) == record["artifact"].get(name), f"capture/native fresh same-input {name} mismatch")
        _require(record.get("peer_pid", 0) > 0 and record["sequence"] > 0, "native C peer/sequence proof missing")
        _require(record["report"]["fpga_execution"] is False, "native executor incorrectly claims FPGA execution")
        headers = evidence["native_request_headers"]
        replies = evidence["native_reply_headers"]
        color_code, color_bytes = FORMATS[case.state.color_format]
        _require(len(headers) == 1 and headers[0]["color_format"] == color_code and
                 headers[0]["bytes"] == 44 and headers[0]["width"] == WIDTH and headers[0]["height"] == HEIGHT and
                 headers[0]["sequence"] == record["sequence"] and
                 headers[0]["color_bytes"] == WIDTH * HEIGHT * color_bytes and
                 headers[0]["depth_format"] == (2 if expected_depth is not None else 0) and
                 headers[0]["depth_bytes"] == (WIDTH * HEIGHT * 4 if expected_depth is not None else 0),
                 "actual native request is not the 44-byte typed MGCHE04 protocol")
        capture_path = next((output / stage / "captures").glob("*.json"))
        _require(headers[0]["json_bytes"] == capture_path.stat().st_size and
                 headers[0]["json_sha256"] == _sha256(capture_path),
                 "actual received native JSON differs byte-for-byte from the fresh C capture")
        _require(len(replies) == 1 and replies[0]["magic"] == "MGCRG04" and replies[0]["bytes"] == 44 and
                 replies[0]["status"] == 0 and all(replies[0][name] == headers[0][name] for name in (
                     "sequence", "width", "height", "color_format", "depth_format", "color_bytes", "depth_bytes")) and
                 replies[0]["payload_bytes"] == headers[0]["color_bytes"] + headers[0]["depth_bytes"] and
                 replies[0]["payload_sha256"] == hashlib.sha256(record["native_color"] + (record["depth"] or b"")).hexdigest(),
                 "actual transmitted native44 reply header/payload differs from the committed executor record")
        evidence["checks"]["native_v4_ipc_committed"] = {"passed": True, "peer_pid": record["peer_pid"],
            "sequence": record["sequence"], "request_magic": "MGCHE04", "reply_magic_checked_by_C": "MGCRG04", "bytes": 44}
        evidence["checks"]["native_vs_oracle"] = _compare(native, expected, case.tolerance, "native actual glReadPixels")
        if case.texture_version == 14:
            evidence["comparisons"]["native_line_tap_vs_softpipe_ewa"] = _compare(native, reference, case.tolerance,
                                                                                "native versus EWA", enforce=False)
        else:
            evidence["checks"]["native_vs_softpipe"] = _compare(native, reference, case.tolerance, "native versus real softpipe")
        evidence["checks"]["native_vs_managed"] = _compare(native, managed, case.tolerance, "native versus managed")
        _compare(record["rgba"], managed, 0, "native executor versus independent managed draw")
        if expected_depth is not None:
            evidence["checks"]["native_depth_stencil"] = _compare(native_depth, expected_depth, 1, "native GL packed Z24S8")
            _require(record["depth"] == bytes(depth_output), "native committed low24-depth/high8-stencil differs from managed bytes")
        _require(record["color_format"] == case.state.color_format and
                 record["depth_format"] == case.state.depth_stencil_format, "native executor attachment tags differ from real GL state")
        _require(len(record["native_color"]) == WIDTH * HEIGHT * color_bytes, "native executor typed color length mismatch")
        for phase in ("managed", "native"):
            submissions = [submission for submission in evidence["submissions"] if submission["phase"] == phase]
            _require(len(submissions) >= 2 and all(submission["format"] == ProgramFormat.LEGACY.value for submission in submissions),
                     f"{phase}: real managed VS and FS submissions required")
            _require({submission["program_sha256"] for submission in submissions} ==
                     set(evidence["compiled_program_sha256"].values()),
                     f"{phase}: submissions must execute both exact fresh compiler VS/FS programs")
            if case.derivatives:
                _require(any(submission.get("quad_lanes") == 4 for submission in submissions),
                         f"{phase}: dFdx/dFdy/implicit LOD must execute the real four-lane QuadSimulationBackend")
                _require(all(count > 0 for count in evidence["quad_derivative_barriers"][phase].values()),
                         f"{phase}: actual synchronized dx/dy mailbox barriers must be serviced")
        _require(_build_identity(build) == identity, "shared Mesa build changed during acceptance; rerun after parent freeze")
        evidence.update(passed=True, status="passed", native_gl_readback=True,
                        shader_abi_version=artifact["shader_abi_version"], capture_version=4,
                        layout_version=artifact["shaders"]["vs"]["layout"]["version"],
                        covered_pixels=int(np.count_nonzero(coverage)), host_steps=managed_report["host_steps"])
        return evidence
    except FrontendBlocked as error:
        _require(not evidence["submissions"] and not evidence["native_request_headers"],
                 "frontend-blocked builtins must never claim managed/native GL shader execution")
        _require(_build_identity(build) == identity, "shared Mesa changed during actual frontend capability probe")
        evidence.update(status="frontend-blocked", scope="actual GLSL frontend fail-closed probe; no VertexID GL execution",
                        frontend={"actual_glsl_version": error.version, "required_glsl_version": 130,
                                  "actual_gl_version": error.gl_version, "advertised_extensions": error.extensions,
                                  "compiler_exit_code": error.exit_code, "aborted": False,
                                  "blocked_builtins": list(case.builtins), "builtin_gl_execution": False,
                                  "shader_rejected": True, "vertexid_gl_execution": False,
                                  "log": str(error.log)})
        evidence["checks"]["frontend_fail_closed"] = {"passed": True, "required_glsl_version": 130,
                                                       "actual_glsl_version": error.version}
        return evidence
    except BaseException as error:
        evidence.update(status="failed", failed_stage=stage, error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc())
        if server is not None:
            evidence["native_errors"] = list(server.errors)
        raise AcceptanceError(f"{case.name} failed at {stage}: {error}; report: {report_path}") from error
    finally:
        evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--suite", choices=("abi4", "anisotropy-v14"), default="abi4")
    parser.add_argument("--case", action="append", choices=tuple(ALL_CASES))
    parser.add_argument("--probe-optional-frontends", action="store_true",
                        help="record optional VertexID/BaseInstance frontend probes separately; blocked is never an execution pass")
    parser.add_argument("--reference-only", action="store_true", help="preflight real softpipe while shared Mesa is unfrozen; never acceptance passed")
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve() if arguments.output else Path(tempfile.mkdtemp(prefix="mgc-abi4-"))
    output.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "scope": "bounded ABI4 software-host acceptance", "fpga_execution": False,
              "not_fpga": True, "reference_only": arguments.reference_only, "cases": [], "output": str(output),
              "optional_frontend_probes": [], "optional_cases": [case.name for case in OPTIONAL_CASES]}
    names = arguments.case or [case.name for case in (ANISOTROPY_CASES if arguments.suite == "anisotropy-v14" else CASES)]
    anisotropic = [ALL_CASES[name].texture_version == 14 for name in names]
    if any(anisotropic) and not all(anisotropic):
        parser.error("v14 line-tap acceptance must not be mixed into the ABI4 baseline report")
    if arguments.suite == "anisotropy-v14" and not all(anisotropic):
        parser.error("the v14 suite accepts only its independent line-tap cases")
    if any(anisotropic) and arguments.probe_optional_frontends:
        parser.error("optional builtin frontend probes belong to the separate ABI4 baseline scope")
    report["suite"] = "anisotropy-v14" if all(anisotropic) else "abi4"
    if all(anisotropic):
        report.update(scope="bounded v14 max4 uniformly spaced line-tap/trilinear acceptance; NOT Softpipe EWA",
                      optional_cases=[], softpipe_ewa_is_universal_oracle=False)
    report["requested_cases"] = names
    report["complete_default_suite"] = set(names) == {case.name for case in CASES}
    report["complete_v14_suite"] = set(names) == {case.name for case in ANISOTROPY_CASES}
    report["requested_frontend_probes"] = ([case.name for case in OPTIONAL_CASES if case.name not in names]
                                          if arguments.probe_optional_frontends and not arguments.reference_only else [])
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        identity = _build_identity(arguments.build.resolve())
        report["build_identity"] = identity
        binary = arguments.harness or build_private_harness(arguments.build, output / "private")
        for name in (*names, *report["requested_frontend_probes"]):
            _require(_build_identity(arguments.build.resolve()) == identity,
                     "shared Mesa changed between cases; aggregate acceptance requires one frozen build")
            destination = ("optional_frontend_probes" if name in report["requested_frontend_probes"] else "cases")
            try:
                directory = output / "optional" / name if destination == "optional_frontend_probes" else output / name
                result = run_case(ALL_CASES[name], arguments.build, binary, directory,
                                  reference_only=arguments.reference_only)
            except AcceptanceError as error:
                result_path = directory / "acceptance.json"
                result = json.loads(result_path.read_text()) if result_path.is_file() else {
                    "case": name, "passed": False, "status": "failed", "error": str(error)}
            report[destination].append(result)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        _require(_build_identity(arguments.build.resolve()) == identity,
                 "shared Mesa changed during aggregate acceptance")
        probes_ok = all(case["status"] in ("passed", "frontend-blocked") for case in report["optional_frontend_probes"])
        report["passed"] = (bool(report["cases"]) and all(case["passed"] for case in report["cases"])
                            and probes_ok and not arguments.reference_only)
        report["accepted_execution_cases"] = [case["case"] for case in report["cases"] if case["passed"]]
        report["frontend_blocked_cases"] = [case["case"] for case in (
            *report["cases"], *report["optional_frontend_probes"]) if case["status"] == "frontend-blocked"]
    except (AcceptanceError, OSError, subprocess.SubprocessError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "scope": report["scope"], "not_fpga": True,
                      "reference_only": arguments.reference_only, "report": str(report_path),
                      "cases": [{"case": case["case"], "passed": case["passed"], "status": case["status"],
                                 "failed_stage": case.get("failed_stage")} for case in report["cases"]],
                      "optional_frontend_probes": [{"case": case["case"], "passed": case["passed"],
                                                    "status": case["status"]}
                                                   for case in report["optional_frontend_probes"]]}, indent=2))
    preflight_ok = arguments.reference_only and bool(report["cases"]) and all(
        case["status"] == "reference-preflight-only" for case in report["cases"])
    return 0 if report["passed"] or preflight_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
