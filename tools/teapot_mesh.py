"""Generate a bounded HOST triangle list from pinned official freeglut data."""

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "assets/teapot/fg_teapot_data.h"
LICENSE_PATH = ROOT / "assets/teapot/COPYING"
UPSTREAM_REVISION = "566cd459201ae29003ef5686c06506d8646e55c6"
UPSTREAM_TAG = "v3.6.0"
UPSTREAM_ROOT = f"https://raw.githubusercontent.com/freeglut/freeglut/{UPSTREAM_REVISION}"
SOURCE_SHA256 = "e0ae6c85628d5e2cf04c937c1dccd1d518188e76b194da007f3d14e993d7349e"
SOURCE_BLOB_SHA1 = "15453442c9df89f7b6c3f733e4d32cb0f3232714"
LICENSE_SHA256 = "b6593d5ec4c113a274abb85b10e8615895cb0ddb89f7912af5fe5aa8df38a275"
ALGORITHM_SHA256 = "defa77635e6f6b418716672290571406dbd912287d904d87cf69776b672a694d"
DEFAULT_SUBDIVISIONS = 3
MAX_SUBDIVISIONS = 4
PATCH_COUNT = 32
MAX_VERTICES = 4096
RECORD = struct.Struct("<7f")
CLIP_BOUND = 0.85
ORTHOGRAPHIC_SCALE = 0.5
CLIP_TRANSLATION = (-0.04, -0.04, 0.0)
YAW_DEGREES = -20.0
PITCH_DEGREES = -15.0


def validate_subdivisions(subdivisions):
    if type(subdivisions) is not int or not 1 <= subdivisions <= MAX_SUBDIVISIONS:
        raise ValueError(f"subdivisions must be an integer in [1, {MAX_SUBDIVISIONS}]")
    return subdivisions


def verified_bytes(path, expected_sha256):
    contents = Path(path).read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}")
    return contents


def array_rows(source, name, convert):
    declaration = re.search(
        rf"\b{re.escape(name)}\s*(?:\[[^\]]*\])+\s*=\s*\{{(.*?)\n\}};",
        source,
        re.DOTALL,
    )
    if declaration is None:
        raise ValueError(f"missing upstream array {name}")
    body = re.sub(r"/\*.*?\*/", "", declaration.group(1), flags=re.DOTALL)
    return tuple(
        tuple(convert(value.strip().removesuffix("f")) for value in row.split(",") if value.strip())
        for row in re.findall(r"\{([^{}]+)\}", body)
    )


def load_patches(source_path=SOURCE_PATH):
    source = verified_bytes(source_path, SOURCE_SHA256).decode("utf-8")
    verified_bytes(LICENSE_PATH, LICENSE_SHA256)
    indices = array_rows(source, "patchdata_teapot", int)
    points = array_rows(source, "cpdata_teapot", float)
    if len(indices) != 10 or any(len(row) != 16 for row in indices):
        raise ValueError("expected ten canonical teapot patches with sixteen indices each")
    if len(points) != 129 or any(len(point) != 3 for point in points):
        raise ValueError("expected 129 canonical teapot control points")
    canonical = tuple((point[0] / 2, (point[2] - 1.575) / 2, -point[1] / 2) for point in points)
    return tuple(
        tuple(tuple(canonical[row[row_index * 4 + column]] for column in range(4)) for row_index in range(4))
        for row in indices
    )


def rotate_quadrant(vector, quadrant):
    horizontal, vertical, depth = vector
    return (
        (horizontal, vertical, depth),
        (depth, vertical, -horizontal),
        (-horizontal, vertical, -depth),
        (-depth, vertical, horizontal),
    )[quadrant]


def expanded_patches(patches):
    for patch_index, control_points in enumerate(patches):
        if patch_index < 6:
            for quadrant in range(4):
                yield patch_index, quadrant, tuple(
                    tuple(rotate_quadrant(point, quadrant) for point in row) for row in control_points
                )
        else:
            yield patch_index, 0, control_points
            yield patch_index, 1, tuple(
                tuple((point[0], point[1], -point[2]) for point in row)
                for row in reversed(control_points)
            )


def bernstein3(parameter):
    if not math.isfinite(parameter) or not 0 <= parameter <= 1:
        raise ValueError("Bezier parameters must be finite and in [0, 1]")
    inverse = 1 - parameter
    basis = (inverse**3, 3 * parameter * inverse**2, 3 * parameter**2 * inverse, parameter**3)
    derivative = (
        -3 * inverse**2,
        3 * inverse**2 - 6 * parameter * inverse,
        6 * parameter * inverse - 3 * parameter**2,
        3 * parameter**2,
    )
    return basis, derivative


def evaluate_bezier(control_points, parameter_u, parameter_v):
    basis_u, derivative_u = bernstein3(parameter_u)
    basis_v, derivative_v = bernstein3(parameter_v)
    weights = ((basis_u, basis_v), (derivative_u, basis_v), (basis_u, derivative_v))
    return tuple(
        tuple(
            math.fsum(
                weights_u[row] * weights_v[column] * control_points[row][column][axis]
                for row in range(4) for column in range(4)
            )
            for axis in range(3)
        )
        for weights_u, weights_v in weights
    )


def cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def unit_normal(vector):
    length = math.sqrt(math.fsum(component * component for component in vector))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("undefined Bezier normal outside official pole fixes")
    return tuple(component / length for component in vector)


def sample_patch(control_points, parameter_u, parameter_v, normal_fix=0):
    position, tangent_u, tangent_v = evaluate_bezier(control_points, parameter_u, parameter_v)
    if parameter_u == 0 and normal_fix in (1, 2):
        normal = (0.0, 1.0 if normal_fix == 1 else -1.0, 0.0)
    else:
        normal = unit_normal(cross(tangent_v, tangent_u))
    return position, normal


def view_rotation_matrix():
    sine_yaw = math.sin(math.radians(YAW_DEGREES))
    cosine_yaw = math.cos(math.radians(YAW_DEGREES))
    sine_pitch = math.sin(math.radians(PITCH_DEGREES))
    cosine_pitch = math.cos(math.radians(PITCH_DEGREES))
    return (
        (cosine_yaw, 0.0, sine_yaw),
        (sine_pitch * sine_yaw, cosine_pitch, -sine_pitch * cosine_yaw),
        (-cosine_pitch * sine_yaw, sine_pitch, cosine_pitch * cosine_yaw),
    )


VIEW_ROTATION = view_rotation_matrix()


def rotate_view(vector):
    return tuple(math.fsum(row[axis] * vector[axis] for axis in range(3)) for row in VIEW_ROTATION)


def transform_position(position):
    rotated = rotate_view(position)
    return tuple(ORTHOGRAPHIC_SCALE * rotated[axis] + CLIP_TRANSLATION[axis] for axis in range(3)) + (1.0,)


def transform_normal(normal):
    return unit_normal(rotate_view(normal))


def source_provenance():
    return {
        "attribution": "Martin Newell's classic teapot, official freeglut reduced patch dataset with added bottom",
        "repository": "https://github.com/freeglut/freeglut",
        "tag": UPSTREAM_TAG,
        "revision": UPSTREAM_REVISION,
        "source_path": "assets/teapot/fg_teapot_data.h",
        "source_url": f"{UPSTREAM_ROOT}/src/fg_teapot_data.h",
        "source_sha256": SOURCE_SHA256,
        "source_git_blob_sha1": SOURCE_BLOB_SHA1,
        "source_bytes": 49914,
        "license": "Permissive freeglut notice (MIT/X11-style); retain verbatim header and COPYING",
        "license_path": "assets/teapot/COPYING",
        "license_url": f"{UPSTREAM_ROOT}/COPYING",
        "license_sha256": LICENSE_SHA256,
        "algorithm_reference_url": f"{UPSTREAM_ROOT}/src/fg_teapot.c",
        "algorithm_reference_sha256": ALGORITHM_SHA256,
    }


def generate_mesh(subdivisions=DEFAULT_SUBDIVISIONS):
    subdivisions = validate_subdivisions(subdivisions)
    payload = bytearray()
    for patch_index, copy_index, control_points in expanded_patches(load_patches()):
        normal_fix = 1 if patch_index == 3 else 2 if patch_index == 5 else 0
        grid = []
        for row in range(subdivisions + 1):
            samples = []
            for column in range(subdivisions + 1):
                position, normal = sample_patch(control_points, row / subdivisions, column / subdivisions, normal_fix)
                samples.append(RECORD.pack(*transform_position(position), *transform_normal(normal)))
            grid.append(samples)
        for row in range(subdivisions):
            for column in range(subdivisions):
                payload.extend(grid[row][column])
                payload.extend(grid[row + 1][column + 1])
                payload.extend(grid[row + 1][column])
                payload.extend(grid[row][column])
                payload.extend(grid[row][column + 1])
                payload.extend(grid[row + 1][column + 1])
    payload = bytes(payload)
    records = tuple(RECORD.iter_unpack(payload))
    expected_count = PATCH_COUNT * 6 * subdivisions**2
    if len(records) != expected_count or len(records) > MAX_VERTICES:
        raise ValueError("mesh exceeds the bounded HOST triangle-list contract")
    if any(not all(math.isfinite(component) for component in record) for record in records):
        raise ValueError("mesh contains non-finite float32 values")
    if any(record[3] != 1 or any(abs(component) > CLIP_BOUND for component in record[:3]) for record in records):
        raise ValueError("mesh is outside the declared orthographic clip bounds")
    metadata = {
        "schema": "mgc-host-teapot-mesh-v1",
        "record_format": "records7float32",
        "byte_order": "little",
        "stride_bytes": RECORD.size,
        "attributes": {"position": {"offset_bytes": 0, "components": 4}, "normal": {"offset_bytes": 16, "components": 3}},
        "primitive": "triangles",
        "indexed": False,
        "subdivisions": subdivisions,
        "subdivisions_definition": "cells per parameter axis; subdivisions + 1 samples per axis",
        "base_patches": 10,
        "expanded_patches": PATCH_COUNT,
        "vertex_count": len(records),
        "triangle_count": len(records) // 3,
        "byte_count": len(payload),
        "mesh_sha256": hashlib.sha256(payload).hexdigest(),
        "position_bounds": {
            "min": [min(record[axis] for record in records) for axis in range(3)],
            "max": [max(record[axis] for record in records) for axis in range(3)],
        },
        "transform": {
            "source_to_canonical": "(x/2, (z-1.575)/2, -y/2), matching freeglut scale=1",
            "view_rotation_order": "R_x(pitch) * R_y(yaw), right-handed, column vectors",
            "yaw_degrees": YAW_DEGREES,
            "pitch_degrees": PITCH_DEGREES,
            "view_rotation_matrix": VIEW_ROTATION,
            "orthographic_uniform_scale": ORTHOGRAPHIC_SCALE,
            "clip_translation": CLIP_TRANSLATION,
            "position_w": 1.0,
            "clip_bound": CLIP_BOUND,
            "normal_transform": "normalize(view_rotation_matrix * canonical_normal); no translation",
        },
        "normal_orientation": "normalize(dP/dv cross dP/du), matching official freeglut; lid/bottom first-row pole fixes",
        "winding": "CCW outward; reverse each upstream fg_teapot.c triangle to agree with its outward normals",
        "required_shader_uniforms": {"u_model": "identity", "u_mvp": "identity"},
        "host_mesh_demo": True,
        "physical_gpu_execution": False,
        "original_sample_geometry_validation": False,
        "execution_claim": "Geometry only; parent owns shader execution and Gallium integration. Not Aakash's missing original generator.",
        "source": source_provenance(),
    }
    return payload, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="raw little-endian position vec4 + normal vec3 triangle list")
    parser.add_argument("--subdivisions", type=int, choices=range(1, MAX_SUBDIVISIONS + 1), default=DEFAULT_SUBDIVISIONS)
    parser.add_argument("--metadata", type=Path, help="JSON provenance sidecar; defaults to OUTPUT.json")
    arguments = parser.parse_args(argv)
    metadata_path = arguments.metadata or Path(str(arguments.output) + ".json")
    output_paths = (arguments.output.resolve(), metadata_path.resolve())
    if output_paths[0] == output_paths[1] or any(path in (SOURCE_PATH, LICENSE_PATH) for path in output_paths):
        parser.error("output and metadata must be distinct and must not overwrite upstream assets")
    try:
        payload, metadata = generate_mesh(arguments.subdivisions)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(payload)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"HOST teapot: {metadata['vertex_count']} vertices, {len(payload)} bytes -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
