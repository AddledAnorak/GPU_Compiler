import hashlib
import json
import math
import struct
import subprocess
import sys

import pytest

from tools import teapot_mesh as teapot


@pytest.fixture(scope="module", params=range(1, 5))
def mesh(request):
    payload, metadata = teapot.generate_mesh(request.param)
    return request.param, payload, metadata, tuple(struct.iter_unpack("<7f", payload))


@pytest.mark.parametrize("parameter", [0.0, 0.125, 1 / 3, 0.5, 0.9, 1.0])
def test_bernstein_partition_and_derivative(parameter):
    basis, derivative = teapot.bernstein3(parameter)
    assert sum(basis) == pytest.approx(1.0)
    assert sum(derivative) == pytest.approx(0.0, abs=1e-14)
    assert all(value >= 0 for value in basis)
    if parameter in (0, 1):
        expected = (1, 0, 0, 0) if parameter == 0 else (0, 0, 0, 1)
        assert basis == expected


@pytest.mark.parametrize("parameter", [-0.01, 1.01, math.inf, -math.inf, math.nan])
def test_bezier_parameter_domain_is_bounded(parameter):
    with pytest.raises(ValueError, match="finite and in"):
        teapot.bernstein3(parameter)


@pytest.mark.parametrize("parameter_u,parameter_v", [(0, 0), (0, 1), (1, 0), (1, 1), (0.25, 0.6), (0.7, 0.4)])
def test_analytic_bicubic_position_derivatives_and_normal(parameter_u, parameter_v):
    quadratic_coefficients = (0.0, 0.0, 1 / 3, 1.0)
    cubic_coefficients = (0.0, 0.0, 0.0, 1.0)
    control_points = tuple(
        tuple(
            (row / 3, column / 3, quadratic_coefficients[row] + cubic_coefficients[column] + row * column / 9)
            for column in range(4)
        )
        for row in range(4)
    )
    position, derivative_u, derivative_v = teapot.evaluate_bezier(control_points, parameter_u, parameter_v)
    slope_u = 2 * parameter_u + parameter_v
    slope_v = 3 * parameter_v**2 + parameter_u
    assert position == pytest.approx((parameter_u, parameter_v, parameter_u**2 + parameter_v**3 + parameter_u * parameter_v))
    assert derivative_u == pytest.approx((1, 0, slope_u))
    assert derivative_v == pytest.approx((0, 1, slope_v))
    sampled_position, normal = teapot.sample_patch(control_points, parameter_u, parameter_v)
    magnitude = math.sqrt(slope_u**2 + slope_v**2 + 1)
    assert sampled_position == position
    assert normal == pytest.approx((slope_u / magnitude, slope_v / magnitude, -1 / magnitude))


def test_undefined_nonpole_normal_is_not_fabricated():
    collapsed_patch = tuple(tuple((1, 2, 3) for column in range(4)) for row in range(4))
    with pytest.raises(ValueError, match="undefined Bezier normal"):
        teapot.sample_patch(collapsed_patch, 0.5, 0.5)


def test_source_and_license_bytes_are_pinned():
    source = teapot.SOURCE_PATH.read_bytes()
    license_notice = teapot.LICENSE_PATH.read_bytes()
    assert len(source) == 49914
    assert hashlib.sha256(source).hexdigest() == "e0ae6c85628d5e2cf04c937c1dccd1d518188e76b194da007f3d14e993d7349e"
    assert hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest() == "15453442c9df89f7b6c3f733e4d32cb0f3232714"
    assert len(license_notice) == 1439
    assert hashlib.sha256(license_notice).hexdigest() == "b6593d5ec4c113a274abb85b10e8615895cb0ddb89f7912af5fe5aa8df38a275"
    assert b"Permission is hereby granted" in source
    assert b"Martin Newell's teapot" in source
    assert b"teasetorig.gz" in source
    assert b"Copyright (c) 1999-2000 Pawel W. Olszta" in license_notice


def test_changed_source_is_rejected_before_tessellation(tmp_path):
    altered_source = tmp_path / "altered.h"
    altered_source.write_bytes(teapot.SOURCE_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        teapot.load_patches(altered_source)


def test_changed_license_is_rejected(tmp_path, monkeypatch):
    altered_license = tmp_path / "COPYING"
    altered_license.write_bytes(teapot.LICENSE_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(teapot, "LICENSE_PATH", altered_license)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        teapot.generate_mesh()


def test_canonical_patch_count_and_freeglut_coordinate_transform():
    patches = teapot.load_patches()
    assert len(patches) == 10
    assert all(len(patch) == 4 and all(len(row) == 4 for row in patch) for patch in patches)
    assert patches[0][0][0] == pytest.approx((0.7, 0.4125, 0.0))
    assert patches[3][0][0] == pytest.approx((0.0, 0.7875, 0.0))
    assert patches[5][0][0] == pytest.approx((0.0, -0.7875, 0.0))
    expanded = tuple(teapot.expanded_patches(patches))
    assert len(expanded) == 32
    assert [sum(patch_index == original for patch_index, copy_index, patch in expanded) for original in range(10)] == [4] * 6 + [2] * 4


def test_official_rotation_reflection_and_normal_orientation():
    patches = teapot.load_patches()
    for patch_index, copy_index, patch in teapot.expanded_patches(patches):
        parameter_u, parameter_v = 0.37, 0.61
        mirrored = patch_index >= 6 and copy_index == 1
        original_position, original_normal = teapot.sample_patch(
            patches[patch_index], 1 - parameter_u if mirrored else parameter_u, parameter_v,
        )
        position, normal = teapot.sample_patch(patch, parameter_u, parameter_v)
        if mirrored:
            expected_position = (original_position[0], original_position[1], -original_position[2])
            expected_normal = (original_normal[0], original_normal[1], -original_normal[2])
        else:
            def expected_rotation(vector):
                return (
                    vector,
                    (vector[2], vector[1], -vector[0]),
                    (-vector[0], vector[1], -vector[2]),
                    (-vector[2], vector[1], vector[0]),
                )[copy_index]
            expected_position = expected_rotation(original_position)
            expected_normal = expected_rotation(original_normal)
        assert position == pytest.approx(expected_position)
        assert normal == pytest.approx(expected_normal)
        if patch_index in (1, 2):
            assert position[0] * normal[0] + position[2] * normal[2] > 0


@pytest.mark.parametrize("patch_index,normal_fix,direction", [(3, 1, 1.0), (5, 2, -1.0)])
def test_official_pole_normals_are_finite_and_rotate(patch_index, normal_fix, direction):
    for original_index, copy_index, patch in teapot.expanded_patches(teapot.load_patches()):
        if original_index != patch_index:
            continue
        for parameter_v in (0, 1 / 3, 2 / 3, 1):
            position, normal = teapot.sample_patch(patch, 0, parameter_v, normal_fix)
            assert all(math.isfinite(component) for component in position)
            assert normal == (0.0, direction, 0.0)
            assert teapot.transform_normal(normal) == pytest.approx(
                (0, direction * math.cos(math.radians(-15)), direction * math.sin(math.radians(-15)))
            )


def test_view_rotation_position_translation_and_normal_transform():
    position = (0.3, -0.2, 0.4)
    normal = (1 / math.sqrt(2), 0, 1 / math.sqrt(2))

    def independent_rotation(vector):
        yaw = math.radians(-20)
        pitch = math.radians(-15)
        yaw_horizontal = math.cos(yaw) * vector[0] + math.sin(yaw) * vector[2]
        yaw_depth = -math.sin(yaw) * vector[0] + math.cos(yaw) * vector[2]
        return (
            yaw_horizontal,
            math.cos(pitch) * vector[1] - math.sin(pitch) * yaw_depth,
            math.sin(pitch) * vector[1] + math.cos(pitch) * yaw_depth,
        )

    rotated_position = independent_rotation(position)
    assert teapot.transform_position(position) == pytest.approx(
        (0.5 * rotated_position[0] - 0.04, 0.5 * rotated_position[1] - 0.04, 0.5 * rotated_position[2], 1)
    )
    assert teapot.transform_normal(normal) == pytest.approx(independent_rotation(normal))
    assert teapot.transform_position((0, 0, 0)) == (-0.04, -0.04, 0, 1)
    for row in range(3):
        for column in range(3):
            dot = sum(teapot.VIEW_ROTATION[row][axis] * teapot.VIEW_ROTATION[column][axis] for axis in range(3))
            assert dot == pytest.approx(1 if row == column else 0, abs=1e-14)


def test_entire_control_hull_is_inside_clip_bounds():
    for patch_index, copy_index, patch in teapot.expanded_patches(teapot.load_patches()):
        for row in patch:
            for control_point in row:
                position = teapot.transform_position(control_point)
                assert position[3] == 1
                assert all(abs(component) < 0.85 for component in position[:3])


def test_bounded_triangle_count_and_binary_layout(mesh):
    subdivisions, payload, metadata, records = mesh
    assert teapot.RECORD.size == 28
    assert len(records) == 32 * 6 * subdivisions**2 <= 4096
    assert len(payload) == len(records) * 28
    assert metadata["vertex_count"] == len(records)
    assert metadata["triangle_count"] == 32 * 2 * subdivisions**2
    assert metadata["byte_count"] == len(payload)
    assert metadata["record_format"] == "records7float32"
    assert metadata["byte_order"] == "little"
    assert metadata["stride_bytes"] == 28
    assert metadata["primitive"] == "triangles" and metadata["indexed"] is False
    assert metadata["attributes"] == {"position": {"offset_bytes": 0, "components": 4}, "normal": {"offset_bytes": 16, "components": 3}}
    assert payload[12:16] == struct.pack("<f", 1.0)
    assert metadata["mesh_sha256"] == hashlib.sha256(payload).hexdigest()


def test_every_float_is_finite_positions_bounded_and_normals_unit(mesh):
    subdivisions, payload, metadata, records = mesh
    for record in records:
        assert len(record) == 7 and all(math.isfinite(component) for component in record)
        assert record[3] == 1
        assert all(-0.85 <= component <= 0.85 for component in record[:3])
        assert math.sqrt(sum(component**2 for component in record[4:])) == pytest.approx(1, abs=1e-7)
    assert metadata["position_bounds"]["min"] == [min(record[axis] for record in records) for axis in range(3)]
    assert metadata["position_bounds"]["max"] == [max(record[axis] for record in records) for axis in range(3)]


def test_mesh_and_metadata_are_deterministic(mesh):
    subdivisions, payload, metadata, records = mesh
    repeated_payload, repeated_metadata = teapot.generate_mesh(subdivisions)
    assert repeated_payload == payload
    assert json.dumps(repeated_metadata, sort_keys=True) == json.dumps(metadata, sort_keys=True)
    assert metadata["required_shader_uniforms"] == {"u_model": "identity", "u_mvp": "identity"}
    assert metadata["host_mesh_demo"] is True
    assert metadata["physical_gpu_execution"] is False
    assert metadata["original_sample_geometry_validation"] is False
    assert metadata["source"]["source_sha256"] == teapot.SOURCE_SHA256
    assert metadata["source"]["revision"] == "566cd459201ae29003ef5686c06506d8646e55c6"


def test_triangle_order_matches_official_differential_normals():
    payload, metadata = teapot.generate_mesh()
    subdivisions = metadata["subdivisions"]
    records = tuple(struct.iter_unpack("<7f", payload))
    for patch_index in range(4, 12):
        start = patch_index * 6 * subdivisions**2
        for offset in range(start, start + 6 * subdivisions**2, 3):
            triangle = records[offset:offset + 3]
            edge_first = tuple(triangle[1][axis] - triangle[0][axis] for axis in range(3))
            edge_second = tuple(triangle[2][axis] - triangle[0][axis] for axis in range(3))
            face_normal = teapot.cross(edge_first, edge_second)
            averaged_normal = tuple(sum(vertex[axis + 4] for vertex in triangle) for axis in range(3))
            assert sum(face_normal[axis] * averaged_normal[axis] for axis in range(3)) > 0
    first_patch = teapot.load_patches()[0]
    expected_parameters = ((0, 0), (1 / 3, 1 / 3), (1 / 3, 0), (0, 0), (0, 1 / 3), (1 / 3, 1 / 3))
    for record, parameters in zip(records[:6], expected_parameters):
        position, normal = teapot.sample_patch(first_patch, *parameters)
        expected = struct.unpack("<7f", struct.pack("<7f", *teapot.transform_position(position), *teapot.transform_normal(normal)))
        assert record == expected


@pytest.mark.parametrize("subdivisions", [0, -1, 5, 1000, True, 3.0, "3", None])
def test_invalid_subdivisions_fail_closed(subdivisions):
    with pytest.raises(ValueError, match="subdivisions must be"):
        teapot.generate_mesh(subdivisions)


def test_standalone_cli_default_output_and_sidecar(tmp_path):
    output = tmp_path / "nested" / "teapot.bin"
    completed = subprocess.run(
        [sys.executable, str(teapot.ROOT / "tools/teapot_mesh.py"), "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    payload, metadata = teapot.generate_mesh()
    assert output.read_bytes() == payload
    assert json.loads((tmp_path / "nested" / "teapot.bin.json").read_text()) == json.loads(json.dumps(metadata))
    assert "HOST teapot: 1728 vertices, 48384 bytes" in completed.stdout


def test_cli_explicit_metadata_and_maximum_subdivisions(tmp_path):
    output = tmp_path / "maximum.bin"
    metadata_path = tmp_path / "metadata" / "maximum.json"
    assert teapot.main(["--output", str(output), "--metadata", str(metadata_path), "--subdivisions", "4"]) == 0
    metadata = json.loads(metadata_path.read_text())
    assert output.stat().st_size == 86016
    assert metadata["vertex_count"] == 3072 and metadata["subdivisions"] == 4


@pytest.mark.parametrize("subdivisions", ["0", "-1", "5", "3.5"])
def test_cli_invalid_subdivisions_write_nothing(tmp_path, subdivisions):
    output = tmp_path / "must-not-exist" / "teapot.bin"
    with pytest.raises(SystemExit) as raised:
        teapot.main(["--output", str(output), "--subdivisions", subdivisions])
    assert raised.value.code == 2
    assert not output.parent.exists()


@pytest.mark.parametrize("protected_path", [teapot.SOURCE_PATH, teapot.LICENSE_PATH])
@pytest.mark.parametrize("protected_argument", ["--output", "--metadata"])
def test_cli_does_not_overwrite_verbatim_upstream_assets(tmp_path, protected_path, protected_argument):
    arguments = {"--output": tmp_path / "mesh.bin", "--metadata": tmp_path / "mesh.json"}
    arguments[protected_argument] = protected_path
    with pytest.raises(SystemExit) as raised:
        teapot.main(["--output", str(arguments["--output"]), "--metadata", str(arguments["--metadata"])])
    assert raised.value.code == 2
    assert not (tmp_path / "mesh.bin").exists()
    assert not (tmp_path / "mesh.json").exists()


def test_cli_output_and_metadata_cannot_alias(tmp_path):
    output = tmp_path / "same-file"
    with pytest.raises(SystemExit) as raised:
        teapot.main(["--output", str(output), "--metadata", str(output)])
    assert raised.value.code == 2 and not output.exists()


def test_default_mesh_reproduces_golden_hashes():
    payload, metadata = teapot.generate_mesh()
    assert hashlib.sha256(payload).hexdigest() == "6c84480234e9689776f7c5e44a9d43960b6a88044c29c7cff56fcf2bd6c6e2b6"
    canonical_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical_metadata).hexdigest() == "799806adad71799cc4bacb83dd79c2c7e07e5d8ec5c80042bc39647fe192caa2"
    assert len(payload) == 48384 and metadata["vertex_count"] == 1728
