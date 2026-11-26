"""Fresh ABI3 instancing composed with mixed-axis and explicit-LOD real GL draws."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import struct
import traceback

import numpy as np
import pytest

from tools import instancing_acceptance as instancing
from tools import mipmap_texture_acceptance as mipmaps
from tools import run_render
from tools.host_executor import HostExecutorServer
from tools.host_render import render_artifact
from tools.host_texture import TEXTURE_BASE, pack_texture
from tools.mgcdriver import Driver, ProgramFormat
from tools.mirrored_texture_acceptance import MAX_DRAW_STEPS
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout
from tools.texture_flow_acceptance import STATUS


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
HARNESS = Path(os.environ.get("MGC_INSTANCING_TEST_HARNESS", ROOT / "tools/mgc_harness"))
INSTANCE_CASES = {case.name: case for case in instancing.CASES}
LEVEL_MASKS = ((1, 1, 0, 1), (0, 1, 1, 1), (1, 0, 1, 0))


@dataclass(frozen=True)
class CompositionCase:
    name: str
    instance_name: str
    texture_filter: str
    wrap_s: str
    wrap_t: str
    lod: int | None = None
    width: int = 3
    height: int = 5

    @property
    def instance(self):
        return replace(INSTANCE_CASES[self.instance_name], blend="disabled")

    @property
    def mode(self):
        return (9 if self.lod is not None else 7) + (self.texture_filter == "linear")


CASES = (
    CompositionCase("mixed7-divisor1-count4", "divisor1-count4", "nearest", "repeat", "mirrored-repeat"),
    CompositionCase("mixed8-offset2-count5", "offset-divisor2-count5", "linear", "clamp-to-edge", "repeat"),
    CompositionCase("mips9-lod1-color2-count5", "color-divisor2-count5", "nearest", "repeat", "clamp-to-edge", 1),
    CompositionCase("mips10-lod1-offset3-color1-count4", "overlap-additive", "linear", "mirrored-repeat", "repeat", 1),
    CompositionCase("mips9-lod4-clamped-offset2-count5", "offset-divisor2-count5", "nearest", "clamp-to-edge", "mirrored-repeat", 4),
    CompositionCase("mips10-lod4-clamped-color3-count5", "color-divisor3-count5", "linear", "mirrored-repeat", "clamp-to-edge", 4),
    CompositionCase("mips9-lod4-1x1-vec4-offset-count4", "vec4-offset-count4", "nearest", "repeat", "mirrored-repeat", 4, 1, 1),
    CompositionCase("mips10-lod4-1x1-color2-count5", "color-divisor2-count5", "linear", "clamp-to-edge", "repeat", 4, 1, 1),
)


def exact_metadata(actual, expected):
    assert type(actual) is dict and set(actual) == set(expected)
    for name, value in expected.items():
        assert type(actual[name]) is type(value) and actual[name] == value, name


def texture_inputs(case):
    dimensions = mipmaps.mipmap_dimensions(case.width, case.height) if case.lod is not None else ((case.width, case.height),)
    masks = (LEVEL_MASKS if len(dimensions) > 1 else
             ((1, 1, 1, 0),) if case.lod is not None else
             (LEVEL_MASKS[0 if case.texture_filter == "nearest" else 2],))
    levels = tuple(np.tile(np.array(mask, np.uint8) * 255, (height, width, 1))
                   for (width, height), mask in zip(dimensions, masks))
    assert len(levels) == len(dimensions)
    assert len({tuple(level[0, 0]) for level in levels}) == len(levels)
    assert all(np.all(level == level[0, 0]) and np.all((level == 0) | (level == 255)) for level in levels)
    return levels


def fresh_shader(directory, case):
    shader = directory / "shader"
    shader.mkdir()
    instance = case.instance
    version = 120 if case.lod is not None else 100
    color = "a_instance_color" if instance.color_components == 4 else "vec4(a_instance_color, 1.0)"
    (shader / "vert.glsl").write_text(
        f"#version {version}\nattribute vec4 a_position;\n"
        f"attribute vec{instance.offset_components} a_instance_offset;\n"
        f"attribute vec{instance.color_components} a_instance_color;\n"
        "varying vec4 v_color;\nvarying vec2 v_uv;\nvoid main() {\n"
        "gl_Position = a_position + vec4(a_instance_offset.xyz, 0.0);\n"
        f"v_color = {color};\nv_uv = a_position.xy * 12.0 + vec2(-1.0625, 2.125);\n}}\n")
    extension = "#extension GL_ARB_shader_texture_lod : require\n" if case.lod is not None else "precision mediump float;\n"
    sample = f"texture2DLod(u_texture, v_uv, {case.lod}.0)" if case.lod is not None else "texture2D(u_texture, v_uv)"
    (shader / "frag.glsl").write_text(
        f"#version {version}\n{extension}uniform sampler2D u_texture;\n"
        "varying vec4 v_color;\nvarying vec2 v_uv;\n"
        f"void main() {{\ngl_FragColor = v_color * {sample};\n}}\n")
    return shader


def expected_texture_contract(case):
    contract = {"version": case.mode, "texture": 0, "sampler": 0, "storage_block": 15,
                "max_width": 16, "max_height": 16, "filter": case.texture_filter,
                "wrap_s": case.wrap_s, "wrap_t": case.wrap_t}
    if case.lod is not None:
        contract.update(lod=case.lod, mip_filter="nearest", max_levels=5)
    return contract


def assert_capture(artifact, case, levels, mesh_bytes):
    instance = case.instance
    assert artifact["schema"] == "mgc-host-draw-v3"
    assert artifact["program_format"] == ProgramFormat.LEGACY.value
    assert type(artifact["shader_abi_version"]) is int and artifact["shader_abi_version"] == 3
    exact_metadata(artifact["host_instancing"], {
        "version": 1, "max_instances": 16, "max_vertices": 4096,
        "indexing": "base-instance-plus-floor-local-instance-over-divisor",
        "shader_abi_version": 3, "instance_id_word": 65535})
    draw = artifact["draw"]
    assert draw["mode"] == "triangles" and draw["count"] == 6
    assert draw["instance_count"] == instance.count and draw.get("start_instance", 0) == 0
    assert draw.get("index_size", 0) == 0 and artifact.get("indices") in (None, "")
    vertex = ShaderLayout(artifact["shaders"]["vs"]["layout"])
    fragment = ShaderLayout(artifact["shaders"]["fs"]["layout"])
    for layout, stage in ((vertex, "vertex"), (fragment, "fragment")):
        assert layout.source["stage"] == stage and layout.source["version"] == 3
        assert layout.source["shader_abi_version"] == 3 and layout.target is ProgramFormat.LEGACY
        exact_metadata(layout.host_instancing, {
            "version": 1, "max_instances": 16, "instance_id_word": 65535,
            "instance_id": "local-zero-based", "uses_instance_id": False})
        assert not layout.source.get("uses_discard", False)
    assert vertex.host_texture is None and vertex.fragment_status is None
    exact_metadata(fragment.fragment_status, STATUS)
    exact_metadata(fragment.host_texture, expected_texture_contract(case))
    inputs = vertex.entries("inputs")
    positions = {entry["location"] for entry in inputs if entry["word_offset"] == 0}
    assert len(positions) == 1
    position = next(iter(positions))
    offsets, colors = instancing.instance_records(instance)
    for binding, divisor, components, records in (
            (0, 0, 4, None), (3, instance.offset_divisor, instance.offset_components, offsets),
            (4, instance.color_divisor, instance.color_components, colors)):
        slots = {entry["word_offset"] // 4 for entry in inputs if entry["location"] == position + binding}
        assert len(slots) == 1
        element = artifact["vertex_elements"][next(iter(slots))]
        buffer = artifact["vertex_buffers"][element["vertex_buffer_index"]]
        assert element.get("instance_divisor", 0) == divisor and element["src_offset"] == 0
        assert buffer["stride"] == (40 if records is None else components * 4)
        expected_bytes = b"".join(struct.pack("<10f", *record[:4], 1, 0.5, 0.25, *record[4:])
                                  for record in struct.iter_unpack("<7f", mesh_bytes)) if records is None else b"".join(
            struct.pack("<" + "f" * len(record), *record) for record in records)
        assert bytes.fromhex(buffer["data_hex"]) == expected_bytes
    sampler = {"normalized_coords": True, "min_filter": case.texture_filter, "mag_filter": case.texture_filter,
               "mip_filter": "nearest" if case.lod is not None else "none",
               "wrap_s": case.wrap_s, "wrap_t": case.wrap_t}
    expected_texture = {"width": case.width, "height": case.height, "format": "R8G8B8A8_UNORM",
                        "data_hex": levels[0].tobytes().hex(), "sampler": sampler}
    if case.lod is not None:
        expected_texture["mipmaps"] = [
            {"level": index, "width": level.shape[1], "height": level.shape[0], "data_hex": level.tobytes().hex()}
            for index, level in enumerate(levels) if index]
    assert artifact["textures"] == [expected_texture]
    selected = levels[0 if case.lod is None else min(case.lod, len(levels) - 1)]
    scalars = np.array([selected.shape[1], selected.shape[0], 0, 0] +
                       [int(channel) // 255 for channel in selected.reshape(-1)], np.float16).view(np.uint16)
    assert pack_texture(fragment, artifact["textures"]) == {
        TEXTURE_BASE + index: int(value) for index, value in enumerate(scalars)}


def assert_execution(coverage, report, expected_coverage, case):
    instancing._check_execution(coverage, report, expected_coverage, 0, case.instance)
    assert report["program_format"] == ProgramFormat.LEGACY.value and report["shader_abi_version"] == 3
    assert report["blend"] == "disabled" and report["depth"] == "disabled"
    expected_sampling = (f"bounded legacy {case.texture_filter}-S:{case.wrap_s}/T:{case.wrap_t} v{case.mode}" +
                         (f" explicit integer LOD {case.lod} nearest-mip" if case.lod is not None else ""))
    assert report["texture_sampling"] == expected_sampling


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_real_gl_instancing_mixed_and_mipmap_composition(monkeypatch, tmp_path, case):
    report_path = tmp_path / "composition-report.json"
    evidence = {"case": case.name, "passed": False, "status": "running", "native_gl_readback": False,
                "fpga_execution": False, "build": str(BUILD), "harness": str(HARNESS),
                "texture_mode": case.mode, "source_lod": case.lod,
                "instance_count": case.instance.count, "offset_divisor": case.instance.offset_divisor,
                "color_divisor": case.instance.color_divisor,
                "oracle": "acceptance Fraction rectangles and commits multiplied by selected uploaded binary mask",
                "logs": {"capture": str(tmp_path / "capture/capture.log"),
                         "native": str(tmp_path / "native/native.log"),
                         "softpipe": str(tmp_path / "reference/softpipe.log")}}
    report_path.write_text(json.dumps(evidence, indent=2) + "\n")
    invocations, submissions = [], []
    original_run, original_submit = run_render.subprocess.run, Driver.submit

    def observed_run(command, **options):
        if command and str(command[0]) == str(HARNESS):
            environment = options["env"]
            action = command[3]
            invocations.append(action)
            assert environment["GALLIUM_DRIVER"] == ("softpipe" if action == "--reference" else "mgc")
            assert environment["MGC_HOST_INSTANCING"] == "1"
            assert environment["MGC_HOST_INSTANCE_DATA"] == str(instances)
            assert environment["MGC_HOST_MESH"] == str(mesh)
            assert environment["MGC_HOST_TEXTURE_DATA"] == str(texture)
            assert environment["MGC_HOST_TEXTURE_FILTER"] == case.texture_filter
            assert environment["MGC_HOST_TEXTURE_WRAP_S"] == case.wrap_s
            assert environment["MGC_HOST_TEXTURE_WRAP_T"] == case.wrap_t
            assert "MGC_HOST_TEXTURE_WRAP" not in environment and "MGC_HOST_BLEND" not in environment
            assert environment.get("MGC_HOST_TEXTURE") == str(case.mode)
            assert environment.get("MGC_HOST_GLSL_PROFILE") == ("desktop-120" if case.lod is not None else None)
            assert "MGC_HOST_MIPMAPS" not in environment
        try:
            return original_run(command, **options)
        except BaseException as error:
            parts = [part.decode("utf-8", errors="replace") if isinstance(part, bytes) else part or ""
                     for part in (getattr(error, "stdout", None), getattr(error, "stderr", None))]
            (tmp_path / "subprocess-exception.log").write_text("".join(parts) + traceback.format_exc())
            raise

    def observed_submit(driver, program, **options):
        submissions.append(program.format.value)
        assert program.format is ProgramFormat.LEGACY
        return original_submit(driver, program, **options)

    server = None
    try:
        assert HARNESS.is_file() and os.access(HARNESS, os.X_OK), "actual live harness required; no rebuild or skip"
        assert all((BUILD / relative).is_file() for relative in
                   ("src/egl/libEGL.so", "src/mapi/es2api/libGLESv2.so")), "actual live Mesa required; no rebuild or skip"
        assert len(os.fsencode(tmp_path / "ex.sock")) <= 107, "run with a short /tmp pytest --basetemp"
        monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
        monkeypatch.setenv("MESA_GLES_VERSION_OVERRIDE", "2.0")
        monkeypatch.setattr(run_render.subprocess, "run", observed_run)
        monkeypatch.setattr(Driver, "submit", observed_submit)
        shader = fresh_shader(tmp_path, case)
        mesh, instances, texture = (tmp_path / name for name in ("vertices.bin", "instances.bin", "texture.bin"))
        mesh_bytes, instance_bytes = instancing.case_mesh(), instancing.instance_bytes(case.instance)
        levels = texture_inputs(case)
        texture_bytes = (mipmaps.encode_mipmaps(levels) if case.lod is not None else
                         struct.pack("<II", case.width, case.height) + levels[0].tobytes())
        mesh.write_bytes(mesh_bytes)
        instances.write_bytes(instance_bytes)
        texture.write_bytes(texture_bytes)
        images, coverages, depths, discarded = instancing.rational_oracle(case.instance)
        assert len(images) == 1 and discarded == [0]
        selected = 0 if case.lod is None else min(case.lod, len(levels) - 1)
        mask = levels[selected][0, 0] // 255
        expected, expected_coverage = images[0] * mask, coverages[0]
        assert len(np.unique(expected[expected_coverage > 0], axis=0)) > 1, "instance colors must remain nonconstant"
        assert not np.array_equal(expected, np.zeros_like(expected))
        for index, level in enumerate(levels):
            if index != selected:
                assert not np.array_equal(expected, images[0] * (level[0, 0] // 255)), "wrong uploaded LOD must be observable"
        expected.tofile(tmp_path / "rational.rgba8")
        expected_coverage.tofile(tmp_path / "rational-coverage.u32")
        evidence.update(selected_level=selected, uploaded_level_masks=[(level[0, 0] // 255).tolist() for level in levels])
        options = {"mesh": mesh, "instance_data": instances, "texture": texture, "blend": "disabled",
                   "texture_filter": case.texture_filter, "texture_wrap_s": case.wrap_s,
                   "texture_wrap_t": case.wrap_t, "mipmaps": case.lod is not None}
        artifact = run_render.capture_draw(shader, BUILD, tmp_path / "capture", host_texture=True, **options)
        assert_capture(artifact, case, levels, mesh_bytes)
        managed, coverage, managed_report = render_artifact(artifact, blend="disabled", max_steps=MAX_DRAW_STEPS)
        managed.tofile(tmp_path / "managed.rgba8")
        coverage.tofile(tmp_path / "managed-coverage.u32")
        (tmp_path / "managed-report.json").write_text(json.dumps(managed_report, indent=2) + "\n")
        assert_execution(coverage, managed_report, expected_coverage, case)
        np.testing.assert_array_equal(managed, expected)
        with HostExecutorServer(tmp_path / "ex.sock", max_requests=1, max_steps=MAX_DRAW_STEPS) as server:
            native = run_render.native_host_readback(shader, BUILD, tmp_path / "native", socket_path=server.path, **options)
        native.tofile(tmp_path / "native-top-left.rgba8")
        assert not server.errors and len(server.records) == 1, server.errors
        record = server.records[0]
        assert_capture(record["artifact"], case, levels, mesh_bytes)
        assert_execution(record["coverage"], record["report"], expected_coverage, case)
        native_report = json.loads((tmp_path / "native/native-report.json").read_text())
        assert native_report["native_gl_readback"] is True and native_report["host_executor_committed"] is True
        assert native_report["fpga_execution"] is False and native_report["repeats"] == 1
        assert native_report["texture_filter"] == case.texture_filter
        assert native_report["texture_wrap_s"] == case.wrap_s and native_report["texture_wrap_t"] == case.wrap_t
        assert native_report.get("mipmaps", False) is (case.lod is not None)
        assert native_report.get("glsl_profile") == ("desktop-120" if case.lod is not None else None)
        assert len(native_report["capture_paths"]) == len(native_report["commits"]) == 1
        assert native_report["commits"][0] == {"sequence": record["sequence"], "width": 16, "height": 16}
        assert json.loads(Path(native_report["capture_paths"][0]).read_text()) == record["artifact"]
        raw = np.frombuffer(Path(native_report["raw_path"]).read_bytes(), np.uint8).reshape(16, 16, 4)
        np.testing.assert_array_equal(raw[::-1], native)
        np.testing.assert_array_equal(record["rgba"], managed)
        np.testing.assert_array_equal(native, managed)
        np.testing.assert_array_equal(native, expected)
        evidence["native_gl_readback"] = True
        reference = run_render.softpipe_reference(shader, BUILD, tmp_path / "reference", **options)
        reference.tofile(tmp_path / "softpipe-top-left.rgba8")
        np.testing.assert_array_equal(reference, expected)
        np.testing.assert_array_equal(reference, managed)
        np.testing.assert_array_equal(reference, native)
        assert invocations == ["--draw", "--host-execute", "--reference"]
        assert len(submissions) == 2 * (case.instance.count * 6 + int(expected_coverage.sum()))
        assert set(submissions) == {ProgramFormat.LEGACY.value}
        for action, path, renderer in (("capture", tmp_path / "capture/capture.log", "mgc"),
                                       ("native", tmp_path / "native/native.log", "mgc"),
                                       ("softpipe", tmp_path / "reference/softpipe.log", "softpipe")):
            log = path.read_text()
            assert f"GL_RENDERER: {renderer}\n" in log, action
            alias = "ARB" if case.lod is not None else "EXT"
            event = (f"MGC_HOST_INSTANCING_GL alias={alias} count={case.instance.count} "
                     f"offset_divisor={case.instance.offset_divisor} color_divisor={case.instance.color_divisor} base_instance=0\n")
            assert log.count(event) == 1, path
            assert "MGC_HOST_EXECUTOR_FAILURE" not in log and "MGC_MIPMAP_EXTENSION_UNAVAILABLE" not in log
            if case.lod is not None:
                assert "MGC_HOST_GLSL_PROFILE: desktop-120; GL_VERSION:" in log and "OpenGL ES" not in log
            else:
                assert "MGC_HOST_GLSL_PROFILE:" not in log
        assert mesh.read_bytes() == mesh_bytes and instances.read_bytes() == instance_bytes and texture.read_bytes() == texture_bytes
        evidence.update(passed=True, status="passed", exact_native_managed_rational_softpipe=True,
                        invocations=invocations, submission_formats=submissions,
                        committed_fragments=int(expected_coverage.sum()), covered_pixels=int(np.count_nonzero(expected_coverage)),
                        max_overdraw=int(expected_coverage.max()), managed_report=managed_report)
    except BaseException as error:
        evidence.update(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        (tmp_path / "exception.log").write_text(evidence["traceback"])
        raise
    finally:
        if server is not None:
            evidence["executor_errors"] = list(server.errors)
            evidence["executor_records"] = len(server.records)
            for index, record in enumerate(server.records):
                (tmp_path / f"executor-{index}-report.json").write_text(json.dumps(record["report"], indent=2) + "\n")
                record["rgba"].tofile(tmp_path / f"executor-{index}.rgba8")
                record["coverage"].tofile(tmp_path / f"executor-{index}-coverage.u32")
        evidence["invocations"] = invocations
        evidence["submission_formats"] = submissions
        report_path.write_text(json.dumps(evidence, indent=2) + "\n")
