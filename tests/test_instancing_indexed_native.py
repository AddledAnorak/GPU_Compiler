"""Private real-GL indexed sidecar; the production harness/API remain arrays-only.

Run ``-k 'not real_gl'`` while shared Mesa workers are unfrozen, then run this
entire module against their finished build. Missing libraries/extensions fail,
not skip. Evidence covers the existing Gallium indexed C path and managed CPU
execution/native glReadPixels, never FPGA execution or a new hardware feature.
"""

from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile

import numpy as np
import pytest

from tools import instancing_acceptance as acceptance
from tools import run_render
from tools.host_executor import HostExecutorServer
from tools.host_render import render_artifact
from tools.mgcdriver import Driver, ProgramFormat
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")).resolve()
SOURCE = ROOT / "tools/mgc_harness.c"
INDICES = (3, 4, 5, 0, 1, 2)
INDEX_BYTES = struct.pack("<6H", *INDICES)
CASE_NAMES = ("color-divisor2-count5", "color-divisor3-count5",
              "instance-id1-divisors2-3-count5", "instance-id15-max16",
              "zero-alpha-count4", "depth-kill-repeat")
CASES = tuple(next(case for case in acceptance.CASES if case.name == name) for name in CASE_NAMES)
HOST_METADATA = {"version": 1, "max_instances": 16, "max_vertices": 4096,
                 "indexing": "base-instance-plus-floor-local-instance-over-divisor",
                 "shader_abi_version": 3, "instance_id_word": 65535}
HELPER_ANCHOR = "static int draw_triangle(GLuint program, const char *readback_path, unsigned blend_mode,\n"
DRAW_ANCHOR = "            draw_instanced(GL_TRIANGLES, 0, vertex_count, instances.count);"
DRAW_REPLACEMENT = """            {
                if (!indexed_sidecar_draw(GL_TRIANGLES, vertex_count, instances.count))
                    goto done;
            }"""
INDEXED_EVENT = ("MGC_INDEXED_INSTANCING_GL proc=glDrawElementsInstancedEXT index_size=2 "
                 "indices=3,4,5,0,1,2 vertices=6 instances={count} base_instance=0")
INDEXED_HELPER = r"""
typedef void (GL_APIENTRYP indexed_instance_draw_proc)(GLenum, GLsizei, GLenum, const void *, GLsizei);

static int indexed_sidecar_draw(GLenum mode, GLsizei vertices, GLsizei instances) {
    const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
    const char *version = (const char *)glGetString(GL_VERSION);
    int arrays_ext = extension_advertised(extensions, "GL_EXT_instanced_arrays");
    int draw_ext = extension_advertised(extensions, "GL_EXT_draw_instanced");
    fprintf(stderr, "MGC_INDEXED_GL_API version=%s ext_instanced_arrays=%d ext_draw_instanced=%d\n",
            version ? version : "unavailable", arrays_ext, draw_ext);
    if (!version || strncmp(version, "OpenGL ES ", 10) || !arrays_ext || !draw_ext) {
        fprintf(stderr, "MGC_INDEXED_GL_UNAVAILABLE: requires actual GLES context with advertised "
                        "GL_EXT_instanced_arrays + GL_EXT_draw_instanced; no alias fallback\n");
        return 0;
    }
    indexed_instance_draw_proc draw_elements =
        (indexed_instance_draw_proc)eglGetProcAddress("glDrawElementsInstancedEXT");
    if (!draw_elements) {
        fprintf(stderr, "MGC_INDEXED_GL_UNAVAILABLE: eglGetProcAddress(glDrawElementsInstancedEXT) returned NULL\n");
        return 0;
    }
    if (mode != GL_TRIANGLES || vertices != 6 || instances < 1 || instances > 16) {
        fprintf(stderr, "MGC_INDEXED_GL_BOUND: requires six quad vertices and 1..16 instances\n");
        return 0;
    }
    static const uint16_t quad_indices[6] = {3, 4, 5, 0, 1, 2};
    GLint previous_buffer = 0;
    GLuint element_buffer = 0;
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &previous_buffer);
    glGenBuffers(1, &element_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quad_indices), quad_indices, GL_STATIC_DRAW);
    GLenum error = glGetError();
    if (error == GL_NO_ERROR && element_buffer) {
        draw_elements(mode, 6, GL_UNSIGNED_SHORT, (const void *)0, instances);
        error = glGetError();
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, (GLuint)previous_buffer);
    glDeleteBuffers(1, &element_buffer);
    GLenum cleanup_error = glGetError();
    if (!element_buffer || error != GL_NO_ERROR || cleanup_error != GL_NO_ERROR) {
        fprintf(stderr, "MGC_INDEXED_GL_ERROR: EBO/draw=0x%x cleanup=0x%x buffer=%u\n",
                error, cleanup_error, element_buffer);
        return 0;
    }
    fprintf(stderr, "MGC_INDEXED_INSTANCING_GL proc=glDrawElementsInstancedEXT index_size=2 "
                    "indices=3,4,5,0,1,2 vertices=6 instances=%d base_instance=0\n", instances);
    return 1;
}

"""


def indexed_source(source):
    for anchor in (HELPER_ANCHOR, DRAW_ANCHOR):
        assert source.count(anchor) == 1, f"private harness requires one unique anchor: {anchor!r}"
    return source.replace(HELPER_ANCHOR, INDEXED_HELPER + HELPER_ANCHOR, 1).replace(
        DRAW_ANCHOR, DRAW_REPLACEMENT, 1)


def assert_exact_metadata(actual, expected):
    assert type(actual) is dict and set(actual) == set(expected)
    for name, value in expected.items():
        assert type(actual[name]) is type(value) and actual[name] == value, name


def assert_indexed_contract(artifact, case):
    assert artifact["schema"] == "mgc-host-draw-v3"
    assert artifact["program_format"] == ProgramFormat.LEGACY.value
    assert type(artifact["shader_abi_version"]) is int and artifact["shader_abi_version"] == 3
    assert_exact_metadata(artifact["host_instancing"], HOST_METADATA)
    draw = artifact["draw"]
    expected_draw = {"mode": "triangles", "start": 0, "count": 6, "index_size": 2,
                     "index_bias": 0, "instance_count": case.count, "start_instance": 0,
                     "primitive_restart": False, "restart_index": 0}
    for name, value in expected_draw.items():
        assert type(draw[name]) is type(value) and draw[name] == value, name
    assert 1 <= case.count <= 16 and draw["count"] * draw["instance_count"] <= 4096
    assert type(artifact["indices"]) is str
    assert bytes.fromhex(artifact["indices"]) == INDEX_BYTES
    assert struct.unpack("<6H", bytes.fromhex(artifact["indices"])) == INDICES
    layouts = {}
    for stage, semantic_stage in (("vs", "vertex"), ("fs", "fragment")):
        shader = artifact["shaders"][stage]
        assert shader["assembly"].startswith("; mgc legacy-simulator backend, ABI=3,")
        layout = layouts[stage] = ShaderLayout(shader["layout"])
        assert layout.target is ProgramFormat.LEGACY
        assert layout.source["stage"] == semantic_stage
        assert type(layout.source["version"]) is int and layout.source["version"] == 3
        assert type(layout.source["shader_abi_version"]) is int and layout.source["shader_abi_version"] == 3
        assert layout.host_texture is None
        assert_exact_metadata(layout.source["host_instancing"], {
            "version": 1, "max_instances": 16, "instance_id_word": 65535,
            "instance_id": "local-zero-based",
            "uses_instance_id": stage == "vs" and case.instance_id_compare is not None})
    assert layouts["vs"].fragment_status is None
    assert layouts["fs"].fragment_status == acceptance.STATUS
    assert layouts["fs"].source.get("uses_discard", False) is case.kill
    inputs = layouts["vs"].entries("inputs")
    assert all(entry["word_offset"] != 65535 for entry in inputs)
    position_locations = {entry["location"] for entry in inputs if entry["word_offset"] == 0}
    assert len(position_locations) == 1
    position_location = next(iter(position_locations))
    assert 1 <= len(artifact["vertex_buffers"]) <= 16
    assert 1 <= len(artifact["vertex_elements"]) <= 32
    offsets, colors = acceptance.instance_records(case)
    for binding, divisor, components, records in (
            (0, 0, 4, None), (3, case.offset_divisor, case.offset_components, offsets),
            (4, case.color_divisor, case.color_components, colors)):
        slots = {entry["word_offset"] // 4 for entry in inputs
                 if entry["location"] == position_location + binding}
        assert len(slots) == 1
        element = artifact["vertex_elements"][next(iter(slots))]
        assert type(element["instance_divisor"]) is int and element["instance_divisor"] == divisor
        assert type(element["src_offset"]) is int and element["src_offset"] == 0
        assert element["format"] == {3: "R32G32B32_FLOAT", 4: "R32G32B32A32_FLOAT"}[components]
        buffer = artifact["vertex_buffers"][element["vertex_buffer_index"]]
        contents = bytes.fromhex(buffer["data_hex"])
        if binding:
            assert buffer["stride"] == components * 4
            assert contents == b"".join(struct.pack("<" + "f" * len(record), *record) for record in records)
        else:
            assert buffer["stride"] == 40 and len(contents) == 6 * 40
            mesh = acceptance.case_mesh()
            for vertex in range(6):
                assert contents[vertex * 40:vertex * 40 + 16] == mesh[vertex * 28:vertex * 28 + 16]
    assert sum(len(bytes.fromhex(buffer["data_hex"])) for buffer in artifact["vertex_buffers"]) <= 6 * 40 + 16 * 8 * 4


def test_bounded_indexed_matrix_retains_required_cases():
    assert len(CASES) == 6 and len(set(CASE_NAMES)) == 6
    assert {case.count for case in CASES} == {4, 5, 16}
    assert {case.offset_divisor for case in CASES} | {case.color_divisor for case in CASES} == {1, 2, 3}
    assert {case.instance_id_compare for case in CASES if case.instance_id_compare is not None} == {1, 15}
    assert any(case.zero_alpha for case in CASES)
    assert any(case.kill and case.depth_test and case.repeats == 2 for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_fresh_inputs_and_reordered_quad_share_nonindexed_rational_oracle(tmp_path, case):
    mesh = acceptance.case_mesh()
    assert len(mesh) == 6 * 28
    vertices = tuple(struct.iter_unpack("<7f", mesh))
    triangles = tuple(tuple(vertices[index] for index in INDICES[start:start + 3]) for start in (0, 3))
    assert Counter(triangles) == Counter((vertices[:3], vertices[3:]))
    assert triangles != (vertices[:3], vertices[3:])
    shader = acceptance._shader(case, tmp_path)
    vertex_source = (shader / "vert.glsl").read_text()
    fragment_source = (shader / "frag.glsl").read_text()
    assert vertex_source.startswith("#version 100\n") and fragment_source.startswith("#version 100\n")
    assert "gl_InstanceID" not in fragment_source
    if case.instance_id_compare is not None:
        assert "#extension GL_EXT_draw_instanced : require\n" in vertex_source
        assert f"if (gl_InstanceIDEXT == {case.instance_id_compare})" in vertex_source
    instances = tmp_path / "instances.bin"
    instances.write_bytes(acceptance.instance_bytes(case))
    assert run_render._validate_instance_data(instances) == {
        "instance_count": case.count, "offset_divisor": case.offset_divisor,
        "color_divisor": case.color_divisor, "offset_components": case.offset_components,
        "color_components": case.color_components, "start_instance": 0}
    images, coverages, depths, discarded = acceptance.rational_oracle(case)
    assert len(images) == len(coverages) == len(depths) == len(discarded) == case.repeats
    assert int(coverages[0].sum()) > 0
    if case.instance_id_compare is not None:
        for control in (acceptance.rational_oracle(case, force_instance_id=0)[0][-1],
                        acceptance.rational_oracle(replace(case, instance_id_compare=None))[0][-1]):
            assert not acceptance.image_comparison(images[-1], control, 0)["passed"]
    if case.zero_alpha:
        assert np.count_nonzero(images[-1][:, :, 3]) == 0
        assert int(coverages[0].sum()) == 64
    if case.depth_test:
        assert [int(coverage.sum()) for coverage in coverages] == [16, 0]
        assert discarded == [40, 40]
        assert np.count_nonzero(depths[-1] == 16384) == 8


@pytest.mark.parametrize("anchor", (HELPER_ANCHOR, DRAW_ANCHOR), ids=("helper", "draw"))
@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_private_source_requires_unique_anchors(anchor, mutation):
    source = SOURCE.read_text()
    changed = source.replace(anchor, "", 1) if mutation == "missing" else source + "\n" + anchor
    with pytest.raises(AssertionError, match="one unique anchor"):
        indexed_source(changed)


def test_private_source_changes_only_instanced_call_and_adds_honest_helper():
    source = SOURCE.read_text()
    generated = indexed_source(source)
    assert generated.replace(INDEXED_HELPER, "", 1).replace(DRAW_REPLACEMENT, DRAW_ANCHOR, 1) == source
    assert generated.count('eglGetProcAddress("glDrawElementsInstancedEXT")') == 1
    assert generated.count('eglGetProcAddress("glDrawArraysInstancedEXT")') == source.count(
        'eglGetProcAddress("glDrawArraysInstancedEXT")') == 1
    assert "(GLenum, GLsizei, GLenum, const void *, GLsizei)" in INDEXED_HELPER
    assert "draw_elements(mode, 6, GL_UNSIGNED_SHORT, (const void *)0, instances);" in INDEXED_HELPER
    assert "glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quad_indices), quad_indices, GL_STATIC_DRAW);" in INDEXED_HELPER
    assert "glDrawElementsInstancedANGLE" not in generated
    assert "glDrawArrays(GL_TRIANGLES, 0, vertex_count);" in generated


@pytest.fixture(scope="module")
def indexed_harness(tmp_path_factory):
    directory = tmp_path_factory.mktemp("indexed-instancing-native")
    libraries = (BUILD / "src/egl/libEGL.so", BUILD / "src/mapi/es2api/libGLESv2.so")
    assert all(path.is_file() for path in libraries), f"real EGL/GLES2 libraries required in {BUILD}; no fake/skip"
    source = SOURCE.read_text()
    generated = directory / "mgc_harness_indexed.c"
    generated.write_text(indexed_source(source))
    tools = directory / "tools"
    tools.mkdir()
    binary = tools / "mgc_harness"
    environment = dict(os.environ)
    environment["PKG_CONFIG_PATH"] = os.pathsep.join(
        filter(None, (str(BUILD / "meson-uninstalled"), environment.get("PKG_CONFIG_PATH"))))
    flags = subprocess.run(["pkg-config", "--cflags", "--libs", "egl", "glesv2"],
                           env=environment, capture_output=True, text=True, timeout=30)
    log = directory / "build.log"
    log.write_text(flags.stdout + flags.stderr)
    assert flags.returncode == 0, f"real EGL/GLES2 pkg-config failed; see {log}"
    link_flags = shlex.split(flags.stdout)
    assert "-lEGL" in link_flags and "-lGLESv2" in link_flags
    assert f"-L{libraries[0].parent}" in link_flags and f"-L{libraries[1].parent}" in link_flags
    command = [*shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-Wall", "-Wextra",
               "-Werror=implicit-function-declaration", str(generated), "-o", str(binary), *link_flags, "-lm"]
    compiled = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    log.write_text(shlex.join(command) + "\n" + flags.stdout + flags.stderr + compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, f"private real EGL/GLES2 harness compilation failed; see {log}"
    assert SOURCE.read_text() == source, "production harness must remain untouched"
    return directory


def test_private_harness_links_real_egl_gles2_without_running_gl(indexed_harness):
    assert (indexed_harness / "tools/mgc_harness").is_file()
    assert (indexed_harness / "build.log").is_file()


def assert_draw_evidence(log_path, case, repeats, renderer):
    log = log_path.read_text()
    assert f"GL_RENDERER: {renderer}\n" in log
    assert "MGC_HOST_INSTANCING_GL alias=EXT" in log
    assert log.count("MGC_INDEXED_GL_API version=OpenGL ES ") == repeats, log_path
    assert log.count(INDEXED_EVENT.format(count=case.count) + "\n") == repeats, log_path
    assert log.count("ext_instanced_arrays=1 ext_draw_instanced=1\n") == repeats, log_path
    assert "MGC_INDEXED_GL_UNAVAILABLE" not in log and "MGC_INDEXED_GL_ERROR" not in log


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_real_gl_indexed_capture_managed_native_softpipe_and_nonindexed_oracle(
        indexed_harness, monkeypatch, tmp_path, case):
    report_path = tmp_path / "indexed-report.json"
    result = {"case": case.name, "passed": False, "status": "running", "native_gl_readback": False,
              "fpga_execution": False, "exact_fp32_conformance": False,
              "scope": "private advertised EXT indexed GL sidecar for existing Gallium C path; production harness/API arrays-only",
              "api": "glDrawElementsInstancedEXT", "index_size": 2, "index_bytes": INDEX_BYTES.hex(),
              "contract": HOST_METADATA, "build": str(BUILD), "private_harness": str(indexed_harness),
              "logs": {"build": str(indexed_harness / "build.log"), "capture": str(tmp_path / "capture/capture.log"),
                       "native": str(tmp_path / "native/native.log"), "softpipe": str(tmp_path / "reference/softpipe.log")},
              "oracle": "existing nonindexed six-vertex quad Fraction rectangles, explicit kill and Z16 commits"}
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    submissions, invocations = [], []
    original_submit, original_run = Driver.submit, run_render.subprocess.run

    def observed_submit(driver, program, **options):
        submissions.append(program.format)
        return original_submit(driver, program, **options)

    def observed_run(command, **options):
        if command and str(command[0]) == str(indexed_harness / "tools/mgc_harness"):
            environment = options["env"]
            expected_environment = run_render.mesa_environment(BUILD, tmp_path)
            for name in ("LD_LIBRARY_PATH", "LIBGL_DRIVERS_PATH", "MESA_LOADER_DRIVER_OVERRIDE",
                         "EGL_PLATFORM", "MGC_COMPILE_TARGET"):
                assert environment[name] == expected_environment[name], name
            assert environment["MGC_HOST_INSTANCING"] == "1"
            assert environment["MGC_HOST_INSTANCE_DATA"] == str(tmp_path / "instances.bin")
            assert environment["MGC_HOST_MESH"] == str(tmp_path / "vertices.bin")
            assert list(map(str, command[1:3])) == [str(tmp_path / "shader/vert.glsl"),
                                                    str(tmp_path / "shader/frag.glsl")]
            invocations.append({"driver": environment["GALLIUM_DRIVER"], "mode": command[3],
                                "instance_data": environment["MGC_HOST_INSTANCE_DATA"],
                                "mesh": environment["MGC_HOST_MESH"]})
        return original_run(command, **options)

    monkeypatch.setattr(run_render, "ROOT", indexed_harness)
    monkeypatch.setattr(Driver, "submit", observed_submit)
    monkeypatch.setattr(run_render.subprocess, "run", observed_run)
    try:
        shader = acceptance._shader(case, tmp_path)
        mesh, instances = tmp_path / "vertices.bin", tmp_path / "instances.bin"
        mesh.write_bytes(acceptance.case_mesh())
        instances.write_bytes(acceptance.instance_bytes(case))
        options = {"mesh": mesh, "instance_data": instances, "depth_test": case.depth_test, "blend": case.blend}
        expected_images, expected_coverages, expected_depths, discarded = acceptance.rational_oracle(case)
        artifact = run_render.capture_draw(shader, BUILD, tmp_path / "capture", **options)
        assert_indexed_contract(artifact, case)
        assert_draw_evidence(tmp_path / "capture/capture.log", case, 1, "mgc")
        resources = acceptance.resource_observations(artifact)
        image, depth = None, np.full((16, 16), 65535, np.uint16) if case.depth_test else None
        managed_images, managed_reports = [], []
        for draw_index in range(case.repeats):
            output_depth = np.empty((16, 16), np.uint16) if case.depth_test else None
            image, coverage, report = render_artifact(
                artifact, depth_test=case.depth_test, blend=case.blend, initial_image=image,
                initial_depth=depth, depth_output=output_depth, max_steps=acceptance.MAX_DRAW_STEPS)
            acceptance._check_execution(coverage, report, expected_coverages[draw_index], discarded[draw_index], case)
            assert report["shader_abi_version"] == 3 and report["program_format"] == ProgramFormat.LEGACY.value
            assert np.array_equal(image, expected_images[draw_index])
            if case.depth_test:
                assert np.array_equal(output_depth, expected_depths[draw_index])
            depth = output_depth
            managed_images.append(image.copy())
            managed_reports.append(report)
            (tmp_path / f"coverage-{draw_index + 1}.u32").write_bytes(coverage.astype("<u4").tobytes())
        with tempfile.TemporaryDirectory(prefix="mgc-indexed-") as temporary:
            with HostExecutorServer(Path(temporary) / "executor.sock", max_requests=case.repeats,
                                    max_steps=acceptance.MAX_DRAW_STEPS) as server:
                native = run_render.native_host_readback(
                    shader, BUILD, tmp_path / "native", socket_path=server.path, repeats=case.repeats, **options)
            assert not server.errors and len(server.records) == case.repeats, server.errors
            records = list(server.records)
        proof = json.loads((tmp_path / "native/native-report.json").read_text())
        assert proof["native_gl_readback"] is True and proof["host_executor_committed"] is True
        assert proof["fpga_execution"] is False and proof["repeats"] == case.repeats
        assert len(proof["commits"]) == len(proof["capture_paths"]) == case.repeats
        for draw_index, record in enumerate(records):
            assert_indexed_contract(record["artifact"], case)
            acceptance._check_execution(record["coverage"], record["report"], expected_coverages[draw_index],
                                        discarded[draw_index], case)
            assert np.array_equal(record["rgba"], managed_images[draw_index])
            assert json.loads(Path(proof["capture_paths"][draw_index]).read_text()) == record["artifact"]
            assert proof["commits"][draw_index]["sequence"] == record["sequence"]
            if case.depth_test:
                assert np.array_equal(record["depth"], expected_depths[draw_index])
        assert np.array_equal(native, records[-1]["rgba"])
        raw_native = np.frombuffer((tmp_path / "native/native.rgba8").read_bytes(), np.uint8).reshape(16, 16, 4)
        assert np.array_equal(raw_native[::-1], native)
        assert_draw_evidence(tmp_path / "native/native.log", case, case.repeats, "mgc")
        result.update(native_gl_readback=True, native_capture_paths=proof["capture_paths"],
                      native_commits=proof["commits"])
        reference = run_render.softpipe_reference(shader, BUILD, tmp_path / "reference", repeats=case.repeats, **options)
        assert_draw_evidence(tmp_path / "reference/softpipe.log", case, case.repeats, "softpipe")
        tolerance = 1 if case.blend != "disabled" else 0
        comparisons = {"native_vs_managed": acceptance.image_comparison(native, image, 0),
                       "managed_vs_nonindexed_rational": acceptance.image_comparison(image, expected_images[-1], 0),
                       "native_vs_nonindexed_rational": acceptance.image_comparison(native, expected_images[-1], 0),
                       "softpipe_vs_nonindexed_rational": acceptance.image_comparison(reference, expected_images[-1], tolerance),
                       "native_vs_softpipe": acceptance.image_comparison(native, reference, tolerance)}
        result["comparisons"] = comparisons
        assert all(check["passed"] for check in comparisons.values()), comparisons
        assert submissions and set(submissions) == {ProgramFormat.LEGACY}
        assert [(invocation["driver"], invocation["mode"]) for invocation in invocations] == [
            ("mgc", "--draw"), ("mgc", "--host-execute"), ("softpipe", "--reference")]
        for name, pixels in (("managed", image), ("native", native), ("softpipe", reference),
                             ("rational", expected_images[-1])):
            (tmp_path / f"{name}.rgba8").write_bytes(pixels.tobytes())
        if case.depth_test:
            (tmp_path / "depth.z16").write_bytes(expected_depths[-1].astype("<u2").tobytes())
        result.update(passed=True, status="passed", native_gl_readback=True, comparisons=comparisons,
                      maximum_channel_error=max(check["max_channel_error"] for check in comparisons.values()),
                      managed_draw_reports=managed_reports, resources=resources, invocations=invocations,
                      input_sha256=hashlib.sha256(instances.read_bytes()).hexdigest(),
                      native_capture_paths=proof["capture_paths"], native_commits=proof["commits"],
                      production_report_label="unchanged arrays-only helper label; indexed proof is EBO captures and private EXT call logs")
    except (AssertionError, ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result.update(status="failed", message=str(error), invocations=invocations)
        raise
    finally:
        report_path.write_text(json.dumps(result, indent=2) + "\n")
