import copy
import json
import socket
import struct

import numpy as np
import pytest

from tools.host_executor import (COLOR_FORMATS, DEPTH_FORMATS, HostExecutorServer,
                                 REPLY_FEATURES, REPLY_FEATURES_MAGIC, REQUEST_FEATURES,
                                 REQUEST_FEATURES_MAGIC, decode_native_color, encode_native_color)
from tools.host_features import BUILTINS, DERIVATIVE_MAILBOX, DRAW_FEATURES, validate_draw_features
from tools.host_pipeline_state import default_state, pack_depth_stencil
from tools.host_render import ShaderExecutor, render_artifact
from tools.mgcasm.fp16 import float_to_fp16_bits
from tools.shader_layout import ShaderLayout


BASES = {"input": 65536, "uniform": 131072, "vertex": 196608,
         "fragment": 1048576, "texture": 1114112}


def scalar(offset, component, location=0):
    return {"word_offset": offset, "location": location, "component": component,
            "type": "float", "bit_size": 32, "interpolation": "smooth"}


def shader_layout(stage):
    outputs = [scalar(component, component, 0 if stage == "vertex" else 2) for component in range(4)]
    result = {"version": 4, "supported": True, "shader_abi_version": 4,
              "program_format": "legacy-mgcsim-v1", "address_units": "words", "stage": stage,
              "bases": dict(BASES), "inputs": copy.deepcopy(outputs) if stage == "vertex" else [],
              "outputs": outputs, "uniforms": [],
              "host_features": {"version": 1, "max_instances": 256, "max_expanded_vertices": 65536,
                                "reserved_input_words": [65504, 65535], "reserved_output_words": [57344, 65534],
                                "builtins": [], "uses_derivatives": False, "derivative_mailbox": None,
                                "spill": {"word_offset": 57344, "max_words": 4096, "words": 0},
                                "fault_word_offset": 65499}}
    if stage == "fragment":
        result.update(uses_discard=False, fragment_status={"version": 1, "word_offset": 65535,
                                                          "pending": 0, "discarded": 1, "committed": 2})
    return result


def managed_fixture(rgba=(1, 0, 0, 1), status=2):
    vertex = "; mgc legacy-simulator backend, ABI=4, explicit hand-assembled protocol fixture\n"
    fragment = vertex
    for component in range(4):
        vertex += (f"movi_i32 {BASES['input'] + component}, r0\nload r0, r1\nwait_load\n"
                   f"movi_i32 {BASES['vertex'] + component}, r0\nstore r1, r0\nwait_short\n")
        fragment += (f"movi_i32 {float_to_fp16_bits(rgba[component])}, r1\n"
                     f"movi_i32 {BASES['fragment'] + component}, r0\nstore r1, r0\nwait_short\n")
    fragment += (f"movi_i32 {status}, r1\nmovi_i32 {BASES['fragment'] + 65535}, r0\n"
                 "store r1, r0\nwait_short\nstop_core\n")
    vertex += "stop_core\n"
    positions = [(-1, -1, 0, 1), (1, -1, 0, 1), (1, 1, 0, 1),
                 (-1, -1, 0, 1), (1, 1, 0, 1), (-1, 1, 0, 1)]
    return {"schema": "mgc-host-draw-v4", "program_format": "legacy-mgcsim-v1", "shader_abi_version": 4,
            "host_features": dict(DRAW_FEATURES), "render_state": default_state(), "sequence": 1,
            "draw": {"mode": "triangles", "count": 6, "start": 0, "instance_count": 1,
                     "start_instance": 0, "primitive_restart": False, "index_size": 0, "index_bias": 0},
            "shaders": {"vs": {"assembly": vertex, "layout": shader_layout("vertex")},
                        "fs": {"assembly": fragment, "layout": shader_layout("fragment")}},
            "vertex_buffers": [{"stride": 16, "data_hex": b"".join(struct.pack("<4f", *position) for position in positions).hex()}],
            "vertex_elements": [{"src_offset": 0, "vertex_buffer_index": 0, "instance_divisor": 0,
                                 "format": "R32G32B32A32_FLOAT"}],
            "constants": {"vs": [], "fs": []}, "indices": None, "textures": []}


def exchange(server, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(str(server.path))
        connection.sendall(payload)
        connection.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            try:
                chunk = connection.recv(65536)
            except ConnectionResetError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def request(artifact, color_tag=1, depth_tag=1, image=None, depth=None):
    image = np.full((4, 4, 4), (3, 9, 17, 255), dtype=np.uint8) if image is None else image
    color = encode_native_color(image, color_tag)
    depth = (pack_depth_stencil(DEPTH_FORMATS[depth_tag][0], 1, 19 if depth_tag == 2 else 0) * 16 if depth_tag else b"") if depth is None else depth
    encoded = json.dumps(artifact, separators=(",", ":")).encode()
    return REQUEST_FEATURES.pack(REQUEST_FEATURES_MAGIC, len(encoded), 1, 4, 4,
                                 color_tag, depth_tag, len(color), len(depth)) + encoded + color + depth


def test_exact_native_v4_header_sizes_and_storage_evidence():
    assert REQUEST_FEATURES.size == REPLY_FEATURES.size == 44
    red = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
    assert encode_native_color(red, 3) == b"\x00\xf8"
    assert pack_depth_stencil("z24s8", 1, 19) == b"\xff\xff\xff\x13"


@pytest.mark.parametrize("color_tag", COLOR_FORMATS)
def test_native_color_roundtrip_preserves_representable_storage(color_tag):
    bytes_per_pixel = COLOR_FORMATS[color_tag][1]
    source = bytes((index * 37 + 11) % 256 for index in range(7 * 5 * bytes_per_pixel))
    assert encode_native_color(decode_native_color(source, 7, 5, color_tag), color_tag) == source


@pytest.mark.parametrize("color_tag", COLOR_FORMATS)
@pytest.mark.parametrize("depth_tag", DEPTH_FORMATS)
def test_native_v4_uses_managed_legacy_words_and_typed_attachments(tmp_path, color_tag, depth_tag):
    artifact = managed_fixture()
    artifact["render_state"] = default_state(color_format=COLOR_FORMATS[color_tag][0],
                                             depth_stencil_format=DEPTH_FORMATS[depth_tag][0])
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        response = exchange(server, request(artifact, color_tag, depth_tag))
    header = REPLY_FEATURES.unpack(response[:44])
    assert header == (REPLY_FEATURES_MAGIC, 1, 0, 4, 4, color_tag, depth_tag,
                      16 * COLOR_FORMATS[color_tag][1], 16 * DEPTH_FORMATS[depth_tag][1])
    expected = encode_native_color(np.full((4, 4, 4), (255, 0, 0, 255), dtype=np.uint8), color_tag)
    assert response[44:44 + len(expected)] == expected
    assert not server.errors and len(server.records) == 1
    assert server.records[0]["report"]["fpga_execution"] is False


@pytest.mark.parametrize("status", (0, 1))
def test_discard_or_failure_never_publishes_attachment_writes(status):
    artifact = managed_fixture(status=status)
    artifact["render_state"]["depth"].update(enabled=True, write=True)
    initial = np.full((4, 4, 4), (3, 9, 17, 255), dtype=np.uint8)
    depth = pack_depth_stencil("z16", 1) * 16
    staging = bytearray(b"\xab" * len(depth))
    if status == 0:
        with pytest.raises(ValueError, match="completion"):
            render_artifact(artifact, width=4, height=4, initial_image=initial,
                            initial_depth_stencil=depth, depth_stencil_output=staging)
        assert staging == b"\xab" * len(depth)
    else:
        image, coverage, report = render_artifact(artifact, width=4, height=4, initial_image=initial,
                                                 initial_depth_stencil=depth, depth_stencil_output=staging)
        assert np.array_equal(image, initial) and not np.any(coverage)
        assert staging == depth and report["committed_fragments"] == 0
    assert np.all(initial == (3, 9, 17, 255))


@pytest.mark.parametrize("change", ("features", "sequence", "state", "schema", "hardware", "depth-format"))
def test_native_v4_rejects_contract_mismatch_with_zero_payload(tmp_path, change):
    artifact = managed_fixture()
    if change == "features":
        artifact["host_features"]["max_instances"] = True
    elif change == "sequence":
        artifact["sequence"] = 2
    elif change == "state":
        artifact["render_state"]["depth"]["comparison"] = "less-equal"
    elif change == "schema":
        artifact["schema"] = "mgc-host-draw-v1"
    elif change == "hardware":
        artifact["program_format"] = "fp16-hardware-v1"
    else:
        artifact["render_state"]["framebuffer"]["depth_stencil_format"] = "z24s8"
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        response = exchange(server, request(artifact))
    assert REPLY_FEATURES.unpack(response) == (REPLY_FEATURES_MAGIC, 1, 1, 4, 4, 1, 1, 0, 0)
    assert not server.records and len(server.errors) == 1


@pytest.mark.parametrize("change", ("old-abi", "builtin", "duplicate-builtin", "spill", "input", "output", "mailbox"))
def test_shader_reflection_rejects_ambiguous_reserved_contracts(change):
    source = shader_layout("fragment")
    if change == "old-abi":
        source.update(version=1, shader_abi_version=1)
    elif change == "builtin":
        source["host_features"]["builtins"] = [{"name": "gl_Unknown"}]
    elif change == "duplicate-builtin":
        builtin = {"name": "gl_FrontFacing", "word_offsets": [65528], "type": "bool", "bit_size": 1, "storage": "raw32"}
        source["host_features"]["builtins"] = [builtin, copy.deepcopy(builtin)]
    elif change == "spill":
        source["host_features"]["spill"]["words"] = 4097
    elif change == "input":
        source["inputs"] = [scalar(65498, 0)]
    elif change == "output":
        source["outputs"][0]["word_offset"] = 57344
    else:
        source["host_features"]["derivative_mailbox"] = {}
    with pytest.raises(ValueError):
        ShaderLayout(source)


def test_old_capture_cannot_silently_accept_host_feature_metadata():
    artifact = managed_fixture()
    artifact["schema"] = "mgc-host-draw-v1"
    with pytest.raises(ValueError, match="version4"):
        validate_draw_features(artifact)


@pytest.mark.parametrize("name,value", (("gl_InstanceID", 17), ("gl_VertexID", 4097),
                                       ("gl_BaseInstance", 128), ("gl_DrawID", 3)))
def test_managed_vertex_builtin_words_are_explicit_and_not_float_payloads(name, value):
    source = shader_layout("vertex")
    offsets, kind, size, storage, stage = BUILTINS[name]
    source["inputs"] = []
    source["outputs"] = [{**scalar(0, 0, 32), "type": "uint"}]
    source["host_features"]["builtins"] = [{"name": name, "word_offsets": offsets, "type": kind,
                                            "bit_size": size, "storage": storage}]
    assembly = ("; mgc legacy-simulator backend, ABI=4, explicit hand-assembled builtin fixture\n"
                f"movi_i32 {BASES['input'] + offsets[0]}, r0\nload r0, r1\nwait_load\n"
                f"movi_i32 {BASES['vertex']}, r0\nstore r1, r0\nwait_short\nstop_core\n")
    executor = ShaderExecutor(assembly, ShaderLayout(source))
    try:
        assert executor.invoke({}, builtins={name: value}) == {0: value}
        with pytest.raises(ValueError, match="builtin"):
            executor.invoke({})
    finally:
        executor.close()


def derivative_fixture():
    artifact = managed_fixture()
    vertex = artifact["shaders"]["vs"]
    vertex["layout"]["outputs"].append(scalar(4, 0, 32))
    vertex["assembly"] = vertex["assembly"].removesuffix("stop_core\n") + (
        f"movi_i32 {BASES['input']}, r0\nload r0, r1\nwait_load\n"
        f"movi_i32 {BASES['vertex'] + 4}, r0\nstore r1, r0\nwait_short\nstop_core\n")
    fragment = artifact["shaders"]["fs"]
    fragment["layout"]["inputs"] = [scalar(0, 0, 32)]
    fragment["layout"]["host_features"].update(uses_derivatives=True, derivative_mailbox=copy.deepcopy(DERIVATIVE_MAILBOX))
    prelude = (f"movi_i32 {BASES['input']}, r0\nload r0, r1\nwait_load\n"
               f"movi_i32 {BASES['fragment'] + 65497}, r0\nstore r1, r0\nwait_short\n"
               f"movi_i32 1, r1\nmovi_i32 {BASES['fragment'] + 65496}, r0\nstore r1, r0\nwait_short\n"
               f"movi_i32 {BASES['input'] + 65498}, r0\nload r0, r1\nwait_load\n"
               f"movi_i32 {BASES['fragment']}, r0\nstore r1, r0\nwait_short\n")
    fragment["assembly"] = ("; mgc legacy-simulator backend, ABI=4, explicit hand-assembled derivative fixture\n" +
                            prelude + fragment["assembly"].split("wait_short\n", 1)[1])
    return artifact


def test_full_managed_raster_linkage_feeds_actual_quad_derivatives():
    image, coverage, report = render_artifact(derivative_fixture(), width=4, height=4)
    assert np.all(image == (128, 0, 0, 255)) and np.all(coverage == 1)
    assert report["committed_fragments"] == 16 and report["fragments"] == 24
    assert report["helper_invocations"] == 8
    assert report["fpga_execution"] is False and report["host_steps"] > 0


def test_helpers_obey_draw_budget_without_committing_native_depth_staging():
    artifact = derivative_fixture()
    output = bytearray(b"\xcd" * 32)
    with pytest.raises(ValueError, match="budget"):
        render_artifact(artifact, width=4, height=4, max_fragments=20, depth_stencil_output=output)
    assert output == b"\xcd" * 32


@pytest.mark.parametrize("component", range(4))
def test_fragment_coordinate_component_subset_selects_original_vec4_component(component):
    source = shader_layout("fragment")
    offset = 65524 + component
    source["host_features"]["builtins"] = [{"name": "gl_FragCoord", "word_offsets": [offset],
                                            "type": "float", "bit_size": 32, "storage": "low_fp16"}]
    assembly = ("; mgc legacy-simulator backend, ABI=4, explicit hand-assembled coordinate fixture\n"
                f"movi_i32 {BASES['input'] + offset}, r0\nload r0, r1\nwait_load\n"
                f"movi_i32 {BASES['fragment']}, r0\nstore r1, r0\nwait_short\n"
                f"movi_i32 2, r1\nmovi_i32 {BASES['fragment'] + 65535}, r0\nstore r1, r0\nwait_short\nstop_core\n")
    executor = ShaderExecutor(assembly, ShaderLayout(source))
    coordinates = (.5, 1.5, .75, 1)
    try:
        result = executor.invoke_fragment({}, builtins={"gl_FragCoord": coordinates})
        assert result.outputs[0] == float_to_fp16_bits(coordinates[component])
    finally:
        executor.close()


@pytest.mark.parametrize("offsets", ([], [65523], [65528], [65524, 65524], [65525, 65524], [True], [65524.0]))
def test_fragment_coordinate_subset_is_not_an_arbitrary_reserved_address_list(offsets):
    source = shader_layout("fragment")
    source["host_features"]["builtins"] = [{"name": "gl_FragCoord", "word_offsets": offsets,
                                            "type": "float", "bit_size": 32, "storage": "low_fp16"}]
    with pytest.raises(ValueError):
        ShaderLayout(source)
@pytest.mark.parametrize("mode", ("capture", "native"))
def test_public_frontend_fresh_abi4_flat_draw(tmp_path, mode):
    from tools.host_executor import HostExecutorServer
    from tools.host_render import render_artifact
    from tools.run_render import capture_draw, native_host_readback
    from tools.run_shader import ROOT

    shader = ROOT / "shaders/01_flat_color"
    build = ROOT / "mesa/build"
    if mode == "capture":
        artifact = capture_draw(shader, build, tmp_path / "capture", host_features=True)
        image, coverage, report = render_artifact(artifact, width=8, height=8)
    else:
        socket_path = tmp_path / "executor.sock"
        with HostExecutorServer(socket_path, max_requests=1) as server:
            image = native_host_readback(shader, build, tmp_path / "native",
                                         socket_path=socket_path, width=8, height=8,
                                         host_features=True)
        assert not server.errors and len(server.records) == 1
        record, = server.records
        artifact, coverage, report = record["artifact"], record["coverage"], record["report"]
        assert np.array_equal(image, record["rgba"])
        assert record["color_format"] == "rgba8" and record["depth_format"] == "none"
    assert artifact["schema"] == "mgc-host-draw-v4"
    assert artifact["shader_abi_version"] == 4
    assert coverage.any() and np.all(image[coverage != 0] == (255, 0, 0, 255))
    assert report["fpga_execution"] is False


@pytest.mark.parametrize("value", (1, "1", None, []))
def test_public_frontend_rejects_nonboolean_abi4_opt_in(tmp_path, value):
    from tools.run_render import capture_draw, native_host_readback
    from tools.run_shader import ROOT

    with pytest.raises(ValueError, match="explicitly boolean"):
        capture_draw(ROOT / "shaders/01_flat_color", ROOT / "mesa/build", tmp_path,
                     host_features=value)
    with pytest.raises(ValueError, match="explicitly boolean"):
        native_host_readback(ROOT / "shaders/01_flat_color", ROOT / "mesa/build", tmp_path,
                             socket_path=tmp_path / "executor.sock", host_features=value)
