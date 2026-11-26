"""Fresh private C ABI4 capture, transactional IPC guards, and host-rendered draws."""

import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time

import pytest

from test_texture_capture import build_texture_guard, read_capture


SOURCE = Path(__file__).parent / "fixtures/host-features-capture/guard.c"
FEATURES = {
    "version": 1, "shader_abi_version": 4, "max_instances": 256,
    "max_expanded_vertices": 65536,
    "indexing": "base-instance-plus-floor-local-instance-over-divisor",
}
FORMATS = ("R8G8B8A8_UNORM", "B8G8R8A8_UNORM", "R8G8B8_UNORM", "R8_UNORM")
COLOR_NAMES = ("rgba8", "bgra8", "rgb565", "r8")
DEPTH_NAMES = ("none", "z16", "z24s8")
COLOR_BYTES = (4, 4, 2, 1)
DEPTH_BYTES = (0, 2, 4)
REJECTED = (
    "feature_max_instances", "layout_version", "duplicate_feature", "texture_contract",
    "binding_contract", "texture_base", "texture_samples", "shader_abi", "missing_view",
    "missing_sampler", "min_filter", "mip_filter", "max_lod", "swizzle", "level_offset",
    "level_stride", "active_execution", "mapped", "feedback", "over_instances",
    "over_vertices", "restart", "vertex_mapped", "bad_factor", "bad_equation", "dither",
    "nan_constant", "scissor", "bad_viewport", "front_cw",
)


@pytest.fixture(scope="module")
def features_guard(tmp_path_factory):
    return build_texture_guard(tmp_path_factory, SOURCE)


def run_features(helper, directory, case="valid", *, version=11, shape=(32, 32),
                 source_format=0, color_format=0, depth_format=1, features="1",
                 target=None, socket_path=None, mutation=None):
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("MGC_") and key != "MGC_RUNTIME_SANITIZE":
            environment.pop(key)
    if features is not None:
        environment["MGC_HOST_FEATURES"] = features
    if version is not None:
        environment["MGC_HOST_TEXTURE"] = str(version)
        environment["MGC_HOST_TEXTURE_WRAP_S"] = "repeat"
        environment["MGC_HOST_TEXTURE_WRAP_T"] = "mirrored-repeat"
    if target is not None:
        environment["MGC_COMPILE_TARGET"] = target
    if socket_path is not None:
        environment["MGC_HOST_EXECUTOR_SOCKET"] = str(socket_path)
    if mutation is not None:
        environment["MGC_TEST_CONTRACT_OLD"], environment["MGC_TEST_CONTRACT_NEW"] = mutation
    environment["MGC_HOST_CAPTURE_DIR"] = str(directory)
    result = subprocess.run([str(helper), case, *(str(value) for value in shape),
                             str(source_format), str(color_format), str(depth_format)],
                            env=environment, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def assert_features(artifact):
    assert artifact["schema"] == "mgc-host-draw-v4"
    assert artifact["shader_abi_version"] == 4
    assert artifact["host_features"] == FEATURES
    assert artifact["program_format"] == "legacy-mgcsim-v1"
    for stage in ("vs", "fs"):
        shader = artifact["shaders"][stage]
        assert shader["layout"]["version"] == shader["layout"]["shader_abi_version"] == 4
        assert shader["layout"]["bases"]["texture"] == 0x110000
        assert shader["layout"]["host_features"]["max_instances"] == 256
        assert shader["assembly"].startswith("; mgc legacy-simulator backend, ABI=4,")


@pytest.mark.parametrize("version", (11, 12, 13, 14))
@pytest.mark.parametrize("source_format", range(4))
@pytest.mark.parametrize("shape", ((1, 1), (3, 5), (1, 32), (32, 1), (32, 32)))
def test_typed_complete_mip_snapshots(features_guard, tmp_path, version, source_format, shape):
    status = run_features(features_guard, tmp_path, version=version, source_format=source_format, shape=shape)
    assert status["accepted"] == 1 and status["error"] == status["callbacks"] == 0
    artifact = read_capture(tmp_path)
    assert_features(artifact)
    texture, = artifact["textures"]
    assert texture["binding"] == texture["texture"] == texture["sampler"] == 0
    assert texture["format"] == FORMATS[source_format]
    filtering = "nearest" if version == 11 else "linear"
    expected_sampler = {
        "normalized_coords": True, "min_filter": filtering, "mag_filter": filtering,
        "mip_filter": "linear" if version >= 13 else "nearest",
        "wrap_s": "repeat", "wrap_t": "mirrored-repeat",
    }
    if version == 14:
        expected_sampler["max_anisotropy"] = 4
    assert texture["sampler_state"] == expected_sampler
    offset = 0
    blocksize = (4, 4, 3, 1)[source_format]
    for level, payload in enumerate((texture, *texture["mipmaps"])):
        width, height = (max(1, edge >> level) for edge in shape)
        assert (payload["width"], payload["height"]) == (width, height)
        assert payload["format"] == FORMATS[source_format]
        if level:
            assert payload["level"] == level
            offset = (offset + 63) & ~63
        size = width * height * blocksize
        assert bytes.fromhex(payload["data_hex"]) == bytes(((offset + index) * 17 + 3) & 255 for index in range(size))
        offset += size
    assert len(texture["mipmaps"]) == max(shape).bit_length() - 1


@pytest.mark.parametrize("case,pairs", (
    ("four_bindings", [(0, 0), (1, 1), (2, 2), (3, 3)]),
    ("aliases", [(0, 0), (1, 1), (2, 2), (3, 3)]),
    ("sparse", [(3, 1), (1, 3)]),
    ("same_texture", [(0, 0), (0, 1)]),
))
def test_static_sparse_pairs_and_read_only_aliases(features_guard, tmp_path, case, pairs):
    assert run_features(features_guard, tmp_path, case)["accepted"] == 1
    artifact = read_capture(tmp_path)
    assert_features(artifact)
    captured_pairs = [(texture["texture"], texture["sampler"]) for texture in artifact["textures"]]
    assert set(captured_pairs) == set(pairs)
    assert [texture["binding"] for texture in artifact["textures"]] == [texture for texture, _ in captured_pairs]
    assert artifact["shaders"]["fs"]["layout"]["host_texture"]["bindings"] == [
        {"texture": texture, "sampler": sampler} for texture, sampler in captured_pairs]


@pytest.mark.parametrize("source_format", range(4))
def test_native_format_default_view_expansion(features_guard, tmp_path, source_format):
    assert run_features(features_guard, tmp_path, "canonical_swizzle", source_format=source_format)["accepted"] == 1
    texture, = read_capture(tmp_path)["textures"]
    assert texture["format"] == FORMATS[source_format]


@pytest.mark.parametrize("source_format", range(4))
def test_noncanonical_view_swizzles_reject(features_guard, tmp_path, source_format):
    status = run_features(features_guard, tmp_path, "swizzle", source_format=source_format)
    assert status["accepted"] == status["sequence"] == status["callbacks"] == 0
    assert status["rejected"] == 1 and status["error"] != 0


@pytest.mark.parametrize("case", REJECTED)
def test_reject_before_callback_publication_or_sequence(features_guard, tmp_path, case):
    status = run_features(features_guard, tmp_path, case)
    assert status["accepted"] == status["callbacks"] == status["sequence"] == 0
    assert status["rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", (
    ('"max_instances":256', '"max_instances":"256"'),
    ('"max_instances":256', '"max_instances":256.0'),
    ('"max_instances":256', '"max_instances":256,"max_instances":256'),
    ('"max_expanded_vertices":65536', '"max_expanded_vertices":65537'),
    ('"reserved_input_words":[65504,65535]', '"reserved_input_words":[0,65535]'),
    ('"reserved_output_words":[57344,65534]', '"reserved_output_words":[57344,65535]'),
    ('"uses_derivatives":false', '"uses_derivatives":true'),
    ('"derivative_mailbox":null', '"derivative_mailbox":{}'),
    ('"word_offset":57344', '"word_offset":57345'),
    ('"max_words":4096', '"max_words":4097'),
    ('"fault_word_offset":65499', '"fault_word_offset":65498'),
    ('"word_offset":0', '"word_offset":65498'),
    ('"word_offset":0', '"word_offset":65504'),
    ('"word_offset":0', '"word_offset":0,"word_offset":0'),
    ('"component":0', '"component":4'),
    ('"type":"float"', '"type":"double"'),
    ('"bit_size":32', '"bit_size":64'),
    ('"interpolation":"smooth"', '"interpolation":"unknown"'),
    ('"texture":0,"sampler":0', '"texture":4,"sampler":0'),
    ('"texture":0,"sampler":0', '"texture":0,"sampler":4'),
    ('"max_bindings":4', '"max_bindings":5'),
    ('"max_samples":32', '"max_samples":33'),
    ('"max_size":32', '"max_size":64'),
    ('"max_levels":6', '"max_levels":7'),
    ('"per_binding_stride":16384', '"per_binding_stride":8192'),
    ('"header_words":32', '"header_words":31'),
    ('"lod_mode":"explicit"', '"lod_mode":"implicit"'),
    ('"filter":"nearest"', '"filter":"linear"'),
    ('"mip_filter":"nearest"', '"mip_filter":"linear"'),
    ('"wrap_s":"repeat"', '"wrap_s":"clamp-to-edge"'),
    ('"texture":1114112', '"texture":131072'),
))
def test_exact_versioned_metadata_rejections(features_guard, tmp_path, mutation):
    status = run_features(features_guard, tmp_path, mutation=mutation)
    assert status["accepted"] == status["callbacks"] == status["sequence"] == 0
    assert status["rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("features", ("", "2", "01", "+1", "1.0", "true", " 1", "1 "))
def test_canonical_opt_in_stays_failed_after_clear(features_guard, tmp_path, features):
    status = run_features(features_guard, tmp_path, features=features)
    assert status["initial_error"] != 0 and status["error"] != 0 and status["rejected"] == 1
    assert status["sequence"] == status["accepted"] == status["callbacks"] == 0


def test_hardware_never_accepts_host_features(features_guard, tmp_path):
    status = run_features(features_guard, tmp_path, target="fp16-hardware-v1")
    assert status["initial_error"] != 0 and status["rejected"] == 1 and status["sequence"] == 0


@pytest.mark.parametrize("features", (None, "0"))
def test_default_callback_abi_stays_one(features_guard, tmp_path, features):
    status = run_features(features_guard, tmp_path, "custom", features=features, version=None)
    assert status["accepted"] == status["callbacks"] == 1 and status["error"] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ("max_instances", "max_vertices", "divisor", "indexed"))
def test_expansion_budgets_and_indexing(features_guard, tmp_path, case):
    assert run_features(features_guard, tmp_path, case)["accepted"] == 1
    assert_features(read_capture(tmp_path))


@pytest.mark.parametrize("case", ("builtins", "derivatives", "implicit", "gradients"))
def test_genuine_compiler_contracts(features_guard, tmp_path, case):
    status = run_features(features_guard, tmp_path, case, version=11 if case in ("implicit", "gradients") else None)
    assert status["accepted"] == 1 and status["error"] == 0
    artifact = read_capture(tmp_path)
    assert_features(artifact)
    if case == "builtins":
        assert len(artifact["shaders"]["vs"]["layout"]["host_features"]["builtins"]) == 4
        assert len(artifact["shaders"]["fs"]["layout"]["host_features"]["builtins"]) == 2
    else:
        assert artifact["shaders"]["fs"]["layout"]["host_features"]["uses_derivatives"] == (case != "gradients")


@pytest.mark.parametrize("mask", range(1, 16))
def test_fragcoord_exact_used_components(features_guard, tmp_path, mask):
    status = run_features(features_guard, tmp_path, f"coord_mask_{mask}", version=None)
    assert status["accepted"] == 1 and status["error"] == 0
    artifact = read_capture(tmp_path)
    assert_features(artifact)
    builtin, = artifact["shaders"]["fs"]["layout"]["host_features"]["builtins"]
    assert builtin == {
        "name": "gl_FragCoord",
        "word_offsets": [65524 + component for component in range(4) if mask & (1 << component)],
        "type": "float", "bit_size": 32, "storage": "low_fp16",
    }


@pytest.mark.parametrize("replacement", (
    "[]", "[65524]", "[65524,65525,65526,65527]", "[65526,65525,65524]",
    "[65524,65525,65525]", "[65524,65525,65528]",
))
def test_fragcoord_changed_component_contract_rejected(features_guard, tmp_path, replacement):
    status = run_features(features_guard, tmp_path, "coord_mask_7", version=None,
                          mutation=('"word_offsets":[65524,65525,65526]',
                                    f'"word_offsets":{replacement}'))
    assert status["accepted"] == status["callbacks"] == status["sequence"] == 0
    assert status["rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("anisotropy", (0, 1, 2, 3, 5, 8, 16, 31))
def test_v14_requires_exact_anisotropy_four(features_guard, tmp_path, anisotropy):
    status = run_features(features_guard, tmp_path, f"anisotropy_{anisotropy}", version=14)
    assert status["accepted"] == status["callbacks"] == status["sequence"] == 0
    assert status["rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", (11, 12, 13))
def test_previous_texture_versions_still_reject_anisotropy(features_guard, tmp_path, version):
    status = run_features(features_guard, tmp_path, "anisotropy_4", version=version)
    assert status["accepted"] == status["callbacks"] == status["sequence"] == 0
    assert status["rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("replacement", ("3", "5", "4.0", "true", '"4"'))
def test_v14_anisotropy_contract_mutation_rejected(features_guard, tmp_path, replacement):
    status = run_features(features_guard, tmp_path, version=14,
                          mutation=('"max_anisotropy":4', f'"max_anisotropy":{replacement}'))
    assert status["accepted"] == status["callbacks"] == status["sequence"] == 0
    assert status["rejected"] == 1 and status["error"] != 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("color_format", range(4))
@pytest.mark.parametrize("depth_format", range(3))
def test_pipeline_attachment_formats(features_guard, tmp_path, color_format, depth_format):
    assert run_features(features_guard, tmp_path, version=None, color_format=color_format, depth_format=depth_format)["accepted"] == 1
    state = read_capture(tmp_path)["render_state"]
    assert state["framebuffer"] == {"color_format": COLOR_NAMES[color_format], "depth_stencil_format": DEPTH_NAMES[depth_format]}


@pytest.mark.parametrize("index", range(15))
def test_separate_standard_factors(features_guard, tmp_path, index):
    from tools.host_pipeline_state import BLEND_FACTORS, validate_state
    assert run_features(features_guard, tmp_path, f"factor_{index}", version=None)["accepted"] == 1
    state = read_capture(tmp_path)["render_state"]
    validate_state(state)
    assert state["blend"]["src_rgb"] == state["blend"]["src_alpha"] == BLEND_FACTORS[index]
    assert state["blend"]["dst_rgb"] == "constant-color"
    assert state["blend"]["dst_alpha"] == "one-minus-dst-alpha"


@pytest.mark.parametrize("index", range(5))
def test_separate_equations(features_guard, tmp_path, index):
    from tools.host_pipeline_state import BLEND_EQUATIONS, validate_state
    assert run_features(features_guard, tmp_path, f"equation_{index}", version=None)["accepted"] == 1
    state = read_capture(tmp_path)["render_state"]
    validate_state(state)
    assert state["blend"]["rgb_equation"] == BLEND_EQUATIONS[index]
    assert state["blend"]["alpha_equation"] == BLEND_EQUATIONS[(index + 1) % 5]


@pytest.mark.parametrize("index", range(8))
def test_all_depth_comparisons(features_guard, tmp_path, index):
    from tools.host_pipeline_state import COMPARISONS, validate_state
    assert run_features(features_guard, tmp_path, f"depth_{index}", version=None)["accepted"] == 1
    state = read_capture(tmp_path)["render_state"]
    validate_state(state)
    assert state["depth"]["comparison"] == COMPARISONS[index]


def test_two_sided_stencil_references_and_masks(features_guard, tmp_path):
    from tools.host_pipeline_state import validate_state
    assert run_features(features_guard, tmp_path, "stencil", version=None, depth_format=2)["accepted"] == 1
    state = read_capture(tmp_path)["render_state"]
    validate_state(state)
    assert state["stencil"]["enabled"] is True
    assert state["stencil"]["front"] == {
        "comparison": "gequal", "reference": 17, "read_mask": 0xa5, "write_mask": 0x5a,
        "fail": "increment-wrap", "zfail": "decrement-clamp", "zpass": "replace",
    }
    assert state["stencil"]["back"]["reference"] == 231
    assert state["stencil"]["back"]["zfail"] == "invert"


def receive_exact(peer, size):
    result = bytearray()
    while len(result) < size:
        part = peer.recv(size - len(result))
        if not part:
            raise AssertionError("truncated request")
        result.extend(part)
    return bytes(result)


@pytest.mark.parametrize("color_format", range(4))
@pytest.mark.parametrize("depth_format", range(3))
def test_native_v4_typed_wire_and_commit(features_guard, tmp_path, color_format, depth_format):
    _native_case(features_guard, tmp_path, "native_valid", color_format, depth_format)


@pytest.mark.parametrize("case", ("native_tag", "native_depth_size", "native_truncated", "native_state", "native_data", "native_maps"))
def test_native_v4_transaction_guards(features_guard, tmp_path, case):
    _native_case(features_guard, tmp_path, case, 2, 2)


@pytest.mark.parametrize("color_format", range(4))
@pytest.mark.parametrize("depth_format", range(3))
def test_actual_compiled_draw_to_native_resource(features_guard, tmp_path, color_format, depth_format):
    from tools.host_executor import HostExecutorServer
    capture = tmp_path / "capture"
    capture.mkdir()
    path = tmp_path / "actual.sock"
    with HostExecutorServer(path, max_requests=1) as server:
        status = run_features(features_guard, capture, version=None, color_format=color_format,
                              depth_format=depth_format, socket_path=path)
    assert not server.errors and len(server.records) == 1, server.errors
    assert status["accepted"] == 1 and status["error"] == status["pins"] == 0
    record = server.records[0]
    assert_features(record["artifact"])
    assert record["coverage"].any()
    if depth_format == 0:
        expected = ((255, 128, 64, 255), (255, 128, 64, 255),
                    (255, 130, 66, 255), (255, 0, 0, 255))[color_format]
        assert (record["rgba"][record["coverage"] != 0] == expected).all()
    color = record["native_color"]
    row_bytes = 3 * COLOR_BYTES[color_format]
    assert bytes.fromhex(status["color_hex"]) == color[row_bytes:] + color[:row_bytes]
    if depth_format:
        depth = record["depth"]
        row_bytes = 3 * DEPTH_BYTES[depth_format]
        assert bytes.fromhex(status["depth_hex"]) == depth[row_bytes:] + depth[:row_bytes]


def _native_case(helper, directory, case, color_format, depth_format):
    capture = directory / "capture"
    capture.mkdir()
    path = directory / "executor.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    listener.settimeout(10)
    evidence, errors = {}, []

    def serve():
        try:
            peer, _ = listener.accept()
            with peer:
                peer.settimeout(10)
                magic, json_size, sequence, width, height, color_tag, depth_tag, color_size, depth_size = struct.unpack("<8sIQIIIIII", receive_exact(peer, 44))
                assert magic == b"MGCHE04\0" and (width, height) == (3, 2)
                assert (color_tag, depth_tag) == (color_format + 1, depth_format)
                assert color_size == width * height * COLOR_BYTES[color_format]
                assert depth_size == width * height * DEPTH_BYTES[depth_format]
                artifact = json.loads(receive_exact(peer, json_size))
                assert_features(artifact)
                evidence["initial_color"] = receive_exact(peer, color_size)
                evidence["initial_depth"] = receive_exact(peer, depth_size)
                assert peer.recv(1) == b""
                time.sleep(0.1)
                color = bytes((index * 7 + 19) & 255 for index in range(color_size))
                depth = bytes((index * 11 + 23) & 255 for index in range(depth_size))
                reply = struct.pack("<8sQIIIIIII", b"MGCRG04\0", sequence, 0, width, height,
                                    color_tag + (case == "native_tag"), depth_tag, color_size,
                                    depth_size + (case == "native_depth_size"))
                evidence["color"], evidence["depth"] = color, depth
                peer.sendall(reply + color + (depth[:-1] if case == "native_truncated" else depth))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        status = run_features(helper, capture, case, version=None, color_format=color_format,
                              depth_format=depth_format, socket_path=path)
    finally:
        thread.join(15)
        listener.close()
    assert not thread.is_alive() and not errors, errors
    assert status["pins"] == 0 and status["sequence"] != 0
    success = case in ("native_valid", "native_maps")
    assert status["accepted"] == int(success) and status["rejected"] == int(not success)
    assert (status["error"] == 0) == success
    if success:
        for field, blocksize in (("color", COLOR_BYTES[color_format]), ("depth", DEPTH_BYTES[depth_format])):
            row = 3 * blocksize
            expected = evidence[field][row:] + evidence[field][:row]
            assert bytes.fromhex(status[f"{field}_hex"]) == expected
    else:
        initial = bytes((index * 17 + 3) & 255 for index in range(6 * COLOR_BYTES[color_format]))
        if case == "native_data":
            initial = bytes([initial[0] ^ 1]) + initial[1:]
        assert bytes.fromhex(status["color_hex"]) == initial
        assert bytes.fromhex(status["depth_hex"]) == bytes((index * 17 + 3) & 255 for index in range(6 * DEPTH_BYTES[depth_format]))
    if case == "native_maps":
        assert status["rejected_maps"] == 3
