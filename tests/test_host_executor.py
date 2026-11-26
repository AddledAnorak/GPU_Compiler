import copy
import json
import os
from pathlib import Path
import socket
import stat
import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import host_render
from tools.fragment_acceptance import mesh_words
from tools.host_executor import HostExecutorServer, MAX_JSON_BYTES, REPLY, REPLY_MAGIC, REQUEST, REQUEST_MAGIC
from tools.host_render import render_artifact
from tools.mgcasm import assemble
from tools.mgcasm import isa as isa_module
from tools.mgcdriver import Driver, Program, ProgramFormat, SimulationBackend
from tools.run_render import capture_draw
from tools.run_shader import ROOT
from tools.texture_acceptance import quad_mesh, texture_pattern


def test_repeated_legacy_invocations_reuse_definition_not_cpu_state(monkeypatch):
    original = isa_module.ISA
    definitions = []

    def counted_isa():
        definition = original()
        definitions.append(definition)
        return definition

    first = Program.from_words(assemble("movi_i32 7, r0\nstop_core\n"), format=ProgramFormat.LEGACY)
    second = Program.from_words(assemble("stop_core\n"), format=ProgramFormat.LEGACY)
    monkeypatch.setattr(isa_module, "ISA", counted_isa)
    with Driver(SimulationBackend()) as driver:
        assert driver.submit(first, buffers=[], max_steps=10).wait(5).registers[0] == 7
        assert driver.submit(second, buffers=[], max_steps=10).wait(5).registers[0] == 0
    assert len(definitions) == 1


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))


@pytest.fixture(scope="module")
def executor_artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("host-executor-captures")
    mesh = directory / "vertices.bin"
    mesh.write_bytes(quad_mesh())
    texture = directory / "texture.bin"
    texture.write_bytes(struct.pack("<II", 2, 2) + texture_pattern(2).tobytes())
    flow = capture_draw(ROOT / "tests/fixtures/host-texture-flow", BUILD, directory / "flow",
                        mesh=mesh, texture=texture, host_texture=True, fragment_status=True)
    zero = capture_draw(ROOT / "tests/fixtures/fragment-discard/zero-rgba", BUILD,
                        directory / "zero", mesh=mesh, fragment_status=True)
    return {"flow": flow, "zero": zero}


def background(width=16, height=16):
    rows, columns = np.indices((height, width))
    return np.stack((columns * 7 % 256, rows * 11 % 256,
                     (rows + columns) * 13 % 256, np.full((height, width), 127)), axis=2).astype(np.uint8)


def request_bytes(artifact, initial, *, sequence=1, blend=0, encoded=None):
    submitted = copy.deepcopy(artifact)
    submitted["sequence"] = sequence
    payload = json.dumps(submitted, separators=(",", ":")).encode() if encoded is None else encoded
    height, width = initial.shape[:2]
    return REQUEST.pack(REQUEST_MAGIC, len(payload), sequence, width, height, blend, initial.nbytes) + payload + initial.tobytes()


def receive_reply(connection):
    chunks = []
    while True:
        try:
            chunk = connection.recv(65536)
        except ConnectionResetError:
            break
        if not chunk:
            break
        chunks.append(chunk)
        assert sum(map(len, chunks)) <= REPLY.size + 128 * 128 * 4
    return b"".join(chunks)


def exchange(server, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2)
        connection.connect(str(server.path))
        connection.sendall(payload)
        connection.shutdown(socket.SHUT_WR)
        return receive_reply(connection)


def assert_error_reply(reply, *, sequence=1, width=16, height=16):
    assert len(reply) == REPLY.size
    assert REPLY.unpack(reply) == (REPLY_MAGIC, sequence, 1, width, height, 0)


@pytest.mark.parametrize("options", (
    {"path": "relative.sock"}, {"path": "/" + "x" * 108}, {"path": "/tmp/nul\0socket"},
    {"max_requests": 0}, {"max_requests": 129}, {"max_requests": True},
    {"max_steps": 0}, {"max_steps": 100000001}, {"max_steps": False},
    {"max_fragments": 0}, {"max_fragments": 65537}, {"max_fragments": True},
    {"timeout": 0}, {"timeout": 5.1}, {"timeout": True}, {"timeout": "1"},
    {"timeout": float("nan")}, {"timeout": float("inf")},
))
def test_host_server_constructor_rejects_unbounded_or_inexact_configuration(tmp_path, options):
    with pytest.raises(ValueError):
        HostExecutorServer(**{"path": tmp_path / "executor.sock", **options})


def test_server_requires_same_user_peer_credential_support(tmp_path, monkeypatch):
    monkeypatch.delattr(socket, "SO_PEERCRED")
    with pytest.raises(ValueError, match="peer credentials"):
        HostExecutorServer(tmp_path / "executor.sock")


@pytest.mark.parametrize("existing", ("file", "symlink", "socket"))
def test_existing_socket_files_and_symlinks_are_not_replaced_or_removed(tmp_path, existing):
    path = tmp_path / "executor.sock"
    listener = None
    target = tmp_path / "target"
    target.write_bytes(b"keep existing bytes")
    target.chmod(0o640)
    if existing == "file":
        path.write_bytes(b"existing endpoint file")
    elif existing == "symlink":
        path.symlink_to(target)
    else:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
    identity = path.lstat()
    try:
        with pytest.raises(OSError):
            with HostExecutorServer(path):
                pytest.fail("server replaced an existing endpoint")
        current = path.lstat()
        assert (current.st_dev, current.st_ino, current.st_mode) == (identity.st_dev, identity.st_ino, identity.st_mode)
        assert target.read_bytes() == b"keep existing bytes" and stat.S_IMODE(target.stat().st_mode) == 0o640
        if existing == "file":
            assert path.read_bytes() == b"existing endpoint file"
        elif existing == "symlink":
            assert path.is_symlink()
    finally:
        if listener is not None:
            listener.close()


@pytest.mark.parametrize("replacement", ("file", "socket"))
def test_context_cleanup_removes_only_its_own_socket_inode(tmp_path, replacement):
    path = tmp_path / "executor.sock"
    listener = None
    try:
        with HostExecutorServer(path):
            path.unlink()
            if replacement == "file":
                path.write_bytes(b"replacement survives")
            else:
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.bind(str(path))
            identity = path.lstat()
        current = path.lstat()
        assert (current.st_dev, current.st_ino, current.st_mode) == (identity.st_dev, identity.st_ino, identity.st_mode)
        if replacement == "file":
            assert path.read_bytes() == b"replacement survives"
    finally:
        if listener is not None:
            listener.close()


def test_same_uid_mode_0600_and_context_cleanup(tmp_path, executor_artifacts):
    path = tmp_path / "executor.sock"
    with HostExecutorServer(path, max_requests=1) as server:
        assert stat.S_ISSOCK(path.stat().st_mode) and stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_uid == os.geteuid()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(path))
            _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            assert uid == os.geteuid()
            connection.sendall(request_bytes(executor_artifacts["zero"], background()))
            connection.shutdown(socket.SHUT_WR)
            reply = receive_reply(connection)
        assert REPLY.unpack(reply[:REPLY.size])[2] == 0
        assert len(server.records) == 1 and server.records[0]["peer_pid"] == os.getpid()
        server._thread.join(1)
        assert not server._thread.is_alive() and not server.errors
        with pytest.raises(ValueError, match="twice"):
            server.__enter__()
    assert not path.exists()


def test_foreign_peer_uid_fails_before_any_rendering(tmp_path, monkeypatch):
    original_getsockopt = socket.socket.getsockopt

    def foreign_credentials(connection, level, option, size):
        result = original_getsockopt(connection, level, option, size)
        if level == socket.SOL_SOCKET and option == socket.SO_PEERCRED:
            pid, uid, gid = struct.unpack("3i", result)
            return struct.pack("3i", pid, uid + 1, gid)
        return result

    monkeypatch.setattr(socket.socket, "getsockopt", foreign_credentials)
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        assert exchange(server, b"") == b""
        assert not server.records and len(server.errors) == 1
        assert "same effective user" in server.errors[0]


@pytest.mark.parametrize("field,value", (
    ("magic", b"INVALID!"), ("sequence", 0), ("json_size", 0), ("json_size", MAX_JSON_BYTES + 1),
    ("width", 0), ("width", 129), ("height", 0), ("height", 129),
    ("blend", 7), ("image_size", 1023), ("image_size", 1025),
))
def test_malformed_bounded_header_closes_without_rgba_output(tmp_path, field, value):
    fields = {"magic": REQUEST_MAGIC, "json_size": 2, "sequence": 1, "width": 16,
              "height": 16, "blend": 0, "image_size": 1024}
    fields[field] = value
    header = REQUEST.pack(*(fields[key] for key in ("magic", "json_size", "sequence", "width", "height", "blend", "image_size")))
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        assert exchange(server, header) == b""
        assert not server.records and len(server.errors) == 1


@pytest.mark.parametrize("encoded", (
    b"{", b"\xff", b"null", b"[]", b'"not an artifact"',
    b'{"program_format":"legacy-mgcsim-v1","program_format":"legacy-mgcsim-v1"}',
    b'{"program_format":"legacy-mgcsim-v1","nested":{"same":1,"same":2}}',
    b'{"program_format":"legacy-mgcsim-v1","bad":NaN}',
    b'{"program_format":"legacy-mgcsim-v1","bad":Infinity}',
    b'{"program_format":"legacy-mgcsim-v1","bad":-Infinity}',
))
def test_invalid_json_duplicate_keys_and_nonfinite_constants_reply_with_no_pixels(tmp_path, encoded):
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange(server, request_bytes({}, background(), encoded=encoded))
        assert_error_reply(reply)
        assert not server.records and len(server.errors) == 1


@pytest.mark.parametrize("mutation", ("hardware", "schema", "header", "missing-layout", "missing-assembly", "missing-status"))
def test_invalid_actual_artifact_provenance_and_missing_fields_fail_closed(tmp_path, executor_artifacts, mutation):
    artifact = copy.deepcopy(executor_artifacts["flow"])
    shader = artifact["shaders"]["fs"]
    if mutation == "hardware":
        artifact["program_format"] = ProgramFormat.HARDWARE.value
    elif mutation == "schema":
        artifact["schema"] = "mgc-host-draw-v99"
    elif mutation == "header":
        shader["assembly"] = shader["assembly"].replace("ABI=1,", "ABI=2,", 1)
    elif mutation == "missing-layout":
        del shader["layout"]
    elif mutation == "missing-assembly":
        del shader["assembly"]
    elif mutation == "missing-status":
        del shader["layout"]["fragment_status"]
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        assert_error_reply(exchange(server, request_bytes(artifact, background())))
        assert not server.records and len(server.errors) == 1


@pytest.mark.parametrize("sequence", (None, False, True, 1.0, -1, 2, "1"))
def test_artifact_sequence_must_be_exact_integer_matching_wire_sequence(tmp_path, executor_artifacts, sequence):
    artifact = copy.deepcopy(executor_artifacts["zero"])
    artifact["sequence"] = sequence
    encoded = json.dumps(artifact).encode()
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange(server, request_bytes(artifact, background(), encoded=encoded))
        assert_error_reply(reply)
        assert not server.records and "sequence disagree" in server.errors[0]


@pytest.mark.parametrize("section", ("header", "json", "image"))
def test_truncated_request_is_not_a_successful_empty_frame(tmp_path, executor_artifacts, section):
    payload = request_bytes(executor_artifacts["zero"], background())
    length = {"header": REQUEST.size - 1, "json": REQUEST.size + 1, "image": len(payload) - 1}[section]
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange(server, payload[:length])
        if section == "header":
            assert reply == b""
        else:
            assert_error_reply(reply)
        assert not server.records and "truncated" in server.errors[0]


@pytest.mark.parametrize("section", ("header", "image"))
def test_incomplete_peer_obeys_absolute_input_deadline(tmp_path, executor_artifacts, section):
    payload = request_bytes(executor_artifacts["zero"], background())
    partial = payload[:4] if section == "header" else payload[:-1]
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1, timeout=0.05) as server:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            connection.connect(str(server.path))
            started = time.monotonic()
            connection.sendall(partial)
            reply = receive_reply(connection)
            assert time.monotonic() - started < 1
        if section == "header":
            assert reply == b""
        else:
            assert_error_reply(reply)
        assert not server.records and len(server.errors) == 1
        assert "timed out" in server.errors[0] or "deadline" in server.errors[0]


@pytest.mark.parametrize("blend", (0, 1))
def test_real_legacy_texture_discard_and_alpha_zero_preserve_explicit_initial_background(tmp_path, executor_artifacts, monkeypatch, blend):
    initial = background()
    original = initial.copy()
    artifact = copy.deepcopy(executor_artifacts["flow"])
    original_artifact = copy.deepcopy(artifact)
    submissions = []
    original_submit = Driver.submit

    def actual_submit(driver, program, *args, **kwargs):
        submissions.append(program.format)
        return original_submit(driver, program, *args, **kwargs)

    monkeypatch.setattr(Driver, "submit", actual_submit)
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange(server, request_bytes(artifact, initial, sequence=71, blend=blend))
        assert REPLY.unpack(reply[:REPLY.size]) == (REPLY_MAGIC, 71, 0, 16, 16, 1024)
        image = np.frombuffer(reply[REPLY.size:], dtype=np.uint8).reshape(16, 16, 4)
        expected = initial.copy()
        if not blend:
            expected[:8, 8:] = 0
            expected[8:, 8:] = [0, 255, 0, 0]
        assert np.array_equal(image, expected)
        record, = server.records
        assert np.array_equal(record["rgba"], expected)
        assert np.all(record["coverage"][:, :8] == 0) and np.all(record["coverage"][:, 8:] == 1)
        assert record["report"]["fragments"] == 256
        assert record["report"]["discarded_fragments"] == record["report"]["committed_fragments"] == 128
        assert record["report"]["blend"] == ("source-over" if blend else "disabled")
        assert not server.errors
    assert np.array_equal(initial, original) and artifact == original_artifact
    assert len(submissions) == 262 and all(target == ProgramFormat.LEGACY for target in submissions)


def test_real_all_zero_rgba_is_successful_committed_output_not_error(tmp_path, executor_artifacts):
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=2) as server:
        for sequence in (1, 2):
            reply = exchange(server, request_bytes(executor_artifacts["zero"], background(), sequence=sequence))
            assert REPLY.unpack(reply[:REPLY.size]) == (REPLY_MAGIC, sequence, 0, 16, 16, 1024)
            assert reply[REPLY.size:] == bytes(1024)
        assert len(server.records) == 2 and not server.errors
        for record in server.records:
            assert np.all(record["coverage"] == 1)
            assert record["report"]["committed_fragments"] == 256 and record["report"]["discarded_fragments"] == 0


@pytest.mark.parametrize("budget", ({"max_steps": 1}, {"max_fragments": 1}))
def test_cpu_instruction_and_fragment_budgets_return_error_without_zero_image(tmp_path, executor_artifacts, budget):
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1, **budget) as server:
        assert_error_reply(exchange(server, request_bytes(executor_artifacts["zero"], background())))
        assert not server.records and "budget" in server.errors[0]


def test_actual_infinite_legacy_execution_is_failure_not_discard_or_zero_image(tmp_path, executor_artifacts):
    artifact = copy.deepcopy(executor_artifacts["zero"])
    shader = artifact["shaders"]["fs"]
    header = shader["assembly"].splitlines()[0]
    shader["assembly"] = header + "\nnever_done:\ngoto never_done\n"
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        assert_error_reply(exchange(server, request_bytes(artifact, background())))
        assert not server.records and "max_steps" in server.errors[0]


@pytest.mark.parametrize("width,height", ((16, 16), (32, 16), (16, 32), (128, 16)))
def test_renderer_initial_framebuffer_is_copied_and_matches_geometry(executor_artifacts, width, height):
    initial = background(width, height)
    original = initial.copy()
    image, coverage, report = render_artifact(executor_artifacts["zero"], width=width, height=height, initial_image=initial)
    assert np.all(image == 0) and np.all(coverage == 1)
    assert report["committed_fragments"] == width * height
    assert np.array_equal(initial, original) and not np.shares_memory(initial, image)


@pytest.mark.parametrize("initial", (b"not pixels", np.zeros((16, 16, 4), dtype=np.float32),
                                    np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((8, 16, 4), dtype=np.uint8)))
def test_renderer_rejects_invalid_initial_image_before_execution(initial):
    with pytest.raises(ValueError, match="initial framebuffer"):
        render_artifact({}, initial_image=initial)


@pytest.mark.parametrize("timeout", (0, -1, 61, True, "1", float("nan"), float("inf")))
def test_renderer_rejects_nonfinite_and_unbounded_execution_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        render_artifact({}, execution_timeout=timeout)


@pytest.mark.parametrize("allowed_invocations", (0, 6, 7))
def test_renderer_deadline_stops_before_vertices_or_between_real_fragments(executor_artifacts, monkeypatch, allowed_invocations):
    completed_invocations = 0
    clock_started = False
    submissions = []
    original_submit = Driver.submit
    original_execute = host_render.ShaderExecutor._execute
    initial = background()
    original = initial.copy()

    def monotonic():
        nonlocal clock_started
        if not clock_started:
            clock_started = True
            return 0
        return 0 if completed_invocations < allowed_invocations else 2

    def actual_execute(executor, *args, **kwargs):
        nonlocal completed_invocations
        result = original_execute(executor, *args, **kwargs)
        completed_invocations += 1
        return result

    def actual_submit(driver, program, *args, **kwargs):
        submissions.append(program.format)
        return original_submit(driver, program, *args, **kwargs)

    monkeypatch.setattr(host_render, "time", SimpleNamespace(monotonic=monotonic))
    monkeypatch.setattr(host_render.ShaderExecutor, "_execute", actual_execute)
    monkeypatch.setattr(Driver, "submit", actual_submit)
    with pytest.raises(ValueError, match="deadline"):
        render_artifact(executor_artifacts["zero"], initial_image=initial, execution_timeout=1)
    assert len(submissions) == completed_invocations == allowed_invocations
    assert np.array_equal(initial, original)


V2_REQUEST_MAGIC = b"MGCHE02\0"
V2_REPLY_MAGIC = b"MGCRG02\0"
V2_REQUEST = struct.Struct("<8sIQIIIII")
V2_REPLY = struct.Struct("<8sQIIIII")


@pytest.fixture(scope="module")
def depth_v2_artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("host-executor-depth-captures")
    texture = directory / "texture.bin"
    texture.write_bytes(struct.pack("<II", 2, 2) + texture_pattern(2).tobytes())
    artifacts = {}
    cases = (("near", "host-texture-flow", (-0.5,), True),
             ("far", "host-texture-flow", (0.5,), True),
             ("nearer-zero", "fragment-discard/zero-rgba", (-0.75,), False),
             ("quantized-tie", "fragment-discard/zero-rgba", (2 ** -17, 0.0), False))
    for name, fixture, depths, textured in cases:
        mesh = directory / f"{name}.bin"
        mesh.write_bytes(mesh_words(depths))
        artifacts[name] = capture_draw(ROOT / "tests/fixtures" / fixture, BUILD,
                                       directory / name, mesh=mesh, depth_test=True,
                                       fragment_status=True, host_texture=textured,
                                       texture=texture if textured else None)
    return artifacts


def request_v2_bytes(artifact, initial, initial_depth, *, sequence=1, blend=0, encoded=None):
    submitted = copy.deepcopy(artifact)
    submitted["sequence"] = sequence
    payload = json.dumps(submitted, separators=(",", ":")).encode() if encoded is None else encoded
    height, width = initial.shape[:2]
    depth_bytes = initial_depth.astype("<u2").tobytes()
    return (V2_REQUEST.pack(V2_REQUEST_MAGIC, len(payload), sequence, width, height,
                            blend, initial.nbytes, len(depth_bytes)) + payload +
            initial.tobytes() + depth_bytes)


def receive_v2_reply(connection):
    chunks = []
    while True:
        try:
            chunk = connection.recv(65536)
        except ConnectionResetError:
            break
        if not chunk:
            break
        chunks.append(chunk)
        assert sum(map(len, chunks)) <= V2_REPLY.size + 128 * 128 * 6
    return b"".join(chunks)


def exchange_v2(server, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2)
        connection.connect(str(server.path))
        connection.sendall(payload)
        connection.shutdown(socket.SHUT_WR)
        return receive_v2_reply(connection)


def decode_v2_reply(reply, *, sequence=1, width=16, height=16):
    image_size, depth_size = width * height * 4, width * height * 2
    assert V2_REPLY.unpack(reply[:V2_REPLY.size]) == (V2_REPLY_MAGIC, sequence, 0,
                                                    width, height, image_size, depth_size)
    assert len(reply) == V2_REPLY.size + image_size + depth_size
    image = np.frombuffer(reply[V2_REPLY.size:V2_REPLY.size + image_size], dtype=np.uint8)
    depth = np.frombuffer(reply[V2_REPLY.size + image_size:], dtype="<u2")
    return image.reshape(height, width, 4).copy(), depth.reshape(height, width).copy()


def assert_v2_error_reply(reply, *, sequence=1, width=16, height=16):
    assert len(reply) == V2_REPLY.size
    assert V2_REPLY.unpack(reply) == (V2_REPLY_MAGIC, sequence, 1, width, height, 0, 0)


def test_v2_independent_wire_layout_and_little_endian_depth(executor_artifacts):
    assert REQUEST.size == 36 and REPLY.size == 32
    assert V2_REQUEST.size == 40 and V2_REPLY.size == 36
    initial_depth = np.full((16, 16), 0x1234, dtype=np.uint16)
    payload = request_v2_bytes(executor_artifacts["zero"], background(), initial_depth,
                               sequence=0x0102030405060708)
    assert payload[:8] == V2_REQUEST_MAGIC
    assert payload[12:20] == b"\x08\x07\x06\x05\x04\x03\x02\x01"
    assert struct.unpack_from("<I", payload, 36) == (512,)
    assert payload[-512:] == b"\x34\x12" * 256
    header = V2_REPLY.pack(V2_REPLY_MAGIC, 1, 0, 16, 16, 1024, 512)
    assert struct.unpack_from("<I", header, 32) == (512,)


@pytest.mark.parametrize("field,value", (
    ("magic", b"MGCHE03\0"), ("sequence", 0), ("json_size", 0),
    ("json_size", MAX_JSON_BYTES + 1), ("width", 0), ("width", 129),
    ("height", 0), ("height", 129), ("blend", 7),
    ("image_size", 1023), ("image_size", 1025), ("depth_size", 0),
    ("depth_size", 1), ("depth_size", 511), ("depth_size", 513),
    ("depth_size", 0xffffffff),
))
def test_v2_malformed_geometry_and_plane_lengths_close_without_either_output(tmp_path, field, value):
    fields = {"magic": V2_REQUEST_MAGIC, "json_size": 2, "sequence": 1,
              "width": 16, "height": 16, "blend": 0, "image_size": 1024, "depth_size": 512}
    fields[field] = value
    header = V2_REQUEST.pack(*(fields[key] for key in ("magic", "json_size", "sequence",
                                                      "width", "height", "blend", "image_size", "depth_size")))
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        assert exchange_v2(server, header) == b""
        assert not server.records and len(server.errors) == 1


@pytest.mark.parametrize("section", ("base-header", "depth-header", "json", "image", "depth"))
def test_v2_truncated_header_and_each_plane_never_publish_success(tmp_path, executor_artifacts, section):
    payload = request_v2_bytes(executor_artifacts["zero"], background(), np.full((16, 16), 65535, dtype=np.uint16))
    json_size = struct.unpack_from("<I", payload, 8)[0]
    length = {"base-header": REQUEST.size - 1, "depth-header": V2_REQUEST.size - 1,
              "json": V2_REQUEST.size + json_size - 1,
              "image": V2_REQUEST.size + json_size + 1023,
              "depth": len(payload) - 1}[section]
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange_v2(server, payload[:length])
        if section in ("base-header", "depth-header"):
            assert reply == b""
        else:
            assert_v2_error_reply(reply)
        assert not server.records and "truncated" in server.errors[0]


@pytest.mark.parametrize("encoded", (
    b"{", b"\xff", b"[]",
    b'{"program_format":"legacy-mgcsim-v1","nested":{"same":1,"same":2}}',
    b'{"program_format":"legacy-mgcsim-v1","bad":NaN}',
    b'{"program_format":"legacy-mgcsim-v1","bad":Infinity}',
))
def test_v2_invalid_json_replies_with_neither_color_nor_depth(tmp_path, encoded):
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange_v2(server, request_v2_bytes({}, background(), np.zeros((16, 16), dtype=np.uint16), encoded=encoded))
        assert_v2_error_reply(reply)
        assert not server.records and len(server.errors) == 1


@pytest.mark.parametrize("mutation", ("hardware", "schema", "header", "status", "sequence"))
def test_v2_actual_artifact_provenance_and_sequence_fail_closed(tmp_path, executor_artifacts, mutation):
    artifact = copy.deepcopy(executor_artifacts["flow"])
    artifact["sequence"] = 1
    if mutation == "hardware":
        artifact["program_format"] = ProgramFormat.HARDWARE.value
    elif mutation == "schema":
        artifact["schema"] = "mgc-host-draw-v99"
    elif mutation == "header":
        shader = artifact["shaders"]["fs"]
        shader["assembly"] = shader["assembly"].replace("ABI=1,", "ABI=2,", 1)
    elif mutation == "status":
        del artifact["shaders"]["fs"]["layout"]["fragment_status"]
    else:
        artifact["sequence"] = 1.0
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange_v2(server, request_v2_bytes(artifact, background(), np.full((16, 16), 65535, dtype=np.uint16),
                                                    encoded=json.dumps(artifact).encode()))
        assert_v2_error_reply(reply)
        assert not server.records and len(server.errors) == 1


def test_v2_independent_initial_depth_and_strict_less_do_not_infer_clear(tmp_path, executor_artifacts):
    initial = background()
    patterns = (np.zeros((16, 16), dtype=np.uint16),
                np.full((16, 16), 65535, dtype=np.uint16),
                np.tile(np.repeat(np.array([16384, 32768, 49151, 65535], dtype=np.uint16), 4), (16, 1)))
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=3) as server:
        for sequence, initial_depth in enumerate(patterns, 1):
            original_depth = initial_depth.copy()
            passing = initial_depth > 32768
            reply = exchange_v2(server, request_v2_bytes(executor_artifacts["zero"], initial, initial_depth, sequence=sequence))
            image, depth = decode_v2_reply(reply, sequence=sequence)
            expected_image, expected_depth = initial.copy(), initial_depth.copy()
            expected_image[passing] = 0
            expected_depth[passing] = 32768
            assert np.array_equal(image, expected_image) and np.array_equal(depth, expected_depth)
            record = server.records[-1]
            assert np.array_equal(record["coverage"], passing.astype(np.uint32))
            assert record["report"]["fragments"] == record["report"]["committed_fragments"] == int(passing.sum())
            assert record["report"]["discarded_fragments"] == 0
            assert record["report"]["depth"] == "less" and record["report"]["depth_storage"] == "Z16_UNORM"
            assert np.array_equal(initial_depth, original_depth)
        assert not server.errors
    assert np.array_equal(initial, background())


@pytest.mark.parametrize("width,height", ((16, 16), (32, 16), (16, 32)))
@pytest.mark.parametrize("blend", (0, 1))
def test_v2_top_left_depth_preserves_discard_and_commits_zero_alpha(tmp_path, depth_v2_artifacts, width, height, blend):
    initial = background(width, height)
    rows, columns = np.indices((height, width))
    initial_depth = np.where((rows + 2 * columns) % 3 == 0, 1000, 65535).astype(np.uint16)
    expected_image, expected_depth = initial.copy(), initial_depth.copy()
    eligible = initial_depth > 16384
    committed = eligible & (columns >= width // 2)
    if not blend:
        expected_image[committed & (rows < height // 2)] = 0
        expected_image[committed & (rows >= height // 2)] = [0, 255, 0, 0]
    expected_depth[committed] = 16384
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange_v2(server, request_v2_bytes(depth_v2_artifacts["near"], initial, initial_depth, blend=blend))
        image, depth = decode_v2_reply(reply, width=width, height=height)
        assert np.array_equal(image, expected_image) and np.array_equal(depth, expected_depth)
        record, = server.records
        assert np.array_equal(record["depth"], expected_depth)
        assert np.array_equal(record["coverage"], committed.astype(np.uint32))
        assert record["report"]["fragments"] == int(eligible.sum())
        assert record["report"]["committed_fragments"] == int(committed.sum())
        assert record["report"]["discarded_fragments"] == int((eligible & ~committed).sum())
        assert not server.errors
    assert np.array_equal(initial, background(width, height))
    assert np.array_equal(initial_depth, np.where((rows + 2 * columns) % 3 == 0, 1000, 65535).astype(np.uint16))


def test_v2_repeated_real_draws_preserve_depth_on_discard_and_write_on_rgba_zero(tmp_path, executor_artifacts, depth_v2_artifacts):
    image, depth = background(), np.full((16, 16), 65535, dtype=np.uint16)
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=5) as server:
        cases = ((depth_v2_artifacts["near"], 128, 128),
                 (executor_artifacts["zero"], 128, 0),
                 (executor_artifacts["zero"], 0, 0),
                 (depth_v2_artifacts["far"], 0, 0),
                 (depth_v2_artifacts["nearer-zero"], 256, 0))
        for sequence, (artifact, commits, discards) in enumerate(cases, 1):
            previous_image, previous_depth = image.copy(), depth.copy()
            reply = exchange_v2(server, request_v2_bytes(artifact, image, depth, sequence=sequence))
            image, depth = decode_v2_reply(reply, sequence=sequence)
            if sequence == 1:
                expected_image, expected_depth = background(), np.full((16, 16), 65535, dtype=np.uint16)
                expected_image[:8, 8:] = 0
                expected_image[8:, 8:] = [0, 255, 0, 0]
                expected_depth[:, 8:] = 16384
            elif sequence == 2:
                expected_image, expected_depth = previous_image.copy(), previous_depth.copy()
                expected_image[:, :8] = 0
                expected_depth[:, :8] = 32768
            elif sequence in (3, 4):
                expected_image, expected_depth = previous_image, previous_depth
            else:
                expected_image, expected_depth = np.zeros((16, 16, 4), dtype=np.uint8), np.full((16, 16), 8192, dtype=np.uint16)
            assert np.array_equal(image, expected_image) and np.array_equal(depth, expected_depth)
            record = server.records[-1]
            assert record["report"]["fragments"] == commits + discards
            assert record["report"]["committed_fragments"] == commits
            assert record["report"]["discarded_fragments"] == discards
        assert len(server.records) == 5 and not server.errors


def test_v2_quantized_less_tie_does_not_change_default_float_postcapture(tmp_path, depth_v2_artifacts):
    artifact = depth_v2_artifacts["quantized-tie"]
    float_image, float_coverage, float_report = render_artifact(artifact, depth_test=True)
    assert np.all(float_image == 0) and np.all(float_coverage == 2)
    assert float_report["committed_fragments"] == 512 and float_report["depth_storage"] == "host float"
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = exchange_v2(server, request_v2_bytes(artifact, background(), np.full((16, 16), 65535, dtype=np.uint16)))
        image, depth = decode_v2_reply(reply)
        assert np.all(image == 0) and np.all(depth == 32768)
        record, = server.records
        assert np.all(record["coverage"] == 1) and record["report"]["committed_fragments"] == 256
        assert record["report"]["depth_storage"] == "Z16_UNORM" and not server.errors


def test_v2_and_v1_requests_interleave_without_depth_or_wire_version_leak(tmp_path, executor_artifacts):
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=3) as server:
        first = exchange(server, request_bytes(executor_artifacts["zero"], background(), sequence=1))
        assert REPLY.unpack(first[:REPLY.size]) == (REPLY_MAGIC, 1, 0, 16, 16, 1024)
        assert len(first) == REPLY.size + 1024 and first[REPLY.size:] == bytes(1024)
        second = exchange_v2(server, request_v2_bytes(executor_artifacts["zero"], background(), np.zeros((16, 16), dtype=np.uint16), sequence=2))
        image, depth = decode_v2_reply(second, sequence=2)
        assert np.array_equal(image, background()) and np.all(depth == 0)
        third = exchange(server, request_bytes(executor_artifacts["zero"], background(), sequence=3))
        assert REPLY.unpack(third[:REPLY.size]) == (REPLY_MAGIC, 3, 0, 16, 16, 1024)
        assert len(third) == REPLY.size + 1024 and third[REPLY.size:] == bytes(1024)
        assert [record["report"]["depth"] for record in server.records] == ["disabled", "less", "disabled"]
        assert server.records[0]["depth"] is None and server.records[2]["depth"] is None
        assert not server.errors


@pytest.mark.parametrize("budget", ({"max_steps": 1}, {"max_fragments": 1}))
def test_v2_execution_budget_failure_has_no_color_or_depth_reply(tmp_path, executor_artifacts, budget):
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1, **budget) as server:
        reply = exchange_v2(server, request_v2_bytes(executor_artifacts["zero"], background(), np.full((16, 16), 65535, dtype=np.uint16)))
        assert_v2_error_reply(reply)
        assert not server.records and "budget" in server.errors[0]


def test_v2_incomplete_depth_payload_obeys_input_deadline(tmp_path, executor_artifacts):
    payload = request_v2_bytes(executor_artifacts["zero"], background(), np.full((16, 16), 65535, dtype=np.uint16))
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1, timeout=0.05) as server:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            connection.connect(str(server.path))
            started = time.monotonic()
            connection.sendall(payload[:-1])
            reply = receive_v2_reply(connection)
            assert time.monotonic() - started < 1
        assert_v2_error_reply(reply)
        assert not server.records and len(server.errors) == 1


@pytest.mark.parametrize("field", ("initial_depth", "depth_output"))
@pytest.mark.parametrize("array", (b"not depth", np.zeros((16, 16), dtype=np.float32),
                                 np.zeros((16, 16), dtype=np.int16), np.zeros((8, 16), dtype=np.uint16),
                                 np.zeros((16, 16, 1), dtype=np.uint16)))
def test_v2_renderer_rejects_depth_type_and_geometry_before_execution(field, array):
    with pytest.raises(ValueError, match="Z16 geometry"):
        render_artifact({}, depth_test=True, **{field: array})


@pytest.mark.parametrize("field", ("initial_depth", "depth_output"))
def test_v2_renderer_requires_explicit_enabled_depth(field):
    with pytest.raises(ValueError, match="enabled depth"):
        render_artifact({}, **{field: np.zeros((16, 16), dtype=np.uint16)})


def test_v2_renderer_rejects_readonly_output_staging():
    staging = np.zeros((16, 16), dtype=np.uint16)
    staging.flags.writeable = False
    with pytest.raises(ValueError, match="writable staging"):
        render_artifact({}, depth_test=True, depth_output=staging)


@pytest.mark.parametrize("alias", (False, True))
def test_v2_renderer_stages_depth_until_full_success_and_supports_owned_alias(executor_artifacts, alias):
    initial, initial_depth = background(), np.full((16, 16), 65535, dtype=np.uint16)
    staging = initial_depth if alias else np.full((16, 16), 0x1234, dtype=np.uint16)
    original_depth = initial_depth.copy()
    image, coverage, report = render_artifact(executor_artifacts["zero"], depth_test=True,
                                             initial_image=initial, initial_depth=initial_depth, depth_output=staging)
    assert np.all(image == 0) and np.all(coverage == 1) and np.all(staging == 32768)
    assert report["committed_fragments"] == 256 and report["depth_storage"] == "Z16_UNORM"
    if not alias:
        assert np.array_equal(initial_depth, original_depth)
    assert np.array_equal(initial, background())


@pytest.mark.parametrize("options", ({"max_steps": 1}, {"max_fragments": 1}))
@pytest.mark.parametrize("alias", (False, True))
def test_v2_renderer_budget_failure_leaves_both_owned_planes_and_staging_unchanged(executor_artifacts, options, alias):
    initial, initial_depth = background(), np.full((16, 16), 65535, dtype=np.uint16)
    staging = initial_depth if alias else np.full((16, 16), 0x1234, dtype=np.uint16)
    original_initial, original_depth, original_staging = initial.copy(), initial_depth.copy(), staging.copy()
    with pytest.raises(ValueError, match="budget"):
        render_artifact(executor_artifacts["zero"], depth_test=True, initial_image=initial,
                        initial_depth=initial_depth, depth_output=staging, **options)
    assert np.array_equal(initial, original_initial) and np.array_equal(initial_depth, original_depth)
    assert np.array_equal(staging, original_staging)


@pytest.mark.parametrize("allowed_invocations", (0, 7, 262))
def test_v2_renderer_deadline_failure_does_not_publish_partial_depth(executor_artifacts, monkeypatch, allowed_invocations):
    completed_invocations = 0
    clock_started = False
    submissions = []
    original_submit = Driver.submit
    original_execute = host_render.ShaderExecutor._execute
    initial, initial_depth = background(), np.full((16, 16), 65535, dtype=np.uint16)
    staging = np.full((16, 16), 0x1234, dtype=np.uint16)

    def monotonic():
        nonlocal clock_started
        if not clock_started:
            clock_started = True
            return 0
        return 0 if completed_invocations < allowed_invocations else 2

    def actual_execute(executor, *args, **kwargs):
        nonlocal completed_invocations
        result = original_execute(executor, *args, **kwargs)
        completed_invocations += 1
        return result

    def actual_submit(driver, program, *args, **kwargs):
        submissions.append(program.format)
        return original_submit(driver, program, *args, **kwargs)

    monkeypatch.setattr(host_render, "time", SimpleNamespace(monotonic=monotonic))
    monkeypatch.setattr(host_render.ShaderExecutor, "_execute", actual_execute)
    monkeypatch.setattr(Driver, "submit", actual_submit)
    with pytest.raises(ValueError, match="deadline"):
        render_artifact(executor_artifacts["zero"], depth_test=True, initial_image=initial,
                        initial_depth=initial_depth, depth_output=staging, execution_timeout=1)
    assert len(submissions) == completed_invocations == allowed_invocations
    assert np.array_equal(initial, background()) and np.all(initial_depth == 65535) and np.all(staging == 0x1234)
