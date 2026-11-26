"""Real C resource/IPC guards; patterned peers test transport, not rendering.

Separate cases use the existing managed executor to execute actual captured NIR.
Full clears follow pipe/p_context.h:719 independently of bound draw masks;
Mesa's st_cb_clear.c resolves GL masks before selecting PIPE_CLEAR_* buffers.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import struct
import subprocess
import threading
import time

import pytest

from tools.host_executor import HostExecutorServer


ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
DRIVER = ROOT / "mesa/src/gallium/drivers/mgc"
SOURCE = ROOT / "tests/fixtures/native-host-executor/guard.c"
REQUEST = struct.Struct("<8sIQIIII")
REPLY = struct.Struct("<8sQIIII")
REQUEST_DEPTH_SIZE = struct.Struct("<I")
REPLY_DEPTH = struct.Struct("<8sQIIIII")
COMMIT = re.compile(r"^MGC_HOST_EXECUTOR_COMMIT sequence=(\d+) width=(\d+) height=(\d+)$", re.M)
STATE_REJECTIONS = (
    "blend_mask", "blend_logicop", "blend_dither", "blend_alpha_coverage", "blend_independent",
    "blend_factors", "raster_scissor", "raster_cull", "raster_discard", "raster_fill",
    "raster_offset", "raster_multisample", "raster_clip", "raster_halfz", "raster_edge",
    "raster_center", "raster_clamp", "depth_enabled", "depth_bounds", "stencil_front",
    "stencil_back", "alpha_test", "viewport_partial", "viewport_flip", "viewport_nan",
    "viewport_swizzle", "viewport_depth", "null_blend", "null_raster", "null_depth",
    "missing_viewport", "sample_mask",
)
RESOURCE_REJECTIONS = (
    "rt_large", "rt_format", "rt_rectangle", "rt_array", "rt_mips", "rt_shared",
    "rt_samples", "rt_storage_samples", "rt_persistent", "rt_stride", "rt_offset",
    "rt_null_data", "rt_size", "rt_alias", "rt_mapped", "rt_depth_attachment",
)
REPLY_REJECTIONS = (
    "reply_magic", "reply_sequence", "reply_status", "reply_unknown_status", "reply_width",
    "reply_height", "reply_size", "reply_huge_size", "reply_short_header", "reply_short_image",
    "reply_trailing",
)


def checked(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, timeout=120, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def executor_guard(tmp_path_factory):
    if not (BUILD / "compile_commands.json").is_file():
        pytest.skip("configure and build Mesa with mgc first")
    environment = dict(os.environ)
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    ninja = shutil.which("ninja", path=environment["PATH"])
    if not ninja:
        pytest.skip("ninja is required to read Mesa's existing link command")
    entry = next(item for item in json.loads((BUILD / "compile_commands.json").read_text())
                 if item["file"].endswith("/mgc_context.c"))
    arguments = shlex.split(entry["command"])
    flags = []
    cursor = 1
    while cursor < len(arguments):
        argument = arguments[cursor]
        if argument in ("-o", "-c", "-MF", "-MQ", "-MT"):
            cursor += 2
        elif argument == "-MD":
            cursor += 1
        else:
            flags.append(argument)
            cursor += 1
    flags += ["-pthread"]
    sanitized = environment.get("MGC_RUNTIME_SANITIZE") == "1"
    if sanitized:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    temporary = tmp_path_factory.mktemp("native-executor-guards")
    object_file = temporary / "guard.o"
    checked([arguments[0], *flags, "-I" + str(DRIVER), "-o", str(object_file),
             "-c", str(SOURCE)], cwd=BUILD)
    commands = checked([ninja, "-C", str(BUILD), "-t", "commands",
                        "src/gallium/drivers/mgc/mgc_test_arith"], env=environment)
    link = shlex.split(commands.strip().splitlines()[-1])
    helper = temporary / "guard"
    link[link.index("-o") + 1] = str(helper)
    original = "src/gallium/drivers/mgc/mgc_test_arith.p/mgc_test_arith.c.o"
    link[link.index(original)] = str(object_file)
    objects = []
    for filename in ("mgc_context.c", "mgc_resource.c", "mgc_screen.c", "mgc_state.c",
                     "mgc_compiler.c", "mgc_vector_compiler.c", "mgc_host_executor.c"):
        output = temporary / (filename + ".o")
        checked([arguments[0], *flags, "-o", str(output), "-c", str(DRIVER / filename)], cwd=BUILD)
        objects.append(str(output))
    position = link.index("src/gallium/drivers/mgc/libmgc.a")
    link[position:position + 1] = objects
    link += ["-pthread", "-Wl,--wrap=getsockopt"]
    if sanitized:
        link += ["-fsanitize=address,undefined"]
    checked(link, cwd=BUILD)
    return helper


@pytest.fixture
def socket_path(tmp_path_factory):
    return tmp_path_factory.mktemp("he") / "s"


def run_guard(helper, directory, path, case, *, capture=True, enabled=True, peer=None):
    environment = dict(os.environ)
    for key in ("MGC_HOST_TEXTURE", "MGC_FRAGMENT_STATUS", "MGC_COMPILE_TARGET", "MGC_HOST_MESH",
                "MGC_ASM_DUMP", "MGC_NIR_DUMP", "MGC_HOST_EXECUTOR_SOCKET", "MGC_HOST_CAPTURE_DIR",
                "MGC_TEST_PEER", "MGC_HOST_INSTANCING", "MGC_HOST_INSTANCE_DATA",
                "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T", "MGC_HOST_GLSL_PROFILE"):
        environment.pop(key, None)
    if case.startswith("abi3_") and case != "abi3_without_optin":
        environment["MGC_HOST_INSTANCING"] = "1"
    if capture:
        environment["MGC_HOST_CAPTURE_DIR"] = str(directory)
    if enabled:
        environment["MGC_HOST_EXECUTOR_SOCKET"] = str(path)
    if peer:
        environment["MGC_TEST_PEER"] = peer
    result = subprocess.run([str(helper), case], env=environment,
                            text=True, capture_output=True, timeout=12)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout), result.stderr


def receive(connection, size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            return bytes(data)
        data.extend(chunk)
    return bytes(data)


@contextmanager
def patterned_peer(path, mode="success"):
    records, errors = [], []
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    os.chmod(path, 0o600)
    listener.listen(1)
    listener.settimeout(8)

    def serve():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(8)
                if mode == "write_timeout":
                    time.sleep(5.3)
                    return
                header = receive(connection, REQUEST.size)
                if not header:
                    records.append({"no_payload": True})
                    return
                magic, json_size, sequence, width, height, blend, image_size = REQUEST.unpack(header)
                assert magic in (b"MGCHE01\0", b"MGCHE02\0") and sequence != 0
                depth_size = REQUEST_DEPTH_SIZE.unpack(receive(connection, 4))[0] if magic == b"MGCHE02\0" else 0
                assert 1 <= json_size <= 12 * 1024 * 1024
                assert 1 <= width <= 128 and 1 <= height <= 128
                assert image_size == width * height * 4 and blend in (0, 1)
                assert depth_size == (width * height * 2 if magic == b"MGCHE02\0" else 0)
                raw_json = receive(connection, json_size)
                artifact = json.loads(raw_json)
                initial = receive(connection, image_size)
                initial_depth = receive(connection, depth_size)
                assert len(initial) == image_size and artifact["sequence"] == sequence
                assert len(initial_depth) == depth_size
                assert connection.recv(1) == b""
                records.append({"raw_json": raw_json, "artifact": artifact, "sequence": sequence,
                                "width": width, "height": height, "blend": blend, "initial": initial,
                                "initial_depth": initial_depth, "magic": magic})
                if mode == "read_timeout":
                    time.sleep(5.3)
                    return
                if mode in ("pin_success", "pin_error", "pin_revalidate"):
                    time.sleep(0.35)
                image = bytes(image_size) if mode == "zero" else bytes(
                    (index * 29 + 11) & 255 for index in range(image_size))
                depth_image = bytes(depth_size) if mode == "zero" else bytes(
                    (index * 31 + 7) & 255 for index in range(depth_size))
                reply_magic, reply_sequence, status = b"MGCRG02\0" if depth_size else b"MGCRG01\0", sequence, 0
                reply_width, reply_height, reply_size = width, height, image_size
                reply_depth_size = depth_size
                if mode == "reply_magic": reply_magic = b"BADRG01\0"
                if mode == "reply_sequence": reply_sequence += 1
                if mode in ("reply_status", "pin_error"):
                    status, reply_size, image = 1, 0, b""
                    reply_depth_size, depth_image = 0, b""
                if mode == "reply_unknown_status": status = 2
                if mode == "reply_width": reply_width += 1
                if mode == "reply_height": reply_height += 1
                if mode == "reply_size": reply_size -= 1
                if mode == "reply_huge_size": reply_size = 0xFFFFFFFF
                if mode == "reply_wrong_version": reply_magic = b"MGCRG01\0"
                if mode == "reply_depth_size": reply_depth_size -= 1
                if mode == "reply_huge_depth_size": reply_depth_size = 0xFFFFFFFF
                header_fields = (reply_magic, reply_sequence, status, reply_width, reply_height, reply_size)
                reply_header = REPLY_DEPTH.pack(*header_fields, reply_depth_size) if depth_size else REPLY.pack(*header_fields)
                reply = reply_header + image + depth_image
                if mode == "reply_short_header": reply = reply[:10]
                if mode == "reply_short_image": reply = reply[:-1]
                if mode == "reply_short_color": reply = reply[:len(reply_header) + image_size - 1]
                if mode == "reply_short_depth": reply = reply[:-1]
                if mode == "reply_short_depth_header": reply = reply[:34]
                if mode == "depth_timeout": reply = reply[:len(reply_header) + image_size]
                if mode == "reply_trailing": reply += b"x"
                if mode == "chunked":
                    for offset in range(0, len(reply), 3):
                        connection.sendall(reply[offset:offset + 3])
                        time.sleep(0.001)
                else:
                    connection.sendall(reply)
                if mode in ("eof_timeout", "depth_timeout"): time.sleep(5.3)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield records
    finally:
        thread.join(9)
        listener.close()
        path.unlink(missing_ok=True)
        assert not thread.is_alive()
        assert not errors, errors


def initial_storage(status):
    return bytes((index * 17 + 3) & 255 for index in range(len(bytes.fromhex(status["storage_hex"]))))


def initial_depth_storage(status):
    return bytes((index * 23 + 5) & 255 for index in range(len(bytes.fromhex(status["depth_hex"]))))


def geometry(case):
    case = case.removeprefix("z_")
    return {"small": (1, 1), "maximum": (128, 128)}.get(case, (3, 2))


def image_storage(status, case, image):
    width, height = geometry(case)
    output = bytearray(initial_storage(status))
    offset, stride = (4, 32) if case in ("padded", "z_padded") else (0, width * 4)
    for row in range(height):
        destination = offset + (height - 1 - row) * stride
        output[destination:destination + width * 4] = image[row * width * 4:(row + 1) * width * 4]
    return bytes(output)


def depth_storage(status, case, image):
    width, height = geometry(case)
    output = bytearray(initial_depth_storage(status))
    offset, stride = (2, 16) if case == "z_padded" else (0, width * 2)
    for row in range(height):
        destination = offset + (height - 1 - row) * stride
        output[destination:destination + width * 2] = image[row * width * 2:(row + 1) * width * 2]
    return bytes(output)


def assert_rejected(status, stderr, *, draws=True):
    assert status["accepted"] == 0 and status["rejected"] == int(draws)
    assert status["error"] != 0 and status["callbacks"] == 0
    assert bytes.fromhex(status["storage_hex"]) == initial_storage(status)
    assert bytes.fromhex(status["depth_hex"]) == initial_depth_storage(status)
    assert COMMIT.findall(stderr) == []
    assert "MGC_HOST_EXECUTOR_FAILURE" in stderr


@pytest.mark.parametrize("case", ("success", "zero", "chunked", "small", "maximum", "padded", "source_over"))
def test_actual_c_owned_feedback_orientation_and_lifetime(executor_guard, tmp_path, socket_path, case):
    with patterned_peer(socket_path, case) as records:
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    width, height = geometry(case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert COMMIT.findall(stderr) == [(str(status["sequence"]), str(width), str(height))]
    record, = records
    assert record["sequence"] == status["sequence"] and record["blend"] == int(case == "source_over")
    image = bytes(width * height * 4) if case == "zero" else bytes(
        (index * 29 + 11) & 255 for index in range(width * height * 4))
    assert bytes.fromhex(status["storage_hex"]) == image_storage(status, case, image)
    expected_initial = initial_storage(status)
    offset, stride = (4, 32) if case == "padded" else (0, width * 4)
    assert record["initial"] == b"".join(expected_initial[offset + row * stride:offset + row * stride + width * 4]
                                          for row in reversed(range(height)))
    capture, = tmp_path.glob("*.json")
    assert capture.read_bytes() == record["raw_json"]


@pytest.mark.parametrize("case", ("success", "source_over"))
def test_actual_existing_managed_executor_to_c_resource(executor_guard, tmp_path, socket_path, case):
    with HostExecutorServer(socket_path, max_requests=1) as server:
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, case, capture=False)
    assert not server.errors and len(server.records) == 1
    record = server.records[0]
    assert status["accepted"] == 1 and status["error"] == 0
    assert COMMIT.findall(stderr) == [(str(record["sequence"]), "3", "2")]
    assert bytes.fromhex(status["storage_hex"]) == image_storage(status, case, record["rgba"].tobytes())
    assert list(tmp_path.iterdir()) == []


def test_explicit_abi3_managed_executor_to_actual_c_resource(executor_guard, tmp_path, socket_path):
    with HostExecutorServer(socket_path, max_requests=1) as server:
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, "abi3_valid", capture=False)
    assert not server.errors and len(server.records) == 1
    record = server.records[0]
    assert record["artifact"]["schema"] == "mgc-host-draw-v3"
    assert record["artifact"]["shader_abi_version"] == 3
    assert status["accepted"] == 1 and status["error"] == 0
    assert COMMIT.findall(stderr) == [(str(record["sequence"]), "3", "2")]
    assert bytes.fromhex(status["storage_hex"]) == image_storage(status, "success", record["rgba"].tobytes())


@pytest.mark.parametrize("case", ("abi3_fs_abi", "abi3_fs_option", "abi3_fs_hardware", "abi3_without_optin"))
def test_instancing_abi_mismatch_never_requests_or_commits(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert_rejected(status, stderr)


@pytest.mark.parametrize("case", STATE_REJECTIONS + RESOURCE_REJECTIONS + ("hardware_target",))
def test_direct_c_state_and_resource_guards(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert_rejected(status, stderr)
    if case in RESOURCE_REJECTIONS:
        assert "render target" in stderr
    if case == "blend_dither":
        assert "blend.dither" in stderr


@pytest.mark.parametrize("case", REPLY_REJECTIONS)
def test_malformed_reply_never_partially_commits(executor_guard, tmp_path, socket_path, case):
    with patterned_peer(socket_path, case) as records:
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert len(records) == 1
    assert_rejected(status, stderr)


@pytest.mark.parametrize("peer", ("uid", "size", "error"))
def test_direct_c_peer_authentication_before_payload(executor_guard, tmp_path, socket_path, peer):
    with patterned_peer(socket_path) as records:
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, "success", peer=peer)
    assert records == [{"no_payload": True}]
    assert_rejected(status, stderr)


@pytest.mark.parametrize("path", ("", "relative", "/" + "x" * 107, "missing"))
def test_bad_executor_environment_fails_closed(executor_guard, tmp_path, socket_path, path):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path if path == "missing" else path, "success")
    assert_rejected(status, stderr)


@pytest.mark.parametrize("case", ("unit_zero_sequence", "unit_wrong_sequence", "unit_json_empty",
                                 "unit_json_large", "unit_json_null"))
def test_direct_c_sequence_and_twelve_mib_bounds(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert_rejected(status, stderr, draws=False)


@pytest.mark.parametrize("case", ("read_timeout", "write_timeout", "eof_timeout"))
def test_bounded_read_write_and_whole_reply_deadlines(executor_guard, tmp_path, socket_path, case):
    started = time.monotonic()
    with patterned_peer(socket_path, case):
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert 4.5 <= time.monotonic() - started < 8
    assert_rejected(status, stderr)


@pytest.mark.parametrize("case", ("pin_success", "pin_error"))
def test_other_context_cannot_map_clear_or_copy_during_ipc(executor_guard, tmp_path, socket_path, case):
    with patterned_peer(socket_path, case):
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert status["pin_rejections"] == 6 and status["pin_recovered"] is True
    if case == "pin_error":
        assert_rejected(status, stderr)
    else:
        assert status["accepted"] == 1 and status["error"] == 0
        assert len(COMMIT.findall(stderr)) == 1


@pytest.mark.parametrize("case", ("clear_valid", "clear_zero", "clear_empty", "clear_unbound", "clear_mask"))
def test_explicit_cpu_rgba8_full_clear(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert status["error"] == 0 and status["accepted"] == 0 and status["rejected"] == 0
    expected = initial_storage(status) if case == "clear_empty" else (
        bytes(24) if case == "clear_zero" else bytes((0, 128, 255, 255)) * 6)
    assert bytes.fromhex(status["storage_hex"]) == expected
    assert not COMMIT.findall(stderr)


@pytest.mark.parametrize("case", ("clear_depth", "clear_stencil", "clear_other_color", "clear_scissor",
                                 "clear_null", "clear_nan", "clear_shared", "clear_alias",
                                 "clear_mapped", "clear_depth_attachment"))
def test_unsupported_clear_leaves_actual_bytes_unchanged(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert_rejected(status, stderr, draws=False)


@pytest.mark.parametrize("case", ("capture_only", "custom", "default_clear", "custom_clear"))
def test_default_capture_and_public_callback_behavior_unchanged(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, "relative", case,
                               enabled=case in ("custom", "custom_clear"))
    assert bytes.fromhex(status["storage_hex"]) == initial_storage(status)
    assert not COMMIT.findall(stderr)
    if case in ("default_clear", "custom_clear"):
        assert status["error"] != 0 and status["accepted"] == 0
        assert "MGC_HOST_EXECUTOR_FAILURE" not in stderr
    else:
        assert status["error"] == 0 and status["accepted"] == 1
        assert status["callbacks"] == int(case == "custom")


@pytest.mark.parametrize("case", ("z_success", "z_zero", "z_chunked", "z_small", "z_maximum",
                                 "z_padded", "z_source_over"))
def test_v2_paired_owned_resources_orientation_and_lifetime(executor_guard, tmp_path, socket_path, case):
    with patterned_peer(socket_path, case.removeprefix("z_")) as records:
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    width, height = geometry(case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert COMMIT.findall(stderr) == [(str(status["sequence"]), str(width), str(height))]
    record, = records
    assert record["magic"] == b"MGCHE02\0" and record["sequence"] == status["sequence"]
    assert record["blend"] == int(case == "z_source_over")
    color_image = bytes(width * height * 4) if case == "z_zero" else bytes(
        (index * 29 + 11) & 255 for index in range(width * height * 4))
    depth_image = bytes(width * height * 2) if case == "z_zero" else bytes(
        (index * 31 + 7) & 255 for index in range(width * height * 2))
    assert bytes.fromhex(status["storage_hex"]) == image_storage(status, case, color_image)
    assert bytes.fromhex(status["depth_hex"]) == depth_storage(status, case, depth_image)
    for initial, key, blocksize, offset, stride in (
        (initial_storage(status), "initial", 4, 4 if case == "z_padded" else 0,
         32 if case == "z_padded" else width * 4),
        (initial_depth_storage(status), "initial_depth", 2, 2 if case == "z_padded" else 0,
         16 if case == "z_padded" else width * 2),
    ):
        assert record[key] == b"".join(initial[offset + row * stride:offset + row * stride + width * blocksize]
                                        for row in reversed(range(height)))
    capture, = tmp_path.glob("*.json")
    assert capture.read_bytes() == record["raw_json"]


@pytest.mark.parametrize("case", ("z_success", "z_source_over"))
def test_v2_actual_existing_executor_commits_color_and_z16(executor_guard, tmp_path, socket_path, case):
    with HostExecutorServer(socket_path, max_requests=1) as server:
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, case, capture=False)
    assert not server.errors and len(server.records) == 1
    record = server.records[0]
    assert status["accepted"] == 1 and status["error"] == 0
    assert COMMIT.findall(stderr) == [(str(record["sequence"]), "3", "2")]
    assert record["depth"] is not None and record["depth"].dtype.name == "uint16"
    assert bytes.fromhex(status["storage_hex"]) == image_storage(status, case, record["rgba"].tobytes())
    assert bytes.fromhex(status["depth_hex"]) == depth_storage(status, case, record["depth"].astype("<u2").tobytes())
    assert bytes.fromhex(status["depth_hex"]) != initial_depth_storage(status)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", (
    "z_missing", "z_disabled", "z_write_disabled", "z_equal", "z_lequal", "z_bounds", "z_stencil",
    "z_format", "z_surface_format", "z_size", "z_large", "z_array", "z_mips", "z_samples",
    "z_storage_samples", "z_shared", "z_persistent", "z_stride", "z_offset", "z_null_data",
    "z_allocation_size", "z_alias", "z_mapped", "z_pinned",
))
def test_direct_c_depth_targets_and_less_write_contract(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert_rejected(status, stderr)


@pytest.mark.parametrize("mode", REPLY_REJECTIONS + (
    "reply_wrong_version", "reply_depth_size", "reply_huge_depth_size", "reply_short_color",
    "reply_short_depth", "reply_short_depth_header",
))
def test_v2_malformed_response_never_commits_either_resource(executor_guard, tmp_path, socket_path, mode):
    with patterned_peer(socket_path, mode) as records:
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, "z_success")
    assert len(records) == 1 and records[0]["magic"] == b"MGCHE02\0"
    assert_rejected(status, stderr)


@pytest.mark.parametrize("case", ("z_pin_success", "z_pin_error", "z_pin_depth_geometry",
                                 "z_pin_depth_identity", "z_pin_color_identity", "z_pin_state"))
def test_v2_dual_pins_and_staged_revalidation(executor_guard, tmp_path, socket_path, case):
    mode = "pin_success" if case == "z_pin_success" else "pin_error" if case == "z_pin_error" else "pin_revalidate"
    with patterned_peer(socket_path, mode):
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert status["pin_rejections"] == 11 and status["pin_recovered"] is True
    if case != "z_pin_success":
        assert_rejected(status, stderr)
    else:
        assert status["accepted"] == 1 and status["error"] == 0
        assert len(COMMIT.findall(stderr)) == 1


def test_v2_full_color_reply_waits_for_depth_with_bounded_deadline(executor_guard, tmp_path, socket_path):
    started = time.monotonic()
    with patterned_peer(socket_path, "depth_timeout"):
        status, stderr = run_guard(executor_guard, tmp_path, socket_path, "z_success")
    assert 4.5 <= time.monotonic() - started < 8
    assert_rejected(status, stderr)


@pytest.mark.parametrize("case", ("clear_z_full", "clear_z_zero", "clear_z_half", "clear_z_coloronly",
                                 "clear_z_depthonly", "clear_z_nullcolor", "clear_z_mask",
                                 "clear_z_write_disabled", "clear_z_unbound"))
def test_cpu_full_z16_clear_ignores_draw_masks_and_quantizes_selected_buffers(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert status["error"] == 0 and status["accepted"] == 0 and status["rejected"] == 0
    expected_color = initial_storage(status) if case in ("clear_z_depthonly", "clear_z_nullcolor") else bytes((0, 128, 255, 255)) * 6
    depth_value = 0 if case == "clear_z_zero" else 32768 if case == "clear_z_half" else 65535
    expected_depth = initial_depth_storage(status) if case == "clear_z_coloronly" else struct.pack("<H", depth_value) * 6
    assert bytes.fromhex(status["storage_hex"]) == expected_color
    assert bytes.fromhex(status["depth_hex"]) == expected_depth
    assert not COMMIT.findall(stderr)


@pytest.mark.parametrize("case", (
    "clear_z_nan", "clear_z_inf", "clear_z_below", "clear_z_above", "clear_z_color_nan",
    "clear_z_format", "clear_z_mapped", "clear_z_color_mapped", "clear_z_pinned", "clear_z_color_pinned",
    "clear_z_stencil", "clear_z_scissor",
))
def test_c_clear_validation_precedes_all_color_and_depth_writes(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, socket_path, case)
    assert_rejected(status, stderr, draws=False)


@pytest.mark.parametrize("case", ("default_z_clear", "custom_z_clear"))
def test_default_or_custom_callback_does_not_enable_depth_clear(executor_guard, tmp_path, socket_path, case):
    status, stderr = run_guard(executor_guard, tmp_path, "relative", case, enabled=case == "custom_z_clear")
    assert status["error"] != 0 and status["accepted"] == 0
    assert bytes.fromhex(status["storage_hex"]) == initial_storage(status)
    assert bytes.fromhex(status["depth_hex"]) == initial_depth_storage(status)
    assert not COMMIT.findall(stderr) and "MGC_HOST_EXECUTOR_FAILURE" not in stderr
