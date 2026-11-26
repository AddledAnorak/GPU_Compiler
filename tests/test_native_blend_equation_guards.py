"""Real C blend-state/Unix-transport guards; patterned replies do not render."""

from contextlib import contextmanager
import json
import os
import socket
import threading
import time

import pytest

import test_native_host_executor_guards as native_guards
from test_native_host_executor_guards import (
    COMMIT,
    REPLY,
    REPLY_DEPTH,
    REQUEST,
    REQUEST_DEPTH_SIZE,
    ROOT,
    assert_rejected,
    depth_storage,
    executor_guard as build_executor_guard,
    geometry,
    image_storage,
    initial_depth_storage,
    initial_storage,
    receive,
    run_guard,
    socket_path,
)


UNIT_MODES = (2, 3, 4, 5, 6)
FACTOR_REJECTIONS = tuple(
    f"factor_{field}_{factor}"
    for field in ("rgb_src", "rgb_dst", "alpha_src", "alpha_dst")
    for factor in ("zero", "src_alpha", "inv_src_alpha")
)
FEATURE_REJECTIONS = (
    "color_mask", "logicop", "dither", "coverage", "coverage_dither", "alpha_to_one",
    "independent", "max_rt", "advanced", "coherent", "extra_target",
)


@pytest.fixture(scope="module")
def blend_executor_guard(tmp_path_factory):
    hooks = ROOT / "tests/fixtures/native-host-executor/harness.c"
    contents = native_guards.SOURCE.read_text()
    replacements = (
        ('#include "../texture-capture/guard.c"',
         f'#include "{ROOT / "tests/fixtures/texture-capture/guard.c"}"'),
        ("#undef main", f'#undef main\n#include "{hooks}"'),
        ("bind_draw_state(pipe, name, width, height);",
         "bind_draw_state(pipe, name, width, height);\n"
         "   configure_native_blend_equation(pipe);"),
        ("unsigned saved_function = probe->owner->depth_stencil_state.depth_func;",
         "unsigned saved_function = probe->owner->depth_stencil_state.depth_func;\n"
         "   struct pipe_blend_state saved_blend = probe->owner->blend_state;"),
        ('if (is_case(probe->name, "z_pin_state")) probe->owner->depth_stencil_state.depth_func = PIPE_FUNC_ALWAYS;',
         'if (is_case(probe->name, "z_pin_state")) probe->owner->depth_stencil_state.depth_func = PIPE_FUNC_ALWAYS;\n'
         "   change_native_blend_during_ipc(probe->owner, probe->name);"),
        ("probe->owner->depth_stencil_state.depth_func = saved_function;",
         "probe->owner->depth_stencil_state.depth_func = saved_function;\n"
         "   probe->owner->blend_state = saved_blend;"),
    )
    for original, replacement in replacements:
        assert contents.count(original) == 1, original
        contents = contents.replace(original, replacement, 1)
    source = tmp_path_factory.mktemp("native-blend-equation-source") / "guard.c"
    source.write_text(contents)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native_guards, "SOURCE", source)
        return build_executor_guard.__wrapped__(tmp_path_factory)


def run_blend_guard(helper, directory, path, case, blend_mode, mutation="none", **kwargs):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MGC_TEST_BLEND_MODE", str(blend_mode))
        patch.setenv("MGC_TEST_BLEND_MUTATION", mutation)
        return run_guard(helper, directory, path, case, **kwargs)


@contextmanager
def blend_peer(path, expected_mode, response="success"):
    assert expected_mode in range(7)
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
                magic, json_size, sequence, width, height, blend, image_size = REQUEST.unpack(
                    receive(connection, REQUEST.size))
                assert magic in (b"MGCHE01\0", b"MGCHE02\0") and sequence > 0
                assert blend == expected_mode
                assert 1 <= json_size <= 12 * 1024 * 1024
                assert 1 <= width <= 128 and 1 <= height <= 128
                assert image_size == width * height * 4
                depth_size = REQUEST_DEPTH_SIZE.unpack(receive(connection, 4))[0] if magic == b"MGCHE02\0" else 0
                assert depth_size == (width * height * 2 if magic == b"MGCHE02\0" else 0)
                raw_json = receive(connection, json_size)
                artifact = json.loads(raw_json)
                initial = receive(connection, image_size)
                initial_depth = receive(connection, depth_size)
                assert len(initial) == image_size and len(initial_depth) == depth_size
                assert artifact["sequence"] == sequence and artifact["shader_abi_version"] == 1
                assert artifact["schema"] == "mgc-host-draw-v1" and "textures" not in artifact
                assert connection.recv(1) == b""
                image = bytes((index * 29 + 11) & 255 for index in range(image_size))
                depth_image = bytes((index * 31 + 7) & 255 for index in range(depth_size))
                records.append(dict(raw_json=raw_json, artifact=artifact, sequence=sequence,
                                    width=width, height=height, blend=blend, initial=initial,
                                    initial_depth=initial_depth, image=image, depth_image=depth_image))
                if response in ("pin_success", "pin_error", "pin_revalidate"):
                    time.sleep(0.35)
                reply_magic = b"MGCRG02\0" if depth_size else b"MGCRG01\0"
                reply_sequence, status = sequence, 0
                reply_size, reply_depth_size = image_size, depth_size
                if response in ("reply_status", "pin_error"):
                    status, reply_size, reply_depth_size = 1, 0, 0
                    image = depth_image = b""
                if response == "reply_sequence":
                    reply_sequence += 1
                if response == "reply_unknown_status":
                    status = 2
                if response == "reply_depth_size":
                    assert depth_size
                    reply_depth_size -= 1
                fields = (reply_magic, reply_sequence, status, width, height, reply_size)
                header = REPLY_DEPTH.pack(*fields, reply_depth_size) if depth_size else REPLY.pack(*fields)
                reply = header + image + depth_image
                if response == "reply_short_color":
                    reply = reply[:len(header) + image_size - 1]
                if response == "reply_short_depth":
                    assert depth_size
                    reply = reply[:-1]
                if response == "reply_trailing":
                    reply += b"x"
                connection.sendall(reply)
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


def assert_committed(status, stderr, record, case, blend_mode, directory):
    width, height = geometry(case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["callbacks"] == 0
    assert COMMIT.findall(stderr) == [(str(status["sequence"]), str(width), str(height))]
    assert record["blend"] == blend_mode and record["sequence"] == status["sequence"]
    assert bytes.fromhex(status["storage_hex"]) == image_storage(status, case, record["image"])
    if case.startswith("z_"):
        assert bytes.fromhex(status["depth_hex"]) == depth_storage(status, case, record["depth_image"])
    else:
        assert status["depth_hex"] == "" and record["initial_depth"] == b""
    for initial, key, blocksize, offset, stride in (
        (initial_storage(status), "initial", 4, 4 if case in ("padded", "z_padded") else 0,
         32 if case in ("padded", "z_padded") else width * 4),
        (initial_depth_storage(status), "initial_depth", 2, 2 if case == "z_padded" else 0,
         16 if case == "z_padded" else width * 2),
    ):
        expected = b"".join(initial[offset + row * stride:offset + row * stride + width * blocksize]
                            for row in reversed(range(height))) if initial else b""
        assert record[key] == expected
    capture, = directory.glob("*.json")
    assert capture.read_bytes() == record["raw_json"]
    assert "blend" not in record["artifact"] and "blend_mode" not in record["artifact"]


@pytest.mark.parametrize("blend_mode", (0, 1, 2, 3, 4, 5, 6))
@pytest.mark.parametrize("case", ("success", "padded", "small", "maximum",
                                 "z_success", "z_padded", "z_small", "z_maximum"))
def test_exact_c_blend_codes_and_atomic_owned_targets(blend_executor_guard, tmp_path, socket_path,
                                                    blend_mode, case):
    with blend_peer(socket_path, blend_mode) as records:
        status, stderr = run_blend_guard(blend_executor_guard, tmp_path, socket_path, case, blend_mode)
    record, = records
    assert_committed(status, stderr, record, case, blend_mode, tmp_path)


@pytest.mark.parametrize("blend_mode", UNIT_MODES)
@pytest.mark.parametrize("case", ("success", "z_success"))
@pytest.mark.parametrize("mutation", FACTOR_REJECTIONS + FEATURE_REJECTIONS +
                         ("mixed_rgb", "mixed_alpha", "unknown_equation_7"))
def test_new_equations_reject_inexact_factors_mixed_functions_and_features(blend_executor_guard,
                                                                        tmp_path, socket_path,
                                                                        blend_mode, case, mutation):
    status, stderr = run_blend_guard(blend_executor_guard, tmp_path, socket_path, case,
                                    blend_mode, mutation)
    assert_rejected(status, stderr)
    assert not socket_path.exists()
    if mutation.startswith(("factor_", "mixed_")):
        assert "source-over" in stderr
    if mutation == "unknown_equation_7":
        assert "unsupported unit-factor blend equation" in stderr


@pytest.mark.parametrize("blend_mode", UNIT_MODES)
@pytest.mark.parametrize("case", ("success", "z_success"))
@pytest.mark.parametrize("response", ("reply_sequence", "reply_status", "reply_unknown_status",
                                      "reply_short_color", "reply_trailing"))
def test_new_equation_failed_transport_never_commits_color_or_depth(blend_executor_guard,
                                                                  tmp_path, socket_path,
                                                                  blend_mode, case, response):
    with blend_peer(socket_path, blend_mode, response) as records:
        status, stderr = run_blend_guard(blend_executor_guard, tmp_path, socket_path, case, blend_mode)
    assert len(records) == 1 and records[0]["blend"] == blend_mode
    assert_rejected(status, stderr)


@pytest.mark.parametrize("blend_mode", UNIT_MODES)
@pytest.mark.parametrize("response", ("reply_depth_size", "reply_short_depth"))
def test_new_equation_color_is_staged_until_complete_z16(blend_executor_guard, tmp_path, socket_path,
                                                       blend_mode, response):
    with blend_peer(socket_path, blend_mode, response) as records:
        status, stderr = run_blend_guard(blend_executor_guard, tmp_path, socket_path,
                                        "z_success", blend_mode)
    assert len(records) == 1 and records[0]["depth_image"]
    assert_rejected(status, stderr)


@pytest.mark.parametrize("blend_mode", UNIT_MODES)
@pytest.mark.parametrize("case", ("z_pin_success", "z_pin_error", "z_pin_depth_geometry",
                                 "z_pin_depth_identity", "z_pin_color_identity", "z_pin_state",
                                 "z_pin_blend_equation", "z_pin_blend_factor"))
def test_new_equation_dual_pins_lifetime_and_commit_revalidation(blend_executor_guard, tmp_path,
                                                               socket_path, blend_mode, case):
    response = "pin_success" if case == "z_pin_success" else "pin_error" if case == "z_pin_error" else "pin_revalidate"
    with blend_peer(socket_path, blend_mode, response) as records:
        status, stderr = run_blend_guard(blend_executor_guard, tmp_path, socket_path, case, blend_mode)
    assert status["pin_rejections"] == 11 and status["pin_recovered"] is True
    record, = records
    if case == "z_pin_success":
        assert_committed(status, stderr, record, case, blend_mode, tmp_path)
    else:
        assert_rejected(status, stderr)
