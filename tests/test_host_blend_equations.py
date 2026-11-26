import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import socket
import struct

import numpy as np
import pytest

from tools.fragment_acceptance import mesh_words
from tools.host_blend import BLEND_MODES, blend_rgba8, validate_blend_mode
from tools.host_executor import HostExecutorServer, REPLY, REQUEST
from tools.host_render import render_artifact
from tools.run_render import capture_draw
from tools.run_shader import ROOT


EXPECTED_MODES = ("disabled", "source-over", "additive", "subtract",
                  "reverse-subtract", "minimum", "maximum")
EQUATION_MODES = EXPECTED_MODES[2:]
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
FIXTURE = ROOT / "tests/fixtures/host-blend"
WIDTH, HEIGHT = 8, 4
CASES = {
    "separate-rgba": ("frag.glsl", (-0.25,), False),
    "near-first": ("frag.glsl", (-0.25, 0.25), False),
    "far-first": ("frag.glsl", (0.25, -0.25), False),
    "per-commit": ("quarter", (0.0, 0.0, 0.0), False),
    "clamped-rgb": ("clamped.frag.glsl", (0.0, 0.0), False),
    "clamped-alpha": ("clamped-alpha.frag.glsl", (-0.5, 0.5), False),
    "zero-alpha-near-first": ("zero-alpha.frag.glsl", (-0.5, 0.5), True),
    "zero-alpha-far-first": ("zero-alpha.frag.glsl", (0.5, -0.5), True),
    "kill-near-first": ("discard.frag.glsl", (-0.5, 0.5), True),
    "kill-far-first": ("discard.frag.glsl", (0.5, -0.5), True),
    "kill-no-depth": ("discard.frag.glsl", (-0.5, 0.5), False),
    "kill-zero-alpha": ("discard-zero-alpha.frag.glsl", (-0.5,), True),
    "zero-rgba": ("zero", (-0.75,), False),
    "quantized-tie": ("quarter", (2 ** -17, 0.0), True),
}
EXTRA_SHADERS = {
    "quarter": "gl_FragColor = vec4(0.25, 0.5, 0.75, 0.125);",
    "zero": "gl_FragColor = vec4(0.0);",
    "nonfinite": ("float value = v_position.z; if (v_position.x > 0.0) value = 65504.0; "
                  "gl_FragColor = vec4(value * 2.0, 0.0, 0.0, 1.0);"),
}
REQUEST_LAYOUT = struct.Struct("<8sIQIIII")
REPLY_LAYOUT = struct.Struct("<8sQIIII")


def fraction_blend(source, destination, mode):
    clamped = tuple(min(max(Fraction(channel), 0), 1) for channel in source)
    normalized = tuple(Fraction(int(channel), 255) for channel in destination)
    if mode == "disabled":
        result = clamped
    elif mode == "source-over":
        alpha = clamped[3]
        result = tuple(clamped[channel] * alpha + normalized[channel] * (1 - alpha)
                       for channel in range(3)) + (alpha + normalized[3] * (1 - alpha),)
    elif mode == "additive":
        result = tuple(first + second for first, second in zip(clamped, normalized))
    elif mode == "subtract":
        result = tuple(first - second for first, second in zip(clamped, normalized))
    elif mode == "reverse-subtract":
        result = tuple(second - first for first, second in zip(clamped, normalized))
    elif mode == "minimum":
        result = tuple(min(first, second) for first, second in zip(clamped, normalized))
    elif mode == "maximum":
        result = tuple(max(first, second) for first, second in zip(clamped, normalized))
    else:
        raise AssertionError("oracle received an unknown equation")
    return tuple(round(min(max(channel, 0), 1) * 255) for channel in result)


def fraction_source(variant, depth):
    if variant == "quarter":
        return (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1, 8))
    if variant == "zero":
        return (0, 0, 0, 0)
    if variant == "clamped.frag.glsl":
        return (Fraction(3, 2), Fraction(-1, 2), Fraction(1, 4), Fraction(1, 2))
    alpha = Fraction(1, 2)
    if variant in ("zero-alpha.frag.glsl", "discard-zero-alpha.frag.glsl"):
        alpha += depth
    elif variant == "clamped-alpha.frag.glsl":
        alpha += 2 * depth
    return (Fraction(1, 2) - depth, 0, Fraction(1, 2) + depth, alpha)


def background():
    initial = np.random.default_rng(825).integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)
    initial[0, :4] = [[0, 0, 0, 0], [255, 255, 255, 255],
                      [20, 36, 43, 65], [129, 131, 132, 134]]
    return initial


def depth_background():
    return np.tile(np.array([0, 1, 16383, 16384, 32767, 32768, 49151, 65535],
                            dtype=np.uint16), (HEIGHT, 1))


def fraction_render(case, initial, mode, initial_depth=None, *, quantized=False, depth_test=None):
    variant, depths, captured_depth_test = CASES[case]
    if depth_test is None:
        depth_test = captured_depth_test
    image = initial.copy()
    coverage = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    depth_image = np.full((HEIGHT, WIDTH), 65535, dtype=np.uint16)
    fragments = discarded = 0
    for row in range(HEIGHT):
        for column in range(WIDTH):
            stored_depth = Fraction(1) if initial_depth is None else Fraction(int(initial_depth[row, column]), 65535)
            for value in depths:
                depth = Fraction(value)
                window_depth = (depth + 1) / 2
                if quantized:
                    window_depth = Fraction(round(window_depth * 65535), 65535)
                if depth_test and window_depth >= stored_depth:
                    continue
                fragments += 1
                if variant.startswith("discard") and column < WIDTH // 2 and depth < 0:
                    discarded += 1
                    continue
                image[row, column] = fraction_blend(fraction_source(variant, depth), image[row, column], mode)
                coverage[row, column] += 1
                stored_depth = window_depth
            depth_image[row, column] = round(stored_depth * 65535)
    return image, coverage, depth_image, {"fragments": fragments, "discarded_fragments": discarded,
                                        "committed_fragments": fragments - discarded}


@pytest.fixture(scope="module")
def equation_artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("host-blend-equation-captures")
    artifacts = {}
    captures = dict(CASES, nonfinite=("nonfinite", (0.75,), False))
    for name, (variant, depths, depth_test) in captures.items():
        output = directory / name
        shader = output / "shader"
        shader.mkdir(parents=True)
        (shader / "vert.glsl").write_text((FIXTURE / "vert.glsl").read_text())
        fragment = (FIXTURE / variant).read_text() if variant not in EXTRA_SHADERS else (
            "#version 100\nprecision mediump float;\nvarying vec3 v_position;\nvoid main() {\n"
            + EXTRA_SHADERS[variant] + "\n}\n")
        (shader / "frag.glsl").write_text(fragment)
        mesh = output / "vertices.bin"
        mesh.write_bytes(mesh_words(depths))
        artifacts[name] = capture_draw(shader, BUILD, output, mesh=mesh,
                                       depth_test=depth_test, fragment_status=True)
        assert artifacts[name]["program_format"] == "legacy-mgcsim-v1"
        assert (output / "draw.json").is_file()
    return artifacts


def test_exact_wire_codes_and_unchanged_binary_headers():
    assert BLEND_MODES == EXPECTED_MODES
    assert type(BLEND_MODES) is tuple
    assert REQUEST.format == REQUEST_LAYOUT.format and REQUEST.size == 36
    assert REPLY.format == REPLY_LAYOUT.format and REPLY.size == 32


class BlendString(str):
    pass


@pytest.mark.parametrize("mode", [None, False, True, 0, 1, 2, 6, 7, "", "2", "add", "min", "max",
                                 "SOURCE-OVER", "Subtract", "source_over", "multiply", "screen",
                                 "subtract-alpha", {}, [], b"additive", BlendString("additive")])
def test_modes_are_exact_strings_without_aliases(mode):
    with pytest.raises(ValueError, match="blend"):
        validate_blend_mode(mode)
    with pytest.raises(ValueError, match="blend"):
        blend_rgba8([0, 0, 0, 0], np.zeros(4, dtype=np.uint8), mode)
    with pytest.raises(ValueError, match="blend"):
        render_artifact({}, blend=mode)


@pytest.mark.parametrize("mode", EXPECTED_MODES)
def test_all_exact_modes_pass_validation_before_capture_checks(mode):
    validate_blend_mode(mode)
    with pytest.raises(ValueError, match="capture schema"):
        render_artifact({}, blend=mode)


@pytest.mark.parametrize("mode", EQUATION_MODES)
def test_fraction_equations_cover_every_destination_byte_and_halfway_ties(mode):
    for value in range(256):
        destination = np.array([value, 255 - value, (value + 65) % 256, (value + 129) % 256], dtype=np.uint8)
        for source in ((Fraction(1, 2),) * 4,
                       (Fraction(1, 4), Fraction(3, 4), Fraction(-1, 2), Fraction(3, 2))):
            saved = destination.copy()
            actual = blend_rgba8([float(channel) for channel in source], destination, mode)
            assert tuple(actual) == fraction_blend(source, destination, mode), (mode, value, source)
            assert actual.dtype == np.uint8 and np.array_equal(destination, saved)


@pytest.mark.parametrize("mode", EQUATION_MODES)
def test_fraction_equations_random_independent_rgba_including_alpha_zero(mode):
    generator = np.random.default_rng(312)
    for index in range(256):
        source = tuple(Fraction(int(channel), 1024) for channel in generator.integers(-512, 1537, 4))
        if index % 2 == 0:
            source = source[:3] + (Fraction(0),)
        destination = generator.integers(0, 256, 4, dtype=np.uint8)
        assert tuple(blend_rgba8([float(channel) for channel in source], destination, mode)) == fraction_blend(source, destination, mode)


def test_source_over_preserves_distinct_rgb_and_alpha_factors():
    source = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1, 4))
    destination = np.array([32, 64, 96, 128], dtype=np.uint8)
    assert tuple(blend_rgba8([float(channel) for channel in source], destination, "source-over")) == fraction_blend(source, destination, "source-over")
    assert np.array_equal(blend_rgba8([1, 0, 1, 0], destination, "source-over"), destination)


@pytest.mark.parametrize("mode", EQUATION_MODES)
@pytest.mark.parametrize("case", CASES)
def test_actual_compiled_captured_artifacts_match_fraction_commit_oracle(equation_artifacts, case, mode):
    initial = background()
    saved = initial.copy()
    image, coverage, report = render_artifact(equation_artifacts[case], width=WIDTH, height=HEIGHT,
                                              blend=mode, depth_test=CASES[case][2], initial_image=initial)
    expected, expected_coverage, _, counts = fraction_render(case, initial, mode)
    assert np.array_equal(image, expected) and np.array_equal(coverage, expected_coverage)
    assert np.array_equal(initial, saved) and not np.shares_memory(image, initial)
    assert report["blend"] == mode and report["host_steps"] > report["fragments"]
    assert report["fpga_execution"] is False
    assert report["vertices"] == len(CASES[case][1]) * 6
    assert all(report[key] == value for key, value in counts.items())


def test_actual_additive_quantizes_each_commit_not_only_final_image(equation_artifacts):
    image, coverage, _ = render_artifact(equation_artifacts["per-commit"], width=WIDTH, height=HEIGHT,
                                        blend="additive")
    assert np.all(image == [192, 255, 255, 96]) and np.all(coverage == 3)
    assert int(image[0, 0, 0]) != round(Fraction(3, 4) * 255)


@pytest.mark.parametrize("mode", EQUATION_MODES)
@pytest.mark.parametrize("case", ("zero-alpha-near-first", "zero-alpha-far-first", "kill-near-first",
                                  "kill-far-first", "kill-zero-alpha", "quantized-tie"))
def test_actual_equations_preserve_z16_less_kill_and_zero_alpha_ordering(equation_artifacts, case, mode):
    initial = background()
    initial_depth = depth_background()
    saved_depth = initial_depth.copy()
    output_depth = np.full_like(initial_depth, 0x4321)
    image, coverage, report = render_artifact(equation_artifacts[case], width=WIDTH, height=HEIGHT,
                                              blend=mode, depth_test=True, initial_image=initial,
                                              initial_depth=initial_depth, depth_output=output_depth)
    expected, expected_coverage, expected_depth, counts = fraction_render(case, initial, mode, initial_depth,
                                                                         quantized=True)
    assert np.array_equal(image, expected) and np.array_equal(coverage, expected_coverage)
    assert np.array_equal(output_depth, expected_depth) and np.array_equal(initial_depth, saved_depth)
    assert report["depth_storage"] == "Z16_UNORM" and report["depth"] == "less"
    assert all(report[key] == value for key, value in counts.items())


@pytest.mark.parametrize("mode", EQUATION_MODES)
@pytest.mark.parametrize("failure", ("steps", "late-steps", "fragments", "deadline", "provenance", "status", "nonfinite"))
def test_actual_render_errors_do_not_publish_partial_color_or_z16(equation_artifacts, mode, failure):
    case = "kill-no-depth" if failure == "status" else "nonfinite" if failure == "nonfinite" else "per-commit"
    artifact = copy.deepcopy(equation_artifacts[case])
    initial = background()
    initial_depth = np.full((HEIGHT, WIDTH), 65535, dtype=np.uint16)
    output_depth = np.full_like(initial_depth, 0x4321)
    saved_image, saved_depth, saved_output = initial.copy(), initial_depth.copy(), output_depth.copy()
    options = {"depth_test": True, "initial_image": initial, "initial_depth": initial_depth,
               "depth_output": output_depth, "width": WIDTH, "height": HEIGHT, "blend": mode}
    if failure == "steps":
        options["max_steps"] = 1
    elif failure == "late-steps":
        _, _, report = render_artifact(artifact, **options)
        output_depth[:] = saved_output
        options["max_steps"] = report["host_steps"] - 1
    elif failure == "fragments":
        options["max_fragments"] = 1
    elif failure == "deadline":
        options["execution_timeout"] = 1e-9
    elif failure == "provenance":
        artifact["shaders"]["fs"]["assembly"] = artifact["shaders"]["fs"]["assembly"].replace("ABI=1,", "ABI=2,", 1)
    elif failure == "status":
        del artifact["shaders"]["fs"]["layout"]["fragment_status"]
    with pytest.raises(ValueError, match={"steps": "budget", "late-steps": "budget", "fragments": "budget",
                                         "deadline": "deadline", "provenance": "provenance", "status": "discard",
                                         "nonfinite": "nonfinite framebuffer output"}[failure]):
        render_artifact(artifact, **options)
    assert np.array_equal(initial, saved_image) and np.array_equal(initial_depth, saved_depth)
    assert np.array_equal(output_depth, saved_output)


def wire_request(artifact, initial, version, code, initial_depth=None, *, sequence=1, encoded=None):
    submitted = copy.deepcopy(artifact)
    submitted["sequence"] = sequence
    payload = json.dumps(submitted, separators=(",", ":")).encode() if encoded is None else encoded
    magic = b"MGCHE01\0" if version == 1 else b"MGCHE02\0"
    header = REQUEST_LAYOUT.pack(magic, len(payload), sequence, WIDTH, HEIGHT, code, initial.nbytes)
    depth_bytes = b"" if version == 1 else initial_depth.astype("<u2").tobytes()
    if version == 2:
        header += struct.pack("<I", len(depth_bytes))
    return header + payload + initial.tobytes() + depth_bytes


def wire_exchange(server, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(3)
        connection.connect(str(server.path))
        connection.sendall(payload)
        connection.shutdown(socket.SHUT_WR)
        reply = bytearray()
        while True:
            try:
                chunk = connection.recv(65536)
            except ConnectionResetError:
                break
            if not chunk:
                return bytes(reply)
            reply.extend(chunk)
            assert len(reply) <= REPLY_LAYOUT.size + 4 + WIDTH * HEIGHT * 6
        return bytes(reply)


@pytest.mark.parametrize("version", (1, 2))
@pytest.mark.parametrize("code,mode", tuple(enumerate(EXPECTED_MODES)))
def test_actual_unix_requests_map_every_exact_code_without_header_changes(tmp_path, equation_artifacts, version, code, mode):
    case = "clamped-rgb" if code >= 2 else "zero-alpha-near-first"
    initial = background() if code >= 2 else np.full((HEIGHT, WIDTH, 4), 127, dtype=np.uint8)
    initial_depth = depth_background()
    expected, expected_coverage, expected_depth, counts = fraction_render(
        case, initial, mode, initial_depth if version == 2 else None,
        quantized=version == 2, depth_test=version == 2)
    payload = wire_request(equation_artifacts[case], initial, version, code, initial_depth,
                           sequence=0x0102030405060708)
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        reply = wire_exchange(server, payload)
        magic = b"MGCRG01\0" if version == 1 else b"MGCRG02\0"
        assert REPLY_LAYOUT.unpack_from(reply) == (magic, 0x0102030405060708, 0, WIDTH, HEIGHT, initial.nbytes)
        offset = REPLY_LAYOUT.size
        if version == 2:
            assert struct.unpack_from("<I", reply, offset) == (initial_depth.nbytes,)
            offset += 4
        actual = np.frombuffer(reply[offset:offset + initial.nbytes], dtype=np.uint8).reshape(initial.shape)
        assert np.array_equal(actual, expected)
        offset += initial.nbytes
        if version == 2:
            actual_depth = np.frombuffer(reply[offset:], dtype="<u2").reshape(initial_depth.shape)
            assert np.array_equal(actual_depth, expected_depth)
            offset += initial_depth.nbytes
        assert len(reply) == offset and not server.errors and len(server.records) == 1
        record = server.records[0]
        assert record["peer_pid"] == os.getpid() and record["report"]["blend"] == mode
        assert np.array_equal(record["coverage"], expected_coverage)
        assert all(record["report"][key] == value for key, value in counts.items())


@pytest.mark.parametrize("version", (1, 2))
@pytest.mark.parametrize("code", (7, 8, 255, 65535, 0xffffffff))
def test_unknown_unix_blend_codes_fail_closed_before_either_output(tmp_path, equation_artifacts, version, code):
    payload = wire_request(equation_artifacts["per-commit"], background(), version, code, depth_background())
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1) as server:
        assert wire_exchange(server, payload) == b""
        assert not server.records and len(server.errors) == 1
        assert "unsupported host state" in server.errors[0]


@pytest.mark.parametrize("version", (1, 2))
@pytest.mark.parametrize("code", (2, 3, 4, 5, 6))
@pytest.mark.parametrize("failure", ("steps", "fragments", "json", "status", "sequence", "nonfinite"))
def test_actual_equation_unix_errors_return_no_partial_planes(tmp_path, equation_artifacts, version, code, failure):
    case = "kill-no-depth" if failure == "status" else "nonfinite" if failure == "nonfinite" else "per-commit"
    artifact = copy.deepcopy(equation_artifacts[case])
    encoded = None
    if failure == "json":
        encoded = b'{"sequence":1,"sequence":1}'
    elif failure == "status":
        del artifact["shaders"]["fs"]["layout"]["fragment_status"]
    elif failure == "sequence":
        artifact["sequence"] = 2
        encoded = json.dumps(artifact).encode()
    initial, initial_depth = background(), np.full((HEIGHT, WIDTH), 65535, dtype=np.uint16)
    saved_image, saved_depth = initial.copy(), initial_depth.copy()
    options = {"max_steps": 1} if failure == "steps" else {"max_fragments": 1} if failure == "fragments" else {}
    with HostExecutorServer(tmp_path / "executor.sock", max_requests=1, **options) as server:
        reply = wire_exchange(server, wire_request(artifact, initial, version, code, initial_depth, encoded=encoded))
        expected = REPLY_LAYOUT.pack(b"MGCRG01\0" if version == 1 else b"MGCRG02\0", 1, 1, WIDTH, HEIGHT, 0)
        if version == 2:
            expected += struct.pack("<I", 0)
        assert reply == expected and not server.records and len(server.errors) == 1
    assert np.array_equal(initial, saved_image) and np.array_equal(initial_depth, saved_depth)
