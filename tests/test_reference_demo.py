"""CPU packet-reference tests, not evidence of Mesa/FPGA end-to-end execution."""

from dataclasses import replace
from pathlib import Path
import random
import struct
import sys
import zlib

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_reference import benchmark, main as benchmark_main
from mgcreference import FRAMEBUFFER_BYTES, Frame, Triangle, load_scene, render, subset_frame
from mgcreference.artifacts import compare_framebuffer, export_frame, write_png, write_ppm
from mgcreference.raster import _light_rows, ceramic_rgb565, pack_ceramic_rgb565, rgb565_to_rgb8
from mgcreference.scene import DEFAULT_DATASET, SCENES, read_words, signed32
from reference_demo import main as demo_main, selected_frames


@pytest.fixture(scope="module")
def scenes(local_hardware_examples):
    return {scene: load_scene(scene) for scene in SCENES}


def synthetic_frame(*triangles, shading="flat"):
    return Frame("synthetic", 0, 0x100000, triangles[0].bank, shading,
                 tuple(triangles), Path("synthetic.data.mem"), "synthetic", len(triangles))


def triangle(**changes):
    packet = Triangle(0xF800, 7 * 1024 + 5 * 4, 4, 4,
                      (0, 0, 3), (1, 0, -1), (0, 1, -1))
    return replace(packet, **changes)


def memory_file(tmp_path, words):
    path = tmp_path / "mesa_gears_hd.data.mem"
    path.write_text("\n".join(f"{word & 0xFFFFFFFF:08x}" for word in words) + "\n")
    return path


@pytest.mark.parametrize("scene,counts", [
    ("utah", [2345, 2358, 2331, 2288, 2331, 2358, 2345, 2309, 2287, 2262, 2287, 2309]),
    ("gears", [4858, 4850]),
    ("single_frame", [4858, 4850]),
    ("bucky12", [526, 533, 531, 530, 556, 552, 548, 544, 556, 534, 534, 537]),
    ("bucky6", [526, 533, 531, 530, 556, 552]),
])
def test_scene_counts_boundaries_and_assembly(scenes, scene, counts):
    frames = scenes[scene]
    assert [frame.available_triangles for frame in frames] == counts
    pointer = 0x100000
    stride = 15 if scene == "utah" else 13
    for frame, count in zip(frames, counts):
        assert frame.header_address == pointer
        assert frame.assembly_verified == (scene in ("utah", "gears", "bucky12"))
        pointer += (1 + count * stride) * 4
        assert all(packet.bank == frame.bank for packet in frame.triangles)


def test_gears_alias_is_not_an_independent_scene(scenes):
    for suffix in ("data.mem", "program.mem"):
        assert (DEFAULT_DATASET / f"single_frame.{suffix}").read_bytes() == (
            DEFAULT_DATASET / f"mesa_gears_hd.{suffix}").read_bytes()
    assert scenes["single_frame"][0].triangles == scenes["gears"][0].triangles


def test_first_utah_record_and_distinct_memory_bases(scenes):
    packet = scenes["utah"][0].triangles[0]
    assert (packet.color_or_light, packet.address, packet.width, packet.height) == (0x3B6B, 0x611F0, 11, 5)
    assert (packet.bank, packet.x, packet.y) == (1, 124, 132)
    assert packet.edges == (-62, -263, 8312)
    assert packet.edge_dx == (873, -102, -771)
    assert packet.edge_dy == (3, 2342, -2344)
    assert (packet.light_dx_bits, packet.light_dy_bits) == (0x19E2, 0xAC3A)
    assert scenes["utah"][0].header_address == 0x100000
    assert packet.address < 0x80000


@pytest.mark.parametrize("word,expected", [(0, 0), (0xFFFFFFFF, -1), (0x80000000, -2147483648),
                                           (0x7FFFFFFF, 2147483647), (0x100000000, 0)])
def test_signed32_wrap(word, expected):
    assert signed32(word) == expected


@pytest.mark.parametrize("renderer", ["numpy", "scalar"])
def test_inclusive_edges_known_mask_and_stats(renderer):
    result = render(synthetic_frame(triangle()), renderer)
    expected = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool)
    assert np.array_equal(result.touched[7:11, 5:9], expected)
    assert (result.candidate_pixels, result.covered_fragments, int(result.touched.sum())) == (16, 10, 10)
    assert np.array_equal(result.pixels[7:11, 5:9], expected * 0xF800)


@pytest.mark.parametrize("renderer", ["numpy", "scalar"])
def test_shared_edge_is_inclusive_and_later_packet_wins(renderer):
    first = triangle(width=2, height=2, edges=(0, 0, 1))
    second = replace(first, color_or_light=0x07E0, edges=(0, 0, -1), edge_dx=(1, 0, 1), edge_dy=(0, 1, 1))
    result = render(synthetic_frame(first, second), renderer)
    assert np.array_equal(result.pixels[7:9, 5:7], [[0xF800, 0x07E0], [0x07E0, 0x07E0]])
    assert result.covered_fragments == 6
    assert result.touched.sum() == 4


@pytest.mark.parametrize("renderer", ["numpy", "scalar"])
def test_int32_overflow_changes_coverage(renderer):
    packet = triangle(width=3, height=1, edges=(0x7FFFFFFF, 0, 0), edge_dx=(1, 0, 0), edge_dy=(0, 0, 0))
    result = render(synthetic_frame(packet), renderer)
    assert result.pixels[7, 5:8].tolist() == [0xF800, 0, 0]


def test_randomized_wrapped_masks_agree():
    generator = random.Random(20261108)
    for _ in range(30):
        packet = triangle(
            width=generator.randint(1, 9), height=generator.randint(1, 9),
            edges=tuple(generator.randint(-2**31, 2**31 - 1) for _ in range(3)),
            edge_dx=tuple(generator.randint(-2**31, 2**31 - 1) for _ in range(3)),
            edge_dy=tuple(generator.randint(-2**31, 2**31 - 1) for _ in range(3)),
        )
        frame = synthetic_frame(packet)
        vector = render(frame)
        scalar = render(frame, "scalar")
        assert np.array_equal(vector.pixels, scalar.pixels)
        assert np.array_equal(vector.touched, scalar.touched)


@pytest.mark.parametrize("scene", ["utah", "gears", "bucky12"])
def test_real_subset_scalar_coverage_oracle(scenes, scene):
    frame = subset_frame(scenes[scene][0], count=32)
    vector = render(frame, shading="coverage")
    scalar = render(frame, "scalar", shading="coverage")
    assert vector.sha256 == scalar.sha256
    assert vector.covered_fragments == scalar.covered_fragments


@pytest.mark.parametrize("scene,fragments,visible,digest", [
    ("utah", 11591, 6309, "2547df5d74dfb18e8621525b622a3f28755930e8ca63d9a3b500c4a162037030"),
    ("gears", 23287, 20744, "1de34143fbdcb09161d1503a6aec07962e7d383a13e7f14d7eb97638782cc3f3"),
    ("bucky12", 26326, 18440, "cb59328efdac0dae6a3c8610bde382b3e86a4f9a722c4849d001965b869130e0"),
])
def test_full_frame_reference_regression(scenes, scene, fragments, visible, digest):
    result = render(scenes[scene][0])
    assert result.covered_fragments == fragments
    assert result.touched.sum() == visible
    assert result.sha256 == digest
    assert result.summary()["actual_gpu_execution"] is False


def test_black_fragment_still_counts_as_coverage():
    result = render(synthetic_frame(triangle(color_or_light=0)))
    assert result.covered_fragments == 10
    assert result.touched.sum() == 10
    assert np.count_nonzero(result.pixels) == 0


def test_both_framebuffer_banks_have_same_preview_coordinates():
    first = triangle()
    hidden = replace(first, address=first.address + FRAMEBUFFER_BYTES)
    assert render(synthetic_frame(first)).sha256 == render(synthetic_frame(hidden)).sha256


def test_light_recurrence_and_chosen_rgb565_quantization():
    packet = triangle(color_or_light=0x3800, width=5, height=2, light_dx_bits=0x3000, light_dy_bits=0x3000)
    light = _light_rows(packet)
    assert np.array_equal(light, np.array([[.5, .625, .75, .875, 1], [.625, .75, .875, 1, 1.125]], dtype=np.float16))
    assert ceramic_rgb565(np.array([-1, 0, 2], dtype=np.float16)).tolist() == [0x3186, 0x3186, 0xFFFF]


def test_rgb565_channel_expansion():
    assert rgb565_to_rgb8(np.array([0, 0xF800, 0x07E0, 0x001F, 0xFFFF], dtype=np.uint16)).tolist() == [
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255],
    ]


def test_final_intensity_infinities_saturate_but_nan_is_rejected():
    values = np.array([-np.inf, np.inf], dtype=np.float16)
    assert pack_ceramic_rgb565(values).tolist() == [0, 0xFFFF]
    with pytest.raises(ValueError, match="NaN"):
        pack_ceramic_rgb565(np.array([np.nan], dtype=np.float16))


@pytest.mark.parametrize("token", ["", "123", "100000000", "zzzzzzzz", "0x000001"])
def test_invalid_memory_tokens(tmp_path, token):
    path = tmp_path / "bad.mem"
    path.write_text(token)
    with pytest.raises(ValueError):
        read_words(path)


@pytest.mark.parametrize("words", [[0], [2, 0], [1] + [0] * 13, [1, 0xFFFF, 0x80000, 1, 1] + [0] * 9,
                                   [1, 0xFFFF, 1023, 1, 1] + [0] * 9,
                                   [1, 0xFFFF, 1020, 2, 1] + [0] * 9])
def test_invalid_frame_packets_are_rejected(tmp_path, words):
    memory_file(tmp_path, words)
    with pytest.raises(ValueError):
        load_scene("gears", tmp_path)


def test_assembly_pointer_mismatch_fails_closed(tmp_path):
    memory_file(tmp_path, [1, 0xFFFF, 0, 1, 1] + [0] * 9)
    (tmp_path / "mesa_gears_hd.asm").write_text("frame_00:\nmovi_i32 1048584, r26\n")
    with pytest.raises(ValueError, match="pointer disagrees"):
        load_scene("gears", tmp_path)


@pytest.mark.parametrize("first,count", [(-1, None), (1, None), (0, 0), (0, 2)])
def test_invalid_subsets(first, count):
    with pytest.raises(ValueError):
        subset_frame(synthetic_frame(triangle()), first, count)


@pytest.mark.parametrize("specification", ["-1", "2", "0,0", "", "no"])
def test_invalid_frame_selection(specification):
    with pytest.raises(ValueError):
        selected_frames(specification, 2)


def test_png_ppm_and_raw_exports_are_deterministic_and_valid(tmp_path):
    frame = synthetic_frame(triangle())
    result = render(frame)
    export_frame(tmp_path, frame, result, scale=1)
    png = tmp_path / "synthetic-frame-00.png"
    first = png.read_bytes()
    export_frame(tmp_path, frame, result, scale=1)
    assert first == png.read_bytes()
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    compressed = b""
    while offset < len(first):
        length = struct.unpack(">I", first[offset:offset + 4])[0]
        kind = first[offset + 4:offset + 8]
        content = first[offset + 8:offset + 8 + length]
        checksum = struct.unpack(">I", first[offset + 8 + length:offset + 12 + length])[0]
        assert checksum == zlib.crc32(kind + content) & 0xFFFFFFFF
        if kind == b"IHDR":
            assert struct.unpack(">IIBBBBB", content) == (256, 256, 8, 2, 0, 0, 0)
        if kind == b"IDAT":
            compressed += content
        offset += length + 12
    expected_rgb = rgb565_to_rgb8(result.pixels)
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(256, 769)
    assert np.all(rows[:, 0] == 0)
    assert np.array_equal(rows[:, 1:].reshape(256, 256, 3), expected_rgb)
    assert (tmp_path / "synthetic-frame-00.ppm").read_bytes() == b"P6\n256 256\n255\n" + expected_rgb.tobytes()
    raw = tmp_path / "synthetic-frame-00.framebuffer.u32le.bin"
    assert len(raw.read_bytes()) == FRAMEBUFFER_BYTES
    assert compare_framebuffer(result, raw)["equal"]
    changed = np.frombuffer(raw.read_bytes(), dtype="<u4").copy()
    changed[0] = 123
    raw.write_bytes(changed.tobytes())
    assert compare_framebuffer(result, raw)["mismatched_pixels"] == 1
    raw.write_bytes(b"short")
    with pytest.raises(ValueError, match="exactly"):
        compare_framebuffer(result, raw)


@pytest.mark.parametrize("writer", [write_png, write_ppm])
def test_bad_image_shapes(writer, tmp_path):
    with pytest.raises(ValueError):
        writer(tmp_path / "bad", np.zeros((2, 2), dtype=np.uint8))


def test_benchmark_samples_exclude_io_and_keep_cpu_label():
    record = benchmark(synthetic_frame(triangle()), warmup=1, repeats=3)
    assert len(record["elapsed_ns"]) == 3
    assert all(sample > 0 for sample in record["elapsed_ns"])
    assert record["min_ms"] <= record["median_ms"] <= record["max_ms"]
    assert record["engine"] == "cpu-reference"
    assert record["actual_gpu_execution"] is False


def test_cli_small_demo_and_benchmark(tmp_path):
    memory_file(tmp_path, [1, 0xF800, 0, 1, 1] + [0] * 9)
    output = tmp_path / "output"
    assert demo_main(["--dataset", str(tmp_path), "--scene", "gears", "--output", str(output)]) == 0
    dump = output / "gears-frame-00.framebuffer.u32le.bin"
    comparison_output = tmp_path / "comparison"
    assert demo_main(["--dataset", str(tmp_path), "--scene", "gears", "--output", str(comparison_output),
                      "--compare-dump", str(dump)]) == 0
    dump.write_bytes(b"\x01\x00\x00\x00" + dump.read_bytes()[4:])
    assert demo_main(["--dataset", str(tmp_path), "--scene", "gears", "--output", str(comparison_output),
                      "--compare-dump", str(dump)]) == 1
    assert demo_main(["--dataset", str(tmp_path), "--scene", "gears", "--output", str(output),
                      "--compare-dump", str(dump)]) == 2
    assert benchmark_main(["--dataset", str(tmp_path), "--scenes", "gears", "--warmup", "0",
                           "--repeats", "1", "--output", str(output / "benchmark.json")]) == 0
    assert demo_main(["--dataset", str(tmp_path), "--scale", "0"]) == 2
    assert benchmark_main(["--repeats", "0"]) == 2


def test_invalid_shading_modes(scenes):
    with pytest.raises(ValueError, match="not RGB565"):
        render(scenes["utah"][0], shading="flat")
    with pytest.raises(ValueError, match="requires Utah"):
        render(scenes["gears"][0], shading="ceramic")


@pytest.mark.parametrize("scene,count", [("gears", 32), ("bucky12", 526)])
def test_real_binary_packet_prefix_matches_cpu_reference(scenes, tmp_path, scene, count):
    pytest.importorskip("tools.mgchw")
    from mgcreference.hardware import run_hardware_prefix

    frame = subset_frame(scenes[scene][0], count=count)
    report = run_hardware_prefix(frame, tmp_path / f"{scene}.u32le.bin")
    comparison = compare_framebuffer(render(frame), Path(report["dump"]))
    assert comparison["equal"]
    assert report["program_or_register_patches"] is False
    assert report["triangles_completed"] == count
    assert report["assembly_binary_equal"] == (scene == "bucky12")
