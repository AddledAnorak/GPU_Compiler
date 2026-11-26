"""Deterministic PNG/PPM and raw framebuffer exports; no image dependency."""

from pathlib import Path
import json
import struct
import zlib

import numpy as np

from .raster import RenderResult, rgb565_to_rgb8
from .scene import FRAMEBUFFER_BYTES, Frame, frame_info


def write_json(path: Path, value: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG requires an H x W x 3 uint8 RGB image")
    height, width = rgb.shape[:2]

    def chunk(kind: bytes, content: bytes) -> bytes:
        checksum = zlib.crc32(kind + content) & 0xFFFFFFFF
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", checksum)

    rows = b"".join(b"\0" + row.tobytes() for row in rgb)
    content = b"\x89PNG\r\n\x1a\n"
    content += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    content += chunk(b"IDAT", zlib.compress(rows, level=9))
    content += chunk(b"IEND", b"")
    Path(path).write_bytes(content)


def write_ppm(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PPM requires an H x W x 3 uint8 RGB image")
    height, width = rgb.shape[:2]
    Path(path).write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes())


def export_frame(output: Path, frame: Frame, result: RenderResult, scale: int = 1) -> dict:
    if not 1 <= scale <= 8:
        raise ValueError("preview scale must be between 1 and 8")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{frame.scene}-frame-{frame.index:02d}"
    rgb = rgb565_to_rgb8(result.pixels).repeat(scale, axis=0).repeat(scale, axis=1)
    write_png(output / f"{stem}.png", rgb)
    write_ppm(output / f"{stem}.ppm", rgb)
    (output / f"{stem}.framebuffer.u32le.bin").write_bytes(result.framebuffer_bytes)
    metadata = {
        **frame_info(frame), **result.summary(), "preview_scale": scale,
        "preview_origin": "row zero at top; no vertical flip",
        "color_expansion": "RGB565 to RGB8 by rounded full-range scaling",
        "framebuffer_encoding": "one bank, row-major 256x256, uint32 little-endian, high 16 bits zero",
        "assumptions": [
            "precomputed packets; no GLSL, NIR, Mesa, driver, FPGA or mgcsim execution",
            "inclusive signed-int32 edge tests; packet order overwrites; no depth test",
            "Utah vfset4 uses sequential FP16 adds; conversion clamps then truncates FP16 products; unverified against RTL",
        ],
        "outputs": {kind: str(output / f"{stem}.{suffix}") for kind, suffix in (
            ("png", "png"), ("ppm", "ppm"), ("framebuffer", "framebuffer.u32le.bin"),
        )},
    }
    write_json(output / f"{stem}.json", metadata)
    return metadata


def compare_framebuffer(result: RenderResult, path: Path) -> dict:
    content = Path(path).read_bytes()
    if len(content) != FRAMEBUFFER_BYTES:
        raise ValueError(f"expected exactly {FRAMEBUFFER_BYTES} bytes for one uint32 framebuffer bank")
    actual = np.frombuffer(content, dtype="<u4").reshape(result.pixels.shape)
    expected = result.pixels.astype(np.uint32)
    differences = actual != expected
    coordinates = np.argwhere(differences)
    return {
        "comparison": "CPU reference versus supplied uint32 framebuffer dump",
        "dump_producer": "external; provenance must be recorded separately",
        "path": str(path), "equal": not bool(differences.any()),
        "mismatched_pixels": int(differences.sum()),
        "matched_pixels": int(differences.size - differences.sum()),
        "nonzero_support_mismatched_pixels": int(np.count_nonzero((actual != 0) != (expected != 0))),
        "mismatches": [
            {"x": int(column), "y": int(row), "reference": int(expected[row, column]),
             "actual": int(actual[row, column])}
            for row, column in coordinates[:16]
        ],
    }
