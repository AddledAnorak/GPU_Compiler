"""Decode data packets, not instruction binaries or legacy mgcsim memory."""

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import re

import numpy as np


WIDTH = HEIGHT = 256
PIXEL_BYTES = 4
ROW_BYTES = WIDTH * PIXEL_BYTES
FRAMEBUFFER_BYTES = ROW_BYTES * HEIGHT
DATA_BASE = 0x100000
DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "hardware-examples/aakash-2026-09-28"
SCENES = {
    "utah": ("utah_12frame_dualbuf_shader", 15, "ceramic"),
    "gears": ("mesa_gears_hd", 13, "flat"),
    "single_frame": ("single_frame", 13, "flat"),
    "bucky12": ("bucky_12frame_dualbuf_clean", 13, "flat"),
    "bucky6": ("bucky_6frame_dualbuf_clean", 13, "flat"),
}
PACKET_FIELDS = (
    "color_or_light_fp16", "framebuffer_byte_address", "width", "height",
    "edge0", "edge1", "edge2", "edge0_dx", "edge1_dx", "edge2_dx",
    "edge0_dy", "edge1_dy", "edge2_dy", "light_dx_fp16", "light_dy_fp16",
)


@dataclass(frozen=True)
class Triangle:
    color_or_light: int
    address: int
    width: int
    height: int
    edges: tuple[int, int, int]
    edge_dx: tuple[int, int, int]
    edge_dy: tuple[int, int, int]
    light_dx_bits: int | None = None
    light_dy_bits: int | None = None

    @property
    def bank(self) -> int:
        return self.address // FRAMEBUFFER_BYTES

    @property
    def x(self) -> int:
        return (self.address % ROW_BYTES) // PIXEL_BYTES

    @property
    def y(self) -> int:
        return (self.address % FRAMEBUFFER_BYTES) // ROW_BYTES


@dataclass(frozen=True)
class Frame:
    scene: str
    index: int
    header_address: int
    bank: int
    shading: str
    triangles: tuple[Triangle, ...]
    source_path: Path
    source_sha256: str
    available_triangles: int
    first_triangle: int = 0
    assembly_verified: bool = False


def read_words(path: Path) -> np.ndarray:
    tokens = path.read_text(encoding="ascii").split()
    if not tokens:
        raise ValueError(f"{path}: empty memory image")
    for index, token in enumerate(tokens):
        if not re.fullmatch(r"[0-9a-fA-F]{8}", token):
            raise ValueError(f"{path}: word {index}: expected exactly eight hex digits")
    return np.fromiter((int(token, 16) for token in tokens), dtype=np.uint32)


def signed32(word: int) -> int:
    return ((int(word) + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def _decode_triangle(record: np.ndarray, path: Path, index: int) -> Triangle:
    values = tuple(int(word) for word in record)
    triangle = Triangle(
        values[0], values[1], values[2], values[3],
        tuple(signed32(word) for word in values[4:7]),
        tuple(signed32(word) for word in values[7:10]),
        tuple(signed32(word) for word in values[10:13]),
        values[13] if len(values) == 15 else None,
        values[14] if len(values) == 15 else None,
    )
    if triangle.address % PIXEL_BYTES or triangle.bank not in (0, 1):
        raise ValueError(f"{path}: triangle {index}: unaligned or unsupported framebuffer address")
    if not (0 < triangle.width <= WIDTH and 0 < triangle.height <= HEIGHT):
        raise ValueError(f"{path}: triangle {index}: invalid bounding-box dimensions")
    if triangle.x + triangle.width > WIDTH or triangle.y + triangle.height > HEIGHT:
        raise ValueError(f"{path}: triangle {index}: bounding box crosses the framebuffer")
    if any(word > 0xFFFF for word in (values[0], *values[13:])):
        raise ValueError(f"{path}: triangle {index}: color/light field has nonzero high bits")
    if len(values) == 15:
        light = np.array([values[0], values[13], values[14]], dtype=np.uint16).view(np.float16)
        if not np.isfinite(light).all():
            raise ValueError(f"{path}: triangle {index}: non-finite FP16 light input")
    return triangle


def _verify_assembly(frames: list[Frame], path: Path) -> list[Frame]:
    if not path.exists():
        return frames
    text = path.read_text(encoding="ascii")
    starts = list(re.finditer(r"^frame_(\d+):", text, flags=re.MULTILINE))
    if len(starts) != len(frames):
        raise ValueError(f"{path}: assembly/data frame-count mismatch")
    expected_loads = [4, 11, 14, 6, 8, 18, 22, 10, 20, 24, 13, 21, 30]
    for index, (frame, start) in enumerate(zip(frames, starts)):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start():end]
        packet = re.search(r"movi_i32\s+(\d+),\s*r26", block)
        if not packet or int(packet[1]) != frame.header_address + 4:
            raise ValueError(f"{path}: frame {index}: packet pointer disagrees with data")
        loads = [int(register) for register in re.findall(r"^load\s+r26,\s*r(\d+)", block, re.MULTILINE)]
        required = expected_loads + ([0, 1] if frame.shading == "ceramic" else [])
        if loads != required:
            raise ValueError(f"{path}: frame {index}: unexpected packet load order")
        count = re.search(r"movi_u16\s+(\d+),\s*r27", block)
        if frame.shading == "flat" and (not count or int(count[1]) != frame.available_triangles):
            raise ValueError(f"{path}: frame {index}: assembly/data triangle-count mismatch")
        if frame.shading == "ceramic" and not re.search(r"load\s+r30,\s*r27", block):
            raise ValueError(f"{path}: frame {index}: missing count-header load")
        frames[index] = replace(frame, assembly_verified=True)
    return frames


def load_scene(scene: str, dataset: Path = DEFAULT_DATASET) -> tuple[Frame, ...]:
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}; choose from {', '.join(SCENES)}")
    stem, stride, shading = SCENES[scene]
    path = Path(dataset) / f"{stem}.data.mem"
    words = read_words(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frames = []
    offset = 0
    while offset < len(words):
        count = int(words[offset])
        end = offset + 1 + count * stride
        if count == 0 or end > len(words):
            raise ValueError(f"{path}: word {offset}: truncated or invalid frame header ({count})")
        records = words[offset + 1:end].reshape(count, stride)
        triangles = tuple(_decode_triangle(record, path, index) for index, record in enumerate(records))
        banks = {triangle.bank for triangle in triangles}
        if len(banks) != 1:
            raise ValueError(f"{path}: frame {len(frames)}: mixed framebuffer banks")
        frames.append(Frame(
            scene, len(frames), DATA_BASE + offset * 4, banks.pop(), shading,
            triangles, path, digest, count,
        ))
        offset = end
    return tuple(_verify_assembly(frames, Path(dataset) / f"{stem}.asm"))


def subset_frame(frame: Frame, first: int = 0, count: int | None = None) -> Frame:
    if first < 0 or first >= len(frame.triangles):
        raise ValueError("first triangle must be within the frame")
    if count is not None and count <= 0:
        raise ValueError("triangle count must be positive")
    end = len(frame.triangles) if count is None else first + count
    if end > len(frame.triangles):
        raise ValueError("requested triangle subset extends beyond the frame")
    return replace(frame, triangles=frame.triangles[first:end], first_triangle=frame.first_triangle + first)


def frame_info(frame: Frame) -> dict:
    return {
        "scene": frame.scene, "frame": frame.index, "bank": frame.bank,
        "header_byte_address": frame.header_address,
        "first_packet_byte_address": frame.header_address + 4,
        "packet_words": 15 if frame.shading == "ceramic" else 13,
        "packet_fields": list(PACKET_FIELDS[:15 if frame.shading == "ceramic" else 13]),
        "source": str(frame.source_path), "source_sha256": frame.source_sha256,
        "assembly_verified": frame.assembly_verified,
        "assembly_verification_scope": "packet pointer, count and load order only; not binary equivalence",
        "known_assembly_binary_mismatch": frame.scene in ("gears", "single_frame"),
        "available_triangles": frame.available_triangles,
        "selected_triangles": len(frame.triangles), "first_triangle": frame.first_triangle,
        "candidate_pixels": sum(triangle.width * triangle.height for triangle in frame.triangles),
    }


def inventory(dataset: Path = DEFAULT_DATASET) -> dict:
    files = []
    for path in sorted(Path(dataset).iterdir()):
        if path.name.endswith((".asm", ".data.mem", ".program.mem")):
            content = path.read_bytes()
            files.append({
                "name": path.name, "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "lines": len(content.splitlines()),
            })
    return {
        "engine": "cpu-reference", "instruction_binaries_executed": False,
        "files": files,
        "scenes": {scene: [frame_info(frame) for frame in load_scene(scene, dataset)] for scene in SCENES},
        "notes": [
            "single_frame data/program are byte-identical to gears; both contain two frames",
            "gears assembly header says six frames; actual assembly/data contain two",
            "separate hardware reconstruction finds gears ASM/program mismatch (597 vs 609 words); binary is execution truth",
            "bucky6 and single_frame have no recovered assembly; their packet schema is inferred",
        ],
    }
