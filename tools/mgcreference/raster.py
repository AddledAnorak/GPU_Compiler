"""Inclusive edge coverage and explicitly chosen CPU ceramic-shader semantics."""

from dataclasses import dataclass
import hashlib

import numpy as np

from .scene import Frame, HEIGHT, WIDTH, Triangle, signed32


@dataclass
class RenderResult:
    pixels: np.ndarray
    touched: np.ndarray
    triangle_count: int
    candidate_pixels: int
    covered_fragments: int
    shading: str
    renderer: str

    @property
    def framebuffer_bytes(self) -> bytes:
        return self.pixels.astype("<u4").tobytes()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.framebuffer_bytes).hexdigest()

    def summary(self) -> dict:
        return {
            "engine": "cpu-reference", "actual_gpu_execution": False,
            "mgcsim_execution": False, "width": WIDTH, "height": HEIGHT,
            "renderer": self.renderer, "shading": self.shading,
            "triangles": self.triangle_count, "candidate_pixels": self.candidate_pixels,
            "covered_fragments": self.covered_fragments,
            "visible_covered_pixels": int(self.touched.sum()),
            "nonzero_pixels": int(np.count_nonzero(self.pixels)),
            "framebuffer_u32le_sha256": self.sha256,
            "coverage_sha256": hashlib.sha256(self.touched.tobytes()).hexdigest(),
            "shader_hardware_equivalence": "unverified" if self.shading == "ceramic" else "not applicable",
        }


def _half(bits: int) -> np.float16:
    return np.array(bits, dtype=np.uint16).view(np.float16)[()]


def _light_rows(triangle: Triangle) -> np.ndarray:
    if triangle.light_dx_bits is None or triangle.light_dy_bits is None:
        raise ValueError("ceramic shading requires a 15-word Utah packet")
    light = np.empty((triangle.height, triangle.width), dtype=np.float16)
    row_start = _half(triangle.color_or_light)
    step_x = _half(triangle.light_dx_bits)
    step_y = _half(triangle.light_dy_bits)
    chunk_step = np.float16(np.float16(step_x + step_x) * np.float16(2))
    for row in range(triangle.height):
        current = row_start
        chunks = triangle.width // 4
        for chunk in range(chunks):
            first = chunk * 4
            light[row, first] = current
            for lane in range(1, 4):
                light[row, first + lane] = np.float16(light[row, first + lane - 1] + step_x)
            current = np.float16(current + chunk_step)
        for column in range(chunks * 4, triangle.width):
            light[row, column] = current
            current = np.float16(current + step_x)
        row_start = np.float16(row_start + step_y)
    return light


def ceramic_rgb565(light: np.ndarray) -> np.ndarray:
    value = np.maximum(light.astype(np.float16), np.float16(0))
    highlight = np.maximum(np.subtract(value, _half(14131), dtype=np.float16), np.float16(0))
    for _ in range(4):
        highlight = np.multiply(highlight, highlight, dtype=np.float16)
    value = np.multiply(value, _half(13783), dtype=np.float16)
    value = np.add(value, _half(12882), dtype=np.float16)
    value = np.add(value, highlight, dtype=np.float16)
    return pack_ceramic_rgb565(value)


def pack_ceramic_rgb565(value: np.ndarray) -> np.ndarray:
    if np.isnan(value).any():
        raise ValueError("NaN ceramic intensity has no defined reference packing")
    value = np.clip(value, np.float16(0), np.float16(1))
    red = np.multiply(value, np.float16(31), dtype=np.float16).astype(np.uint16)
    green = np.multiply(value, np.float16(63), dtype=np.float16).astype(np.uint16)
    return (red << 11) | (green << 5) | red


def _colors(triangle: Triangle, shading: str) -> int | np.ndarray:
    if shading == "coverage":
        return 0xFFFF
    if shading == "flat":
        return triangle.color_or_light
    with np.errstate(over="ignore", invalid="raise"):
        return ceramic_rgb565(_light_rows(triangle))


def _numpy_mask(triangle: Triangle) -> np.ndarray:
    row, column = np.ogrid[:triangle.height, :triangle.width]
    mask = np.ones((triangle.height, triangle.width), dtype=np.bool_)
    for edge, step_x, step_y in zip(triangle.edges, triangle.edge_dx, triangle.edge_dy):
        value = edge + column * step_x + row * step_y
        value = ((value + 0x80000000) & 0xFFFFFFFF) - 0x80000000
        mask &= value >= 0
    return mask


def _scalar_mask(triangle: Triangle) -> np.ndarray:
    mask = np.zeros((triangle.height, triangle.width), dtype=np.bool_)
    edges = list(triangle.edges)
    for row in range(triangle.height):
        values = edges.copy()
        for column in range(triangle.width):
            mask[row, column] = all(value >= 0 for value in values)
            values = [signed32(value + delta) for value, delta in zip(values, triangle.edge_dx)]
        edges = [signed32(value + delta) for value, delta in zip(edges, triangle.edge_dy)]
    return mask


def render(frame: Frame, renderer: str = "numpy", shading: str = "auto") -> RenderResult:
    if renderer not in ("numpy", "scalar"):
        raise ValueError("renderer must be numpy or scalar")
    if shading == "auto":
        shading = frame.shading
    if shading not in ("flat", "ceramic", "coverage"):
        raise ValueError("shading must be auto, flat, ceramic or coverage")
    if shading == "flat" and frame.shading == "ceramic":
        raise ValueError("Utah word zero is FP16 light, not RGB565; use ceramic or coverage")
    if shading == "ceramic" and frame.shading != "ceramic":
        raise ValueError("ceramic shading requires Utah light-gradient packets")
    pixels = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    touched = np.zeros_like(pixels, dtype=np.bool_)
    covered = 0
    candidates = 0
    mask_function = _numpy_mask if renderer == "numpy" else _scalar_mask
    for triangle in frame.triangles:
        mask = mask_function(triangle)
        color = _colors(triangle, shading)
        rows = slice(triangle.y, triangle.y + triangle.height)
        columns = slice(triangle.x, triangle.x + triangle.width)
        pixels[rows, columns][mask] = color if isinstance(color, int) else color[mask]
        touched[rows, columns] |= mask
        covered += int(mask.sum())
        candidates += triangle.width * triangle.height
    return RenderResult(pixels, touched, len(frame.triangles), candidates, covered, shading, renderer)


def rgb565_to_rgb8(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.uint32)
    return np.stack((
        (((values >> 11) & 31) * 255 + 15) // 31,
        (((values >> 5) & 63) * 255 + 31) // 63,
        ((values & 31) * 255 + 15) // 31,
    ), axis=-1).astype(np.uint8)
