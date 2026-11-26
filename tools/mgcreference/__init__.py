"""CPU-only reference for Aakash's precomputed edge-function scenes."""

from .scene import DATA_BASE, FRAMEBUFFER_BYTES, Frame, Triangle, load_scene, subset_frame
from .raster import RenderResult, render, rgb565_to_rgb8

__all__ = [
    "DATA_BASE", "FRAMEBUFFER_BYTES", "Frame", "Triangle", "RenderResult",
    "load_scene", "subset_frame", "render", "rgb565_to_rgb8",
]
