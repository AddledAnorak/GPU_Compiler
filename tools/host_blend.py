"""Bounded RGBA8 host equations; tuple indices are the Unix wire codes.

The equation modes use ONE/ONE for both RGB and alpha. Minimum and maximum
ignore factors, but this contract does not accept arbitrary blend factors.
"""

from __future__ import annotations

import numpy as np


BLEND_MODES = ("disabled", "source-over", "additive", "subtract",
               "reverse-subtract", "minimum", "maximum")


def validate_blend_mode(mode):
    if type(mode) is not str or mode not in BLEND_MODES:
        raise ValueError("unsupported explicit host blend mode")


def blend_rgba8(source, destination, mode):
    """Clamp shader RGBA, blend against stored RGBA8, and quantize one commit.

    Byte-scaled normalized arithmetic preserves exact halfway values without
    introducing a reciprocal-255 round trip for stored destination bytes.
    """
    validate_blend_mode(mode)
    source = np.clip(source, 0, 1)
    if mode == "source-over":
        alpha = source[3]
        source = source * 255
        destination = destination.astype(float)
        source = np.concatenate((source[:3] * alpha + destination[:3] * (1 - alpha),
                                 [255 * alpha + destination[3] * (1 - alpha)]))
        return np.rint(np.clip(source, 0, 255)).astype(np.uint8)
    elif mode != "disabled":
        source = source * 255
        destination = destination.astype(float)
        if mode == "additive":
            source = source + destination
        elif mode == "subtract":
            source = source - destination
        elif mode == "reverse-subtract":
            source = destination - source
        elif mode == "minimum":
            source = np.minimum(source, destination)
        elif mode == "maximum":
            source = np.maximum(source, destination)
        return np.rint(np.clip(source, 0, 255)).astype(np.uint8)
    return np.rint(np.clip(source, 0, 1) * 255).astype(np.uint8)
