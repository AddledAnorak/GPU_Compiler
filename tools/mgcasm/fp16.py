"""FP16 (IEEE-754 binary16) bit-pattern helpers shared by mgcasm and mgcsim.

Uses numpy.float16 as the reference implementation for bit-exact rounding
and special values (inf/NaN), per the plan's verification requirement that
FP16 arithmetic be checked against numpy.float16.
"""

from __future__ import annotations

import numpy as np


def float_to_fp16_bits(value: float) -> int:
    """Convert a Python float to its IEEE-754 binary16 bit pattern."""
    arr = np.array([value], dtype=np.float16)
    return int(arr.view(np.uint16)[0])


def fp16_bits_to_float(bits: int) -> float:
    """Convert a 16-bit pattern to the Python float it represents."""
    arr = np.array([bits & 0xFFFF], dtype=np.uint16)
    return float(arr.view(np.float16)[0])


def _bits_to_np(bits: int) -> np.float16:
    return np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0]


def _np_to_bits(v: np.float16) -> int:
    return int(np.array([v], dtype=np.float16).view(np.uint16)[0])


def fp16_add(a_bits: int, b_bits: int) -> int:
    # Arithmetic is done in np.float16 throughout (not round-tripped
    # through a Python float) so IEEE special values -- inf, nan,
    # divide-by-zero -- follow real binary16 semantics rather than
    # Python's (which raises ZeroDivisionError instead of yielding inf).
    with np.errstate(all="ignore"):
        return _np_to_bits(_bits_to_np(a_bits) + _bits_to_np(b_bits))


def fp16_sub(a_bits: int, b_bits: int) -> int:
    with np.errstate(all="ignore"):
        return _np_to_bits(_bits_to_np(a_bits) - _bits_to_np(b_bits))


def fp16_mul(a_bits: int, b_bits: int) -> int:
    with np.errstate(all="ignore"):
        return _np_to_bits(_bits_to_np(a_bits) * _bits_to_np(b_bits))


def fp16_div(a_bits: int, b_bits: int) -> int:
    with np.errstate(all="ignore"):
        return _np_to_bits(_bits_to_np(a_bits) / _bits_to_np(b_bits))
