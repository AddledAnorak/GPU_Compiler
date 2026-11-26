"""Supplied C scalar numerics and explicitly selected sample FP16 rules."""

import math
import struct


class NumericFault(ValueError):
    pass


def _unpack_half(bits):
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def half_to_float(bits, require):
    if bits & 0x8000:
        require("negative-fp16-sign")
    return _unpack_half(bits)


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_to_half(value, require):
    if math.isnan(value):
        require("arithmetic-nan")
        return 0x7E00
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (bits >> 16) & 0x8000
    exponent = ((bits >> 23) & 0xFF) - 112
    fraction = bits & 0x7FFFFF
    if exponent <= 0:
        return sign if exponent < -10 else sign | (0x800000 | fraction) >> (14 - exponent)
    if exponent >= 31:
        return sign | (0x7E00 if fraction else 0x7C00)
    return sign | (exponent << 10) | (fraction >> 13)


def scale_power(bits, power):
    sign = bits & 0x8000
    exponent = (bits >> 10) & 31
    if exponent == 31:
        return bits & 0xFFFF
    if exponent == 0 or exponent + power <= 0:
        return sign
    if exponent + power >= 31:
        return sign | 0x7C00
    return sign | ((exponent + power) << 10) | (bits & 0x3FF)


def binary(opcode, selector, left_bits, right_bits, require):
    left = half_to_float(left_bits, require)
    right = half_to_float(right_bits, require)
    if opcode == 4:
        value = left - right if selector else left + right
    elif opcode == 5:
        value = left * right
    elif opcode == 6:
        if right == 0.0:
            return ((left_bits ^ right_bits) & 0x8000) | 0x7C00
        value = left / right
    else:
        take_left = left > right if selector else left < right
        return (left_bits if take_left else right_bits) & 0xFFFF
    return float_to_half(value, require)


def root(selector, bits, require):
    value = half_to_float(bits, require)
    if value < 0.0 or (selector == 3 and value == 0.0):
        return 0x7E00
    square_root = _float32(math.sqrt(value))
    return float_to_half(square_root if selector == 2 else 1.0 / square_root, require)


def round_to_integer(selector, bits, require):
    value = half_to_float(bits, require)
    if math.isfinite(value):
        magnitude = math.floor(abs(value) + 0.5)
        if selector == 0:
            integral = math.trunc(value)
        elif selector == 1:
            integral = magnitude if value >= 0.0 else -magnitude
        elif selector == 2:
            integral = math.ceil(value)
        else:
            integral = math.floor(value)
        value = float(integral) if integral else math.copysign(0.0, value)
    return float_to_half(value, require)


def packed_integer(bits, maximum, require):
    value = half_to_float(bits, require)
    if math.isnan(value):
        raise NumericFault("C float-to-integer conversion of NaN is undefined")
    return int(min(maximum, max(0.0, value)))


def integer32(bits, require):
    value = half_to_float(bits, require)
    if value <= 0.0:
        return 0
    if not math.isfinite(value):
        raise NumericFault("C float-to-integer conversion of NaN or positive infinity is undefined")
    return int(value)


def _sample_round(value):
    if math.isnan(value):
        raise NumericFault("NaN results are not specified by the sample FP16 evidence")
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return 0xFC00 if value < 0 else 0x7C00


def sample_binary(selector, left_bits, right_bits):
    left = _unpack_half(left_bits)
    right = _unpack_half(right_bits)
    if math.isnan(left) or math.isnan(right):
        raise NumericFault("NaN operands are not specified by the sample FP16 evidence")
    if selector == 7:
        return _sample_round(left + right)
    if selector == 8:
        return _sample_round(left - right)
    if selector == 9:
        return _sample_round(left * right)
    if selector == 10:
        return (left_bits if left >= right else right_bits) & 0xFFFF
    raise NumericFault(f"unknown sample FP16 selector {selector}")


def sample_setup(seed_bits, step_bits):
    values = [seed_bits & 0xFFFF]
    for _ in range(3):
        values.append(sample_binary(7, values[-1], step_bits))
    return tuple(values)


def sample_ceramic(bits, fp16_scale):
    value = _unpack_half(bits)
    if math.isnan(value):
        raise NumericFault("NaN RGB565 color is not specified by the sample evidence")
    value = max(0.0, min(1.0, value))
    red, green = value * 31, value * 63
    if fp16_scale:
        red = _unpack_half(_sample_round(red))
        green = _unpack_half(_sample_round(green))
    return (int(red) << 11) | (int(green) << 5) | int(red)
