"""Pure, bounded CPU-host fragment state; not a GL or hardware implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math


STATE_VERSION = 1
COLOR_FORMATS = ("rgba8", "bgra8", "rgb565", "r8")
DEPTH_STENCIL_FORMATS = ("none", "z16", "z24s8")
BLEND_FACTORS = (
    "zero", "one", "src-color", "one-minus-src-color", "dst-color",
    "one-minus-dst-color", "src-alpha", "one-minus-src-alpha", "dst-alpha",
    "one-minus-dst-alpha", "constant-color", "one-minus-constant-color",
    "constant-alpha", "one-minus-constant-alpha", "src-alpha-saturate",
)
BLEND_EQUATIONS = ("add", "subtract", "reverse-subtract", "min", "max")
COMPARISONS = ("never", "less", "equal", "lequal", "greater", "notequal", "gequal", "always")
STENCIL_OPS = (
    "keep", "zero", "replace", "increment-clamp", "decrement-clamp", "invert",
    "increment-wrap", "decrement-wrap",
)
_COLOR_SIZES = {"rgba8": 4, "bgra8": 4, "rgb565": 2, "r8": 1}
_DEPTH_SIZES = {"z16": 2, "z24s8": 4}
_DEPTH_MAXIMUM = {"z16": 65535, "z24s8": 16777215}


def _object(value, keys, name):
    if (type(value) is not dict or len(value) != len(keys)
            or any(type(key) is not str for key in value) or set(value) != set(keys)):
        raise ValueError(f"{name} must be a dict with exactly {', '.join(keys)}")
    return value


def _enum(value, choices, name):
    if type(value) is not str or value not in choices:
        raise ValueError(f"unsupported {name}")


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")


def _integer(value, maximum, name):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")


def _normalized(value, name, *, clamp=False):
    if type(value) not in (int, float, Fraction) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite int, float or Fraction")
    if clamp:
        value = min(max(value, 0), 1)
    elif not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return Fraction(value)


def _rgba(value, name):
    if type(value) not in (tuple, list) or len(value) != 4:
        raise ValueError(f"{name} must contain exactly four channels")
    return tuple(_normalized(channel, name, clamp=True) for channel in value)


def _buffer(value, size, name):
    if type(value) not in (bytes, bytearray) or len(value) != size:
        raise ValueError(f"{name} must be exactly {size} bytes or bytearray bytes")
    return bytes(value)


@dataclass(frozen=True, slots=True)
class StencilFace:
    comparison: str = "always"
    reference: int = 0
    read_mask: int = 255
    write_mask: int = 255
    fail: str = "keep"
    zfail: str = "keep"
    zpass: str = "keep"

    def __post_init__(self):
        _enum(self.comparison, COMPARISONS, "stencil comparison")
        for name in ("reference", "read_mask", "write_mask"):
            _integer(getattr(self, name), 255, f"stencil {name}")
        for name in ("fail", "zfail", "zpass"):
            _enum(getattr(self, name), STENCIL_OPS, f"stencil {name}")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "stencil face")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PipelineState:
    version: int = STATE_VERSION
    color_format: str = "rgba8"
    depth_stencil_format: str = "z16"
    blend_enabled: bool = False
    src_rgb: str = "one"
    dst_rgb: str = "zero"
    src_alpha: str = "one"
    dst_alpha: str = "zero"
    rgb_equation: str = "add"
    alpha_equation: str = "add"
    constant: tuple = (0, 0, 0, 0)
    color_mask: tuple = (True, True, True, True)
    depth_enabled: bool = False
    depth_write: bool = True
    depth_comparison: str = "less"
    stencil_enabled: bool = False
    stencil_front: StencilFace = field(default_factory=StencilFace)
    stencil_back: StencilFace = field(default_factory=StencilFace)

    def __post_init__(self):
        if type(self.version) is not int or self.version != STATE_VERSION:
            raise ValueError("unsupported rendering state version")
        _enum(self.color_format, COLOR_FORMATS, "color format")
        _enum(self.depth_stencil_format, DEPTH_STENCIL_FORMATS, "depth/stencil format")
        for name in ("blend_enabled", "depth_enabled", "depth_write", "stencil_enabled"):
            _boolean(getattr(self, name), name)
        for name in ("src_rgb", "dst_rgb", "src_alpha", "dst_alpha"):
            choices = BLEND_FACTORS if name.startswith("src_") else BLEND_FACTORS[:-1]
            _enum(getattr(self, name), choices, name)
        for name in ("rgb_equation", "alpha_equation"):
            _enum(getattr(self, name), BLEND_EQUATIONS, name)
        if type(self.constant) is not tuple or len(self.constant) != 4:
            raise ValueError("blend constant must be a four-channel tuple")
        for channel in self.constant:
            if type(channel) not in (int, float):
                raise ValueError("blend constant channels must be JSON int or float values")
            _normalized(channel, "blend constant")
        if type(self.color_mask) is not tuple or len(self.color_mask) != 4:
            raise ValueError("color mask must be a four-channel tuple")
        for channel in self.color_mask:
            _boolean(channel, "color mask channel")
        _enum(self.depth_comparison, COMPARISONS, "depth comparison")
        for face in (self.stencil_front, self.stencil_back):
            if type(face) is not StencilFace:
                raise ValueError("stencil faces must be StencilFace values")
            face.__post_init__()
        if self.depth_enabled and self.depth_stencil_format == "none":
            raise ValueError("depth testing requires a depth attachment")
        if self.stencil_enabled and self.depth_stencil_format != "z24s8":
            raise ValueError("stencil testing requires z24s8")

    def to_dict(self):
        return {
            "version": self.version,
            "framebuffer": {"color_format": self.color_format,
                            "depth_stencil_format": self.depth_stencil_format},
            "blend": {"enabled": self.blend_enabled, "src_rgb": self.src_rgb,
                      "dst_rgb": self.dst_rgb, "src_alpha": self.src_alpha,
                      "dst_alpha": self.dst_alpha, "rgb_equation": self.rgb_equation,
                      "alpha_equation": self.alpha_equation, "constant": list(self.constant)},
            "color_mask": list(self.color_mask),
            "depth": {"enabled": self.depth_enabled, "write": self.depth_write,
                      "comparison": self.depth_comparison},
            "stencil": {"enabled": self.stencil_enabled, "front": self.stencil_front.to_dict(),
                        "back": self.stencil_back.to_dict()},
        }

    @classmethod
    def from_dict(cls, value):
        _object(value, ("version", "framebuffer", "blend", "color_mask", "depth", "stencil"), "state")
        framebuffer = _object(value["framebuffer"], ("color_format", "depth_stencil_format"), "framebuffer")
        blend = _object(value["blend"], ("enabled", "src_rgb", "dst_rgb", "src_alpha", "dst_alpha",
                                         "rgb_equation", "alpha_equation", "constant"), "blend")
        depth = _object(value["depth"], ("enabled", "write", "comparison"), "depth")
        stencil = _object(value["stencil"], ("enabled", "front", "back"), "stencil")
        if (type(blend["constant"]) is not list or len(blend["constant"]) != 4
                or type(value["color_mask"]) is not list or len(value["color_mask"]) != 4):
            raise ValueError("serialized constant and color mask must be four-element lists")
        return cls(
            version=value["version"], **framebuffer, blend_enabled=blend["enabled"],
            src_rgb=blend["src_rgb"], dst_rgb=blend["dst_rgb"], src_alpha=blend["src_alpha"],
            dst_alpha=blend["dst_alpha"], rgb_equation=blend["rgb_equation"],
            alpha_equation=blend["alpha_equation"], constant=tuple(blend["constant"]),
            color_mask=tuple(value["color_mask"]), depth_enabled=depth["enabled"],
            depth_write=depth["write"], depth_comparison=depth["comparison"],
            stencil_enabled=stencil["enabled"], stencil_front=StencilFace.from_dict(stencil["front"]),
            stencil_back=StencilFace.from_dict(stencil["back"]),
        )


def validate_state(state):
    """Validate a complete exact dict or an immutable PipelineState, without coercion."""
    if type(state) is PipelineState:
        state.__post_init__()
        return state
    return PipelineState.from_dict(state)


def default_state(*, blend_mode="disabled", color_format="rgba8", depth_stencil_format="z16"):
    """Build the explicit legacy disabled/source-over mapping as a fresh version-1 dict."""
    _enum(blend_mode, ("disabled", "source-over"), "legacy blend mode")
    source_over = blend_mode == "source-over"
    return PipelineState(
        color_format=color_format, depth_stencil_format=depth_stencil_format,
        blend_enabled=source_over, src_rgb="src-alpha" if source_over else "one",
        dst_rgb="one-minus-src-alpha" if source_over else "zero",
        dst_alpha="one-minus-src-alpha" if source_over else "zero",
    ).to_dict()


def _quantize(value, maximum):
    return round(min(max(value, 0), 1) * maximum)


def _encode_color(color_format, rgba):
    if color_format == "rgb565":
        word = (_quantize(rgba[0], 31) << 11) | (_quantize(rgba[1], 63) << 5) | _quantize(rgba[2], 31)
        return word.to_bytes(2, "little")
    if color_format == "r8":
        return bytes((_quantize(rgba[0], 255),))
    quantized = tuple(_quantize(channel, 255) for channel in rgba)
    if color_format == "bgra8":
        quantized = (quantized[2], quantized[1], quantized[0], quantized[3])
    return bytes(quantized)


def _decode_color(color_format, data):
    if color_format == "rgb565":
        word = int.from_bytes(data, "little")
        return (Fraction(word >> 11, 31), Fraction((word >> 5) & 63, 63),
                Fraction(word & 31, 31), Fraction(1))
    if color_format == "r8":
        return (Fraction(data[0], 255), Fraction(0), Fraction(0), Fraction(1))
    channels = (data[2], data[1], data[0], data[3]) if color_format == "bgra8" else data
    return tuple(Fraction(channel, 255) for channel in channels)


def pack_color(color_format, rgba):
    """Clamp finite RGBA and pack normalized channels using exact ties-to-even."""
    _enum(color_format, COLOR_FORMATS, "color format")
    return _encode_color(color_format, _rgba(rgba, "RGBA"))


def unpack_color(color_format, data):
    """Return exact normalized Fraction RGBA; absent RGB is zero and absent alpha one."""
    _enum(color_format, COLOR_FORMATS, "color format")
    return _decode_color(color_format, _buffer(data, _COLOR_SIZES[color_format], "color"))


def _encode_depth_stencil(depth_format, depth_code, stencil):
    word = depth_code if depth_format == "z16" else depth_code | (stencil << 24)
    return word.to_bytes(_DEPTH_SIZES[depth_format], "little")


def _decode_depth_stencil(depth_format, data):
    word = int.from_bytes(data, "little")
    return (word, 0) if depth_format == "z16" else (word & 16777215, word >> 24)


def pack_depth_stencil(depth_format, depth, stencil=0):
    """Pack little-endian Z16 or Z24S8 with depth low 24 bits, stencil high 8 bits."""
    _enum(depth_format, DEPTH_STENCIL_FORMATS[1:], "depth/stencil format")
    normalized = _normalized(depth, "depth")
    _integer(stencil, 255, "stencil")
    if depth_format == "z16" and stencil != 0:
        raise ValueError("z16 cannot store stencil")
    return _encode_depth_stencil(depth_format, _quantize(normalized, _DEPTH_MAXIMUM[depth_format]), stencil)


def unpack_depth_stencil(depth_format, data):
    """Return (exact normalized Fraction depth, integer stencil)."""
    _enum(depth_format, DEPTH_STENCIL_FORMATS[1:], "depth/stencil format")
    raw = _buffer(data, _DEPTH_SIZES[depth_format], "depth/stencil")
    depth_code, stencil = _decode_depth_stencil(depth_format, raw)
    return Fraction(depth_code, _DEPTH_MAXIMUM[depth_format]), stencil


def _factor(name, channel, source, destination, constant):
    if name == "src-alpha-saturate":
        return Fraction(1) if channel == 3 else min(source[3], 1 - destination[3])
    inverse = name.startswith("one-minus-")
    base = name.removeprefix("one-minus-")
    if base == "zero":
        result = Fraction(0)
    elif base == "one":
        result = Fraction(1)
    elif base == "src-color":
        result = source[channel]
    elif base == "dst-color":
        result = destination[channel]
    elif base == "src-alpha":
        result = source[3]
    elif base == "dst-alpha":
        result = destination[3]
    elif base == "constant-color":
        result = constant[channel]
    else:
        result = constant[3]
    return 1 - result if inverse else result


def _blend(state, source, destination):
    if not state.blend_enabled:
        result = source
    else:
        constant = tuple(Fraction(channel) for channel in state.constant)
        channels = []
        for channel in range(4):
            equation = state.alpha_equation if channel == 3 else state.rgb_equation
            if equation == "min":
                value = min(source[channel], destination[channel])
            elif equation == "max":
                value = max(source[channel], destination[channel])
            else:
                source_factor = state.src_alpha if channel == 3 else state.src_rgb
                destination_factor = state.dst_alpha if channel == 3 else state.dst_rgb
                source_term = source[channel] * _factor(source_factor, channel, source, destination, constant)
                destination_term = destination[channel] * _factor(destination_factor, channel, source, destination, constant)
                if equation == "add":
                    value = source_term + destination_term
                elif equation == "subtract":
                    value = source_term - destination_term
                else:
                    value = destination_term - source_term
            channels.append(min(max(value, 0), 1))
        result = tuple(channels)
    return tuple(result[channel] if state.color_mask[channel] else destination[channel] for channel in range(4))


def blend_color(state, source_rgba, destination_color):
    """Pure blend/mask/format operation, without depth or stencil tests."""
    pipeline = validate_state(state)
    source = _rgba(source_rgba, "source RGBA")
    raw = _buffer(destination_color, _COLOR_SIZES[pipeline.color_format], "destination color")
    destination = _decode_color(pipeline.color_format, raw)
    return _encode_color(pipeline.color_format, _blend(pipeline, source, destination))


def _compare(comparison, incoming, stored):
    return {
        "never": False, "less": incoming < stored, "equal": incoming == stored,
        "lequal": incoming <= stored, "greater": incoming > stored,
        "notequal": incoming != stored, "gequal": incoming >= stored, "always": True,
    }[comparison]


def _stencil_operation(operation, stored, reference):
    if operation == "keep":
        return stored
    if operation == "zero":
        return 0
    if operation == "replace":
        return reference
    if operation == "increment-clamp":
        return min(stored + 1, 255)
    if operation == "decrement-clamp":
        return max(stored - 1, 0)
    if operation == "invert":
        return stored ^ 255
    if operation == "increment-wrap":
        return (stored + 1) & 255
    return (stored - 1) & 255


@dataclass(frozen=True, slots=True)
class FragmentCommit:
    color: bytes
    depth_stencil: bytes | None
    outcome: str
    color_written: bool = False
    depth_written: bool = False
    stencil_written: bool = False

    @property
    def passed(self):
        return self.outcome == "passed"

    @property
    def killed(self):
        return self.outcome == "killed"


def commit_fragment(state, source_rgba, destination_color, *, source_depth,
                    destination_depth_stencil, front_facing=True, killed=False):
    """Validate then stage one fragment's color/depth/stencil transaction.

    No caller-owned object is changed. A killed fragment suppresses all writes;
    otherwise stencil precedes depth, selecting exactly one fail/zfail/zpass op.
    Incoming depth is quantized to the attachment precision before comparison.
    """
    pipeline = validate_state(state)
    source = _rgba(source_rgba, "source RGBA")
    color = _buffer(destination_color, _COLOR_SIZES[pipeline.color_format], "destination color")
    depth = _normalized(source_depth, "source depth")
    _boolean(front_facing, "front_facing")
    _boolean(killed, "killed")
    depth_format = pipeline.depth_stencil_format
    if depth_format == "none":
        if destination_depth_stencil is not None:
            raise ValueError("no depth/stencil attachment requires None")
        attachment = None
        incoming_depth = stored_depth = stored_stencil = 0
    else:
        attachment = _buffer(destination_depth_stencil, _DEPTH_SIZES[depth_format], "destination depth/stencil")
        stored_depth, stored_stencil = _decode_depth_stencil(depth_format, attachment)
        incoming_depth = _quantize(depth, _DEPTH_MAXIMUM[depth_format])
    if killed:
        return FragmentCommit(color, attachment, "killed")
    face = pipeline.stencil_front if front_facing else pipeline.stencil_back
    if pipeline.stencil_enabled and not _compare(
        face.comparison, face.reference & face.read_mask, stored_stencil & face.read_mask,
    ):
        outcome, operation = "stencil-fail", face.fail
    elif pipeline.depth_enabled and not _compare(pipeline.depth_comparison, incoming_depth, stored_depth):
        outcome, operation = "depth-fail", face.zfail
    else:
        outcome, operation = "passed", face.zpass
    stencil_written = pipeline.stencil_enabled and operation != "keep" and face.write_mask != 0
    if stencil_written:
        updated = _stencil_operation(operation, stored_stencil, face.reference)
        stored_stencil = (stored_stencil & (255 ^ face.write_mask)) | (updated & face.write_mask)
    passed = outcome == "passed"
    depth_written = passed and pipeline.depth_enabled and pipeline.depth_write
    if depth_written:
        stored_depth = incoming_depth
    if depth_written or stencil_written:
        attachment = _encode_depth_stencil(depth_format, stored_depth, stored_stencil)
    present_channels = 1 if pipeline.color_format == "r8" else 3 if pipeline.color_format == "rgb565" else 4
    color_written = passed and any(pipeline.color_mask[:present_channels])
    if color_written:
        destination = _decode_color(pipeline.color_format, color)
        color = _encode_color(pipeline.color_format, _blend(pipeline, source, destination))
    return FragmentCommit(color, attachment, outcome, color_written, depth_written, stencil_written)
