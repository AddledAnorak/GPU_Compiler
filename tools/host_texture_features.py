"""Validate captured ABI4 texture resources and pack input words, never outputs."""

from __future__ import annotations

import re

from tools.mgcasm.fp16 import float_to_fp16_bits


TEXTURE_BASE = 0x110000
TEXTURE_REGION_WORDS = 65536
TEXTURE_STRIDE_WORDS = 16384
TEXTURE_HEADER_WORDS = 32
MAX_BINDINGS = 4
MAX_SIZE = 32
MAX_LEVELS = 6

_METADATA_LIMITS = {
    "max_bindings": MAX_BINDINGS,
    "max_samples": 32,
    "max_size": MAX_SIZE,
    "max_levels": MAX_LEVELS,
    "per_binding_stride": TEXTURE_STRIDE_WORDS,
    "header_words": TEXTURE_HEADER_WORDS,
}
_METADATA_FIELDS = {
    "version", "bindings", "lod_mode", "wrap_s", "wrap_t", "filter", "mip_filter",
    *_METADATA_LIMITS,
}
_CAPTURE_FIELDS = {
    "binding", "texture", "sampler", "width", "height", "format", "data_hex",
    "sampler_state", "mipmaps",
}
_MIPMAP_FIELDS = {"level", "width", "height", "format", "data_hex"}
_WRAPS = {"clamp-to-edge", "repeat", "mirrored-repeat"}
_LOD_MODES = {"explicit", "implicit", "gradients", "mixed"}
_FORMAT_CHANNELS = {
    "R8G8B8A8_UNORM": 4,
    "B8G8R8A8_UNORM": 4,
    "R8G8B8_UNORM": 3,
    "R8_UNORM": 1,
}
_NORMALIZED_WORDS = tuple(float_to_fp16_bits(channel / 255.0) for channel in range(256))


def _exact_fields(value, fields, name):
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} requires exactly {', '.join(sorted(fields))}")


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _choice(value, choices, name):
    if type(value) is not str or value not in choices:
        raise ValueError(f"unsupported {name}")
    return value


def _exact_values(value, expected, name):
    _exact_fields(value, set(expected), name)
    if any(type(value[key]) is not type(required) or value[key] != required
           for key, required in expected.items()):
        raise ValueError(f"{name} does not match the compiled texture contract")


def validate_texture_features_metadata(metadata: dict) -> dict:
    """Return a detached exact compiler host_texture contract for versions11–14."""
    version = metadata.get("version") if type(metadata) is dict else None
    fields = _METADATA_FIELDS
    if type(version) is int and version == 14:
        fields = fields | {"max_anisotropy"}
    _exact_fields(metadata, fields, "ABI4 host texture metadata")
    version = _integer(version, "host texture version", 11, 14)
    if version == 14:
        _integer(metadata["max_anisotropy"], "max_anisotropy", 4, 4)
    for name, expected in _METADATA_LIMITS.items():
        _integer(metadata[name], name, expected, expected)
    _choice(metadata["lod_mode"], _LOD_MODES, "texture LOD mode")
    for axis in ("wrap_s", "wrap_t"):
        _choice(metadata[axis], _WRAPS, axis)
    expected_filter = "nearest" if version == 11 else "linear"
    expected_mip_filter = "linear" if version in (13, 14) else "nearest"
    _choice(metadata["filter"], {expected_filter}, "texture filter for version")
    _choice(metadata["mip_filter"], {expected_mip_filter}, "texture mip filter for version")
    bindings = metadata["bindings"]
    if type(bindings) is not list or not 1 <= len(bindings) <= MAX_BINDINGS:
        raise ValueError("host texture bindings must be an explicit list of one to four pairs")
    pairs = set()
    for binding in bindings:
        _exact_fields(binding, {"texture", "sampler"}, "texture binding pair")
        pair = tuple(_integer(binding[name], name, 0, MAX_BINDINGS - 1)
                     for name in ("texture", "sampler"))
        if pair in pairs:
            raise ValueError("duplicate used texture/sampler binding pair")
        pairs.add(pair)
    return {**metadata, "bindings": [dict(binding) for binding in bindings]}


def _layout_metadata(source):
    _integer(source.get("version"), "compiler layout version", 4, 4)
    _integer(source.get("shader_abi_version"), "shader ABI version", 4, 4)
    if source.get("supported") is not True:
        raise ValueError("texture packing requires a supported compiler layout")
    _choice(source.get("program_format"), {"legacy-mgcsim-v1"}, "texture program format")
    _choice(source.get("address_units"), {"words"}, "texture address units")
    _choice(source.get("stage"), {"vertex", "fragment"}, "shader stage")
    features = source.get("host_features")
    if type(features) is not dict:
        raise ValueError("ABI4 texture packing requires host_features metadata")
    _integer(features.get("version"), "host features version", 1, 1)
    bases = source.get("bases")
    _exact_fields(bases, {"input", "uniform", "vertex", "fragment", "texture"}, "ABI4 bases")
    spans = []
    for name, base in bases.items():
        _integer(base, f"{name} base", 0, (1 << 32) - TEXTURE_REGION_WORDS)
        spans.append((base, base + TEXTURE_REGION_WORDS))
    if bases["texture"] != TEXTURE_BASE:
        raise ValueError("ABI4 texture base must be 0x110000 word-addressed")
    if any(max(left[0], right[0]) < min(left[1], right[1])
           for index, left in enumerate(spans) for right in spans[index + 1:]):
        raise ValueError("ABI4 resource domains overlap or alias")
    uniforms = source.get("uniforms")
    if type(uniforms) is not list:
        raise ValueError("ordinary uniform reflection must be a list")
    for uniform in uniforms:
        if type(uniform) is not dict:
            raise ValueError("ordinary uniform reflection must contain objects")
        _integer(uniform.get("word_offset"), "uniform word offset", 0, TEXTURE_REGION_WORDS - 1)
    if "host_texture" not in source:
        return None
    if source["stage"] != "fragment":
        raise ValueError("ABI4 host textures require a fragment shader")
    return validate_texture_features_metadata(source["host_texture"])


def _level_data(entry, expected_dimensions=None, expected_format=None):
    width = _integer(entry["width"], "texture width", 1, MAX_SIZE)
    height = _integer(entry["height"], "texture height", 1, MAX_SIZE)
    if expected_dimensions is not None and (width, height) != expected_dimensions:
        raise ValueError("mipmap dimensions must follow the complete floor-halved chain")
    source_format = _choice(entry["format"], set(_FORMAT_CHANNELS), "texture source format")
    if expected_format is not None and source_format != expected_format:
        raise ValueError("all mipmap levels must use the texture resource format")
    encoded = entry["data_hex"]
    byte_count = width * height * _FORMAT_CHANNELS[source_format]
    if (type(encoded) is not str or len(encoded) != byte_count * 2 or
            re.fullmatch(r"[0-9a-fA-F]+", encoded) is None):
        raise ValueError("texture data_hex must contain exact tightly packed source-format bytes")
    return width, height, source_format, bytes.fromhex(encoded)


def _capture_levels(capture, contract):
    sampler = {
        "normalized_coords": True,
        "min_filter": contract["filter"],
        "mag_filter": contract["filter"],
        "mip_filter": contract["mip_filter"],
        "wrap_s": contract["wrap_s"],
        "wrap_t": contract["wrap_t"],
    }
    if contract["version"] == 14:
        sampler["max_anisotropy"] = contract["max_anisotropy"]
    _exact_values(capture["sampler_state"], sampler, "captured sampler_state")
    base = _level_data(capture)
    width, height, source_format = base[:3]
    levels = [base]
    mipmaps = capture["mipmaps"]
    level_count = max(width, height).bit_length()
    if type(mipmaps) is not list or len(mipmaps) != level_count - 1:
        raise ValueError("mipmaps must provide the complete chain through 1x1")
    for level_index, entry in enumerate(mipmaps, 1):
        _exact_fields(entry, _MIPMAP_FIELDS, "captured mipmap level")
        _integer(entry["level"], "mipmap level", level_index, level_index)
        width, height = max(1, width // 2), max(1, height // 2)
        levels.append(_level_data(entry, (width, height), source_format))
    if (len(levels) > MAX_LEVELS or
            TEXTURE_HEADER_WORDS + sum(level[0] * level[1] * 4 for level in levels) > TEXTURE_STRIDE_WORDS):
        raise ValueError("complete texture payload exceeds its bounded binding plane")
    return levels


def _texel_words(source_format, contents):
    channels = _FORMAT_CHANNELS[source_format]
    for offset in range(0, len(contents), channels):
        if source_format == "B8G8R8A8_UNORM":
            rgba = (contents[offset + 2], contents[offset + 1], contents[offset], contents[offset + 3])
        elif channels == 4:
            rgba = contents[offset:offset + 4]
        elif channels == 3:
            rgba = (*contents[offset:offset + 3], 255)
        else:
            rgba = (contents[offset], 0, 0, 255)
        for channel in rgba:
            yield _NORMALIZED_WORDS[channel]


def pack_texture_features(layout_or_contract, textures: list[dict]) -> dict[int, int]:
    """Return texture-domain-relative RAW32 input words for every captured mip.

    Accept an ABI4 compiler layout, an object exposing its .source, or the exact
    host_texture metadata contract. The caller owns shader execution and the
    texture buffer at TEXTURE_BASE; no LOD selection or sampling occurs here.
    """
    source = getattr(layout_or_contract, "source", layout_or_contract)
    if type(source) is not dict:
        raise ValueError("texture packing requires a compiler layout or exact metadata contract")
    if type(textures) is not list:
        raise ValueError("captured textures must be an explicit list")
    if any(name in source for name in ("host_texture", "bases", "program_format")):
        contract = _layout_metadata(source)
    else:
        contract = validate_texture_features_metadata(source)
    if contract is None:
        if textures:
            raise ValueError("captured textures require compiled host_texture metadata")
        return {}
    if len(textures) != len(contract["bindings"]):
        raise ValueError("captured textures must match every used texture/sampler pair exactly")
    expected_pairs = {(pair["texture"], pair["sampler"]) for pair in contract["bindings"]}
    captured_pairs = set()
    resources = {}
    for capture in textures:
        _exact_fields(capture, _CAPTURE_FIELDS, "captured texture")
        binding = _integer(capture["binding"], "captured binding", 0, MAX_BINDINGS - 1)
        texture = _integer(capture["texture"], "captured texture slot", 0, MAX_BINDINGS - 1)
        sampler = _integer(capture["sampler"], "captured sampler slot", 0, MAX_BINDINGS - 1)
        if binding != texture:
            raise ValueError("captured binding must identify its texture slot, not its sampler")
        pair = (texture, sampler)
        if pair not in expected_pairs:
            raise ValueError("captured texture/sampler pair is not declared by the shader")
        if pair in captured_pairs:
            raise ValueError("duplicate captured texture/sampler pair")
        levels = _capture_levels(capture, contract)
        if texture in resources:
            if resources[texture] != levels:
                raise ValueError("conflicting read-only texture binding plane alias")
        else:
            resources[texture] = levels
        captured_pairs.add(pair)
    if captured_pairs != expected_pairs:
        raise ValueError("captured textures must match every used texture/sampler pair exactly")
    payload = {}
    for texture in sorted(resources):
        levels = resources[texture]
        plane = texture * TEXTURE_STRIDE_WORDS
        payload.update({plane + offset: 0 for offset in range(TEXTURE_HEADER_WORDS)})
        payload[plane] = len(levels)
        texel_offset = TEXTURE_HEADER_WORDS
        for level_index, (width, height, source_format, contents) in enumerate(levels):
            record = plane + 1 + level_index * 3
            payload.update({record: width, record + 1: height, record + 2: texel_offset})
            for word in _texel_words(source_format, contents):
                payload[plane + texel_offset] = word
                texel_offset += 1
    return payload
