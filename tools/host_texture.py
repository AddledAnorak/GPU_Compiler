"""Validate and pack input texels for the compiled legacy host texture ABI."""

from __future__ import annotations

import re

from tools.mgcasm.fp16 import float_to_fp16_bits
from tools.mgcdriver import ProgramFormat
from tools.host_instancing import validate_shader_instancing


HOST_TEXTURE_LAYOUT = {
    "version": 1,
    "texture": 0,
    "sampler": 0,
    "storage_block": 15,
    "max_width": 16,
    "max_height": 16,
    "filter": "nearest",
    "wrap": "clamp-to-edge",
}

SAMPLER_STATE = {
    "normalized_coords": True,
    "min_filter": "nearest",
    "mag_filter": "nearest",
    "mip_filter": "none",
    "wrap_s": "clamp-to-edge",
    "wrap_t": "clamp-to-edge",
}

HOST_TEXTURE_LINEAR_LAYOUT = {**HOST_TEXTURE_LAYOUT, "version": 2, "filter": "linear"}
LINEAR_SAMPLER_STATE = {**SAMPLER_STATE, "min_filter": "linear", "mag_filter": "linear"}
HOST_TEXTURE_REPEAT_LAYOUT = {**HOST_TEXTURE_LAYOUT, "version": 3, "wrap": "repeat"}
HOST_TEXTURE_LINEAR_REPEAT_LAYOUT = {**HOST_TEXTURE_LINEAR_LAYOUT, "version": 4, "wrap": "repeat"}
REPEAT_SAMPLER_STATE = {**SAMPLER_STATE, "wrap_s": "repeat", "wrap_t": "repeat"}
LINEAR_REPEAT_SAMPLER_STATE = {**LINEAR_SAMPLER_STATE, "wrap_s": "repeat", "wrap_t": "repeat"}
HOST_TEXTURE_MIRRORED_REPEAT_LAYOUT = {**HOST_TEXTURE_LAYOUT, "version": 5, "wrap": "mirrored-repeat"}
HOST_TEXTURE_LINEAR_MIRRORED_REPEAT_LAYOUT = {
    **HOST_TEXTURE_LINEAR_LAYOUT, "version": 6, "wrap": "mirrored-repeat"}
MIRRORED_REPEAT_SAMPLER_STATE = {**SAMPLER_STATE, "wrap_s": "mirrored-repeat", "wrap_t": "mirrored-repeat"}
LINEAR_MIRRORED_REPEAT_SAMPLER_STATE = {
    **LINEAR_SAMPLER_STATE, "wrap_s": "mirrored-repeat", "wrap_t": "mirrored-repeat"}

TEXTURE_BASE = 15 * 4096
TEXTURE_WRAPS = ("clamp-to-edge", "repeat", "mirrored-repeat")


def _exact_object(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{name} must have exactly the supported fields")
    if any(type(value[key]) is not type(required) or value[key] != required
           for key, required in expected.items()):
        raise ValueError(f"unsupported {name} contract or state")


def host_texture_contract(version, *, wrap_s=None, wrap_t=None, lod=None):
    if type(version) is not int or version not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        raise ValueError("unsupported host texture version")
    if version in (7, 8, 9, 10):
        if any(type(wrap) is not str or wrap not in TEXTURE_WRAPS for wrap in (wrap_s, wrap_t)):
            raise ValueError("mixed-axis host texture requires explicit supported S/T wraps")
        contract = {**HOST_TEXTURE_LAYOUT, "version": version,
                    "filter": "linear" if version in (8, 10) else "nearest",
                    "wrap_s": wrap_s, "wrap_t": wrap_t}
        del contract["wrap"]
        if version in (9, 10):
            if type(lod) is not int or not 0 <= lod <= 4:
                raise ValueError("host mipmaps require a reflected explicit integer LOD in [0, 4]")
            contract.update(lod=lod, mip_filter="nearest", max_levels=5)
        elif lod is not None:
            raise ValueError("non-mipmap host texture versions do not accept LOD metadata")
        return contract
    if wrap_s is not None or wrap_t is not None or lod is not None:
        raise ValueError("paired host texture versions do not accept per-axis metadata")
    contracts = (HOST_TEXTURE_LAYOUT, HOST_TEXTURE_LINEAR_LAYOUT,
                 HOST_TEXTURE_REPEAT_LAYOUT, HOST_TEXTURE_LINEAR_REPEAT_LAYOUT,
                 HOST_TEXTURE_MIRRORED_REPEAT_LAYOUT, HOST_TEXTURE_LINEAR_MIRRORED_REPEAT_LAYOUT)
    return dict(contracts[version - 1])


def host_texture_metadata_contract(metadata):
    if type(metadata) is not dict:
        raise ValueError("host texture metadata must be an object")
    return host_texture_contract(metadata.get("version"), wrap_s=metadata.get("wrap_s"),
                                 wrap_t=metadata.get("wrap_t"), lod=metadata.get("lod"))


def _texture_bytes(encoded, width, height):
    if (type(encoded) is not str or len(encoded) != width * height * 8 or
            re.fullmatch(r"[0-9a-fA-F]+", encoded) is None):
        raise ValueError("texture bytes must be exact tightly packed row-major RGBA8 hexadecimal")
    return bytes.fromhex(encoded)


def pack_texture(layout, textures):
    """Return uniform scalar words containing only dimensions and uploaded texels."""
    source = getattr(layout, "source", layout)
    if type(source) is not dict or type(source.get("version")) is not int:
        raise ValueError("texture packing requires a compiler layout object and integer version")
    if type(textures) is not list:
        raise ValueError("captured textures must be an explicit list")
    instancing = validate_shader_instancing(source)
    if "host_texture" not in source:
        if (source["version"] != 1 and not instancing) or textures:
            raise ValueError("textures require an actual host texture compiler layout")
        return {}
    if (source["version"] not in (2, 3) or source.get("supported") is not True or
            source.get("program_format") != ProgramFormat.LEGACY.value or
            type(source.get("shader_abi_version")) is not int or
            source["shader_abi_version"] != (3 if instancing else 1) or source.get("address_units") != "words" or
            source.get("stage") != "fragment"):
        raise ValueError("host texture requires supported legacy fragment layout v2 with shader ABI 1")
    metadata = source["host_texture"]
    if type(metadata) is not dict:
        raise ValueError("host texture metadata must be an object")
    contract = host_texture_metadata_contract(metadata)
    _exact_object(metadata, contract, "host texture metadata")
    uniforms = source.get("uniforms")
    if type(uniforms) is not list:
        raise ValueError("ordinary uniform reflection must be a list")
    for entry in uniforms:
        if (type(entry) is not dict or type(entry.get("word_offset")) is not int or
                not 0 <= entry["word_offset"] < 65536):
            raise ValueError("ordinary uniform reflection has an invalid scalar offset")
        if entry["word_offset"] >= TEXTURE_BASE:
            raise ValueError("ordinary uniforms alias reserved texture storage block 15")
    if len(textures) != 1:
        raise ValueError("host texture requires exactly one captured texture")
    texture = textures[0]
    mipmaps = contract["version"] in (9, 10)
    fields = {"width", "height", "format", "data_hex", "sampler"} | ({"mipmaps"} if mipmaps else set())
    if type(texture) is not dict or set(texture) != fields:
        raise ValueError("captured texture must have exactly width, height, format, data_hex and sampler")
    width, height = texture["width"], texture["height"]
    if any(type(value) is not int or not 1 <= value <= 16 for value in (width, height)):
        raise ValueError("texture dimensions must be integers in [1, 16]")
    if type(texture["format"]) is not str or texture["format"] != "R8G8B8A8_UNORM":
        raise ValueError("host texture requires R8G8B8A8_UNORM")
    sampler = {**SAMPLER_STATE, "min_filter": contract["filter"], "mag_filter": contract["filter"],
               "wrap_s": contract.get("wrap_s", contract.get("wrap")),
               "wrap_t": contract.get("wrap_t", contract.get("wrap")),
               "mip_filter": "nearest" if mipmaps else "none"}
    _exact_object(texture["sampler"], sampler, "texture sampler")
    contents = _texture_bytes(texture["data_hex"], width, height)
    if mipmaps:
        levels = [(width, height, contents)]
        expected_width, expected_height = width, height
        expected_count = max(width, height).bit_length() - 1
        entries = texture["mipmaps"]
        if type(entries) is not list or len(entries) != expected_count:
            raise ValueError("host mipmaps require the complete bounded chain through 1x1")
        for level, entry in enumerate(entries, 1):
            expected_width, expected_height = max(1, expected_width // 2), max(1, expected_height // 2)
            expected = {"level": level, "width": expected_width, "height": expected_height}
            if (type(entry) is not dict or set(entry) != {*expected, "data_hex"} or
                    any(type(entry[key]) is not int or entry[key] != value for key, value in expected.items())):
                raise ValueError("host mipmap levels must have exact consecutive level numbers and dimensions")
            levels.append((expected_width, expected_height,
                           _texture_bytes(entry["data_hex"], expected_width, expected_height)))
        width, height, contents = levels[min(contract["lod"], len(levels) - 1)]
    payload = {TEXTURE_BASE: float_to_fp16_bits(width),
               TEXTURE_BASE + 1: float_to_fp16_bits(height),
               TEXTURE_BASE + 2: 0, TEXTURE_BASE + 3: 0}
    payload.update({TEXTURE_BASE + 4 + index: float_to_fp16_bits(channel / 255.0)
                    for index, channel in enumerate(contents)})
    return payload
