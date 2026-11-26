import copy
import itertools
import math
import struct
from types import SimpleNamespace

import pytest

from tools.host_texture_features import pack_texture_features, validate_texture_features_metadata


BASE = 0x110000
STRIDE = 16384
HEADER = 32
FORMATS = {
    "R8G8B8A8_UNORM": 4,
    "B8G8R8A8_UNORM": 4,
    "R8G8B8_UNORM": 3,
    "R8_UNORM": 1,
}
WRAPS = ("clamp-to-edge", "repeat", "mirrored-repeat")


def metadata(version=11, pairs=((0, 0),), lod_mode="explicit", wrap_s=WRAPS[0], wrap_t=WRAPS[0]):
    contract = {
        "version": version,
        "bindings": [{"texture": texture, "sampler": sampler} for texture, sampler in pairs],
        "max_bindings": 4,
        "max_samples": 32,
        "max_size": 32,
        "max_levels": 6,
        "per_binding_stride": 16384,
        "header_words": 32,
        "lod_mode": lod_mode,
        "wrap_s": wrap_s,
        "wrap_t": wrap_t,
        "filter": "nearest" if version == 11 else "linear",
        "mip_filter": "linear" if version in (13, 14) else "nearest",
    }
    if version == 14:
        contract["max_anisotropy"] = 4
    return contract


def layout(contract=None):
    return {
        "version": 4,
        "supported": True,
        "errors": [],
        "program_format": "legacy-mgcsim-v1",
        "shader_abi_version": 4,
        "address_units": "words",
        "stage": "fragment",
        "bases": {"input": 0x10000, "uniform": 0x20000, "vertex": 0x30000,
                  "fragment": 0x100000, "texture": BASE},
        "inputs": [],
        "outputs": [],
        "uniforms": [],
        "host_features": {"version": 1},
        "host_texture": metadata() if contract is None else copy.deepcopy(contract),
    }


def captured_texture(width=2, height=2, source_format="R8G8B8A8_UNORM", contents=None,
                     texture=0, sampler=0, contract=None):
    contract = metadata() if contract is None else contract
    channels = FORMATS[source_format]
    if contents is None:
        contents = bytes(index % 256 for index in range(width * height * channels))
    capture = {
        "binding": texture,
        "texture": texture,
        "sampler": sampler,
        "width": width,
        "height": height,
        "format": source_format,
        "data_hex": contents.hex(),
        "sampler_state": {
            "normalized_coords": True,
            "min_filter": contract["filter"],
            "mag_filter": contract["filter"],
            "mip_filter": contract["mip_filter"],
            "wrap_s": contract["wrap_s"],
            "wrap_t": contract["wrap_t"],
        },
        "mipmaps": [],
    }
    if contract["version"] == 14:
        capture["sampler_state"]["max_anisotropy"] = 4
    while (width, height) != (1, 1):
        width, height = max(1, width // 2), max(1, height // 2)
        level = len(capture["mipmaps"]) + 1
        capture["mipmaps"].append({
            "level": level, "width": width, "height": height, "format": source_format,
            "data_hex": bytes((index + level * 47) % 256
                              for index in range(width * height * channels)).hex(),
        })
    return capture


def fp16_word(channel):
    return struct.unpack("<H", struct.pack("<e", channel / 255.0))[0]


def canonical_channels(entry):
    source = bytes.fromhex(entry["data_hex"])
    channels = FORMATS[entry["format"]]
    converted = []
    for offset in range(0, len(source), channels):
        texel = list(source[offset:offset + channels])
        if entry["format"] == "B8G8R8A8_UNORM":
            texel[0], texel[2] = texel[2], texel[0]
        elif channels == 3:
            texel.append(255)
        elif channels == 1:
            texel.extend((0, 0, 255))
        converted.extend(texel)
    return converted


@pytest.mark.parametrize("version,lod_mode,wrap_s,wrap_t", tuple(itertools.product(
    (11, 12, 13), ("explicit", "implicit", "gradients", "mixed"), WRAPS, WRAPS)))
def test_exact_metadata_versions_modes_and_wraps(version, lod_mode, wrap_s, wrap_t):
    source = metadata(version, ((3, 0), (0, 2)), lod_mode, wrap_s, wrap_t)
    original = copy.deepcopy(source)
    contract = validate_texture_features_metadata(source)
    assert contract == source == original
    assert contract is not source
    assert contract["bindings"] is not source["bindings"]
    contract["bindings"][0]["sampler"] = 1
    assert source == original


@pytest.mark.parametrize("source_format,contents,expected", [
    ("R8G8B8A8_UNORM", bytes((1, 64, 128, 255)), (1, 64, 128, 255)),
    ("B8G8R8A8_UNORM", bytes((128, 64, 1, 255)), (1, 64, 128, 255)),
    ("R8G8B8_UNORM", bytes((1, 64, 128)), (1, 64, 128, 255)),
    ("R8_UNORM", bytes((128,)), (128, 0, 0, 255)),
])
def test_independent_channel_format_and_little_endian_words(source_format, contents, expected):
    capture = captured_texture(1, 1, source_format, contents)
    payload = pack_texture_features(metadata(), [capture])
    assert [payload[offset] for offset in range(4)] == [1, 1, 1, 32]
    words = [payload[32 + component] for component in range(4)]
    assert words == [fp16_word(channel) for channel in expected]
    assert b"".join(struct.pack("<I", word) for word in words) == b"".join(
        struct.pack("<e", channel / 255.0) + b"\x00\x00" for channel in expected)
    assert all(type(word) is int and 0 <= word <= 0xFFFF for word in payload.values())


@pytest.mark.parametrize("source_format", FORMATS)
def test_all_256_unsigned_source_values_round_to_finite_low_fp16(source_format):
    contents = bytes(channel for channel in range(256) for component in range(FORMATS[source_format]))
    capture = captured_texture(32, 8, source_format, contents)
    payload = pack_texture_features(metadata(), [capture])
    expected = canonical_channels(capture)
    assert [payload[HEADER + offset] for offset in range(1024)] == [fp16_word(value) for value in expected]
    assert all(math.isfinite(struct.unpack("<e", struct.pack("<H", payload[HEADER + offset]))[0])
               for offset in range(1024))


def test_raw32_header_reserved_zeroes_and_all_levels_after_header():
    capture = captured_texture(4, 2)
    payload = pack_texture_features(layout(), [capture])
    assert [payload[offset] for offset in range(10)] == [3, 4, 2, 32, 2, 1, 64, 1, 1, 72]
    assert all(payload[offset] == 0 for offset in range(10, 32))
    assert sorted(payload) == list(range(76))
    expected = [channel for entry in [capture, *capture["mipmaps"]] for channel in canonical_channels(entry)]
    assert [payload[offset] for offset in range(32, 76)] == [fp16_word(channel) for channel in expected]
    assert payload[1] != struct.unpack("<H", struct.pack("<e", 4.0))[0]


@pytest.mark.parametrize("width,height,dimensions", [
    (1, 1, [(1, 1)]),
    (1, 32, [(1, 32), (1, 16), (1, 8), (1, 4), (1, 2), (1, 1)]),
    (32, 1, [(32, 1), (16, 1), (8, 1), (4, 1), (2, 1), (1, 1)]),
    (3, 5, [(3, 5), (1, 2), (1, 1)]),
    (31, 17, [(31, 17), (15, 8), (7, 4), (3, 2), (1, 1)]),
    (32, 31, [(32, 31), (16, 15), (8, 7), (4, 3), (2, 1), (1, 1)]),
    (32, 32, [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]),
])
def test_complete_square_rectangular_and_npot_chains(width, height, dimensions):
    capture = captured_texture(width, height)
    payload = pack_texture_features(metadata(), [capture])
    assert payload[0] == len(dimensions)
    texel_offset = HEADER
    for level, (level_width, level_height) in enumerate(dimensions):
        record = 1 + level * 3
        assert [payload[record + component] for component in range(3)] == [level_width, level_height, texel_offset]
        texel_offset += level_width * level_height * 4
    assert len(payload) == texel_offset
    assert sorted(payload) == list(range(texel_offset))


def test_four_maximum_textures_fit_domain_without_aliasing_or_input_mutation():
    contract = metadata(13, ((3, 0), (0, 3), (2, 1), (1, 2)))
    captures = [captured_texture(32, 32, source_format, texture=pair["texture"], sampler=pair["sampler"],
                                 contract=contract)
                for pair, source_format in zip(reversed(contract["bindings"]), FORMATS)]
    source = layout(contract)
    original = copy.deepcopy((source, captures))
    payload = pack_texture_features(SimpleNamespace(source=source), captures)
    assert (source, captures) == original
    assert len(payload) == 4 * 5492
    assert max(payload) == 3 * STRIDE + 5491
    assert all(type(offset) is int and 0 <= offset < 65536 for offset in payload)
    for capture in captures:
        plane = capture["texture"] * STRIDE
        assert payload[plane] == 6
        assert [payload[plane + offset] for offset in (1, 2, 3)] == [32, 32, 32]
        expected = [channel for entry in [capture, *capture["mipmaps"]] for channel in canonical_channels(entry)]
        assert [payload[plane + HEADER + offset] for offset in range(5460)] == [fp16_word(value) for value in expected]
        assert plane + 5492 not in payload
    assert BASE not in payload


def test_sparse_planes_are_indexed_by_texture_not_sampler_or_list_order():
    contract = metadata(pairs=((3, 0), (0, 3)))
    captures = [captured_texture(1, 1, contents=bytes((7, 8, 9, 10)), texture=3, sampler=0),
                captured_texture(1, 1, contents=bytes((21, 22, 23, 24)), texture=0, sampler=3)]
    payload = pack_texture_features(contract, list(reversed(captures)))
    assert payload[3 * STRIDE + HEADER] == fp16_word(7)
    assert payload[HEADER] == fp16_word(21)
    assert set(payload) == set(range(36)) | set(range(3 * STRIDE, 3 * STRIDE + 36))


def test_distinct_textures_can_share_a_declared_sampler():
    contract = metadata(pairs=((0, 2), (3, 2)))
    captures = [captured_texture(1, 1, texture=0, sampler=2), captured_texture(1, 1, texture=3, sampler=2)]
    payload = pack_texture_features(contract, captures)
    assert payload[0] == payload[3 * STRIDE] == 1
    assert len(payload) == 72


@pytest.mark.parametrize("version,lod_mode", tuple(itertools.product(
    (11, 12, 13), ("explicit", "implicit", "gradients", "mixed"))))
def test_runtime_filter_and_lod_mode_do_not_select_or_drop_mip_inputs(version, lod_mode):
    contract = metadata(version, lod_mode=lod_mode)
    capture = captured_texture(8, 4, contract=contract)
    baseline = pack_texture_features(metadata(), [captured_texture(8, 4)])
    assert pack_texture_features(contract, [capture]) == baseline


@pytest.mark.parametrize("level", (0, 1, 2, 3))
def test_modifying_each_uploaded_level_changes_only_its_packed_texel_words(level):
    capture = captured_texture(8, 4)
    before = pack_texture_features(metadata(), [capture])
    entry = capture if level == 0 else capture["mipmaps"][level - 1]
    original = entry["data_hex"]
    entry["data_hex"] = "ff" + original[2:]
    after = pack_texture_features(metadata(), [capture])
    texel_offset = before[3 + 3 * level]
    assert {offset for offset in before if before[offset] != after[offset]} == {texel_offset}
    assert after[texel_offset] == 0x3C00


@pytest.mark.parametrize("field", list(metadata()))
def test_metadata_missing_fields_reject(field):
    contract = metadata()
    del contract[field]
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("extra", ("texture", "sampler", "storage_block", "lod", "max_anisotropy"))
def test_metadata_extra_fields_reject(extra):
    contract = metadata()
    contract[extra] = 0
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("field", ("version", "max_bindings", "max_samples", "max_size", "max_levels",
                                   "per_binding_stride", "header_words"))
@pytest.mark.parametrize("value", (None, True, False, 1.0, float("nan"), float("inf"), float("-inf"),
                                   "4", [], {}, -1, 0, 1 << 32))
def test_metadata_integer_fields_require_exact_bounded_ints(field, value):
    contract = metadata()
    contract[field] = value
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("version", (1, 9, 10, 14, 15))
def test_unknown_legacy_and_unextended_anisotropic_metadata_reject(version):
    contract = metadata(version)
    if version == 14:
        del contract["max_anisotropy"]
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("field,value", [
    ("max_bindings", 3), ("max_samples", 31), ("max_size", 16), ("max_levels", 5),
    ("per_binding_stride", 4096), ("header_words", 31), ("lod_mode", "constant"),
    ("lod_mode", None), ("lod_mode", True), ("wrap_s", "clamp"), ("wrap_t", "mirror"),
    ("wrap_s", None), ("wrap_t", 0), ("filter", "linear"), ("mip_filter", "linear"),
])
def test_metadata_inexact_limits_and_enums_reject(field, value):
    contract = metadata()
    contract[field] = value
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("version,field,value", [
    (12, "filter", "nearest"), (12, "mip_filter", "linear"),
    (13, "filter", "nearest"), (13, "mip_filter", "nearest"),
])
def test_filters_must_agree_with_their_version(version, field, value):
    contract = metadata(version)
    contract[field] = value
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("bindings", (None, {}, (), [], [{"texture": 0, "sampler": 0}] * 5))
def test_metadata_requires_one_to_four_explicit_used_pairs(bindings):
    contract = metadata()
    contract["bindings"] = bindings
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("entry", (None, [], {}, {"texture": 0}, {"sampler": 0},
                                   {"texture": 0, "sampler": 0, "binding": 0}))
def test_binding_pairs_require_exact_fields(entry):
    contract = metadata()
    contract["bindings"] = [entry]
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("field", ("texture", "sampler"))
@pytest.mark.parametrize("value", (None, True, False, 0.0, -1, 4, "0", float("nan"), float("inf")))
def test_binding_pair_indices_have_exact_int_types_and_bounds(field, value):
    contract = metadata()
    contract["bindings"][0][field] = value
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


def test_duplicate_used_pairs_reject():
    with pytest.raises(ValueError, match="duplicate"):
        validate_texture_features_metadata(metadata(pairs=((0, 1), (0, 1))))


@pytest.mark.parametrize("value", (None, [], (), "metadata"))
def test_metadata_and_layout_require_objects(value):
    with pytest.raises(ValueError):
        validate_texture_features_metadata(value)
    with pytest.raises(ValueError):
        pack_texture_features(value, [])


@pytest.mark.parametrize("textures", (None, {}, (), [], [None], [captured_texture()] * 2))
def test_capture_requires_an_exact_matching_explicit_list(textures):
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), textures)


@pytest.mark.parametrize("field", list(captured_texture()))
def test_capture_missing_fields_reject(field):
    capture = captured_texture()
    del capture[field]
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("extra", ("row_stride", "depth", "target", "data", "sampler_object", "storage_block"))
def test_capture_extra_fields_reject(extra):
    capture = captured_texture()
    capture[extra] = 0
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("field", ("binding", "texture", "sampler"))
@pytest.mark.parametrize("value", (None, True, False, 0.0, -1, 4, "0", float("nan"), float("inf")))
def test_capture_binding_types_and_bounds(field, value):
    capture = captured_texture()
    capture[field] = value
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


def test_capture_cannot_relabel_its_texture_plane_as_sampler_or_list_index():
    contract = metadata(pairs=((3, 0),))
    capture = captured_texture(texture=3, sampler=0)
    capture["binding"] = 0
    with pytest.raises(ValueError, match="binding"):
        pack_texture_features(contract, [capture])


def test_undeclared_sampler_pair_and_missing_used_texture_reject():
    with pytest.raises(ValueError, match="pair"):
        pack_texture_features(metadata(), [captured_texture(sampler=1)])
    with pytest.raises(ValueError, match="match"):
        pack_texture_features(metadata(pairs=((0, 0), (3, 1))), [captured_texture()])


def test_duplicate_capture_pairs_cannot_mask_a_missing_texture():
    contract = metadata(pairs=((0, 0), (3, 1)))
    captures = [captured_texture(), captured_texture()]
    original = copy.deepcopy((contract, captures))
    with pytest.raises(ValueError, match="duplicate captured texture/sampler pair"):
        pack_texture_features(contract, captures)
    assert (contract, captures) == original


@pytest.mark.parametrize("version,source_format,texture,reverse", tuple(itertools.product(
    (11, 12, 13), FORMATS, (0, 3), (False, True))))
def test_distinct_pairs_can_pack_identical_readonly_aliases_once(version, source_format, texture, reverse):
    contract = metadata(version, pairs=((texture, 0), (texture, 1)))
    source = layout(contract)
    capture = captured_texture(32, 32, source_format, texture=texture, sampler=0, contract=contract)
    alias = copy.deepcopy(capture)
    alias["sampler"] = 1
    alias["data_hex"] = alias["data_hex"].upper()
    for entry in alias["mipmaps"]:
        entry["data_hex"] = entry["data_hex"].upper()
    captures = [alias, capture] if reverse else [capture, alias]
    original = copy.deepcopy((source, captures))
    assert validate_texture_features_metadata(contract) == contract
    payload = pack_texture_features(SimpleNamespace(source=source), captures)
    assert payload == pack_texture_features(metadata(version, pairs=((texture, 0),)), [capture])
    assert sorted(payload) == list(range(texture * STRIDE, texture * STRIDE + 5492))
    assert (source, captures) == original


def test_sparse_readonly_aliases_keep_used_pair_coverage_separate_from_planes():
    contract = metadata(pairs=((3, 0), (0, 2), (3, 1), (0, 3)))
    captures = [captured_texture(4, 2, texture=pair["texture"], sampler=pair["sampler"])
                for pair in contract["bindings"]]
    original = copy.deepcopy((contract, captures))
    payload = pack_texture_features(contract, list(reversed(captures)))
    baseline = pack_texture_features(metadata(pairs=((3, 0), (0, 2))), captures[:2])
    assert payload == baseline
    assert set(payload) == set(range(76)) | set(range(3 * STRIDE, 3 * STRIDE + 76))
    assert (contract, captures) == original


@pytest.mark.parametrize("width,height,source_format", [
    (4, 2, "R8G8B8A8_UNORM"),
    (2, 4, "R8G8B8A8_UNORM"),
    (2, 2, "B8G8R8A8_UNORM"),
    (2, 2, "R8G8B8_UNORM"),
    (2, 2, "R8_UNORM"),
])
def test_readonly_alias_dimensions_and_source_formats_must_match(width, height, source_format):
    contract = metadata(pairs=((0, 0), (0, 1)))
    captures = [captured_texture(), captured_texture(width, height, source_format, sampler=1)]
    original = copy.deepcopy((contract, captures))
    with pytest.raises(ValueError, match="alias"):
        pack_texture_features(contract, captures)
    assert (contract, captures) == original


def test_canonical_word_equivalence_cannot_hide_conflicting_alias_source_formats():
    contract = metadata(pairs=((0, 0), (0, 1)))
    captures = [captured_texture(1, 1, contents=bytes((0, 0, 0, 255))),
                captured_texture(1, 1, "R8G8B8_UNORM", contents=bytes((0, 0, 0)), sampler=1)]
    original = copy.deepcopy((contract, captures))
    assert pack_texture_features(metadata(), captures[:1]) == pack_texture_features(
        metadata(pairs=((0, 1),)), captures[1:])
    with pytest.raises(ValueError, match="alias"):
        pack_texture_features(contract, captures)
    assert (contract, captures) == original


@pytest.mark.parametrize("level,reverse", tuple(itertools.product(range(6), (False, True))))
def test_every_readonly_alias_level_requires_byte_exact_contents(level, reverse):
    contract = metadata(pairs=((3, 0), (3, 1)))
    capture = captured_texture(32, 32, texture=3, sampler=0)
    alias = copy.deepcopy(capture)
    alias["sampler"] = 1
    entry = alias if level == 0 else alias["mipmaps"][level - 1]
    entry["data_hex"] = "ff" + entry["data_hex"][2:]
    captures = [alias, capture] if reverse else [capture, alias]
    original = copy.deepcopy((contract, captures))
    with pytest.raises(ValueError, match="alias"):
        pack_texture_features(contract, captures)
    assert (contract, captures) == original


@pytest.mark.parametrize("field,value", [
    ("normalized_coords", 1), ("min_filter", "linear"), ("mag_filter", "linear"),
    ("mip_filter", "linear"), ("wrap_s", "repeat"), ("wrap_t", "mirrored-repeat"),
])
def test_identical_readonly_aliases_still_validate_each_sampler_contract(field, value):
    contract = metadata(pairs=((0, 0), (0, 1)))
    captures = [captured_texture(), captured_texture(sampler=1)]
    captures[1]["sampler_state"][field] = value
    original = copy.deepcopy((contract, captures))
    with pytest.raises(ValueError, match="sampler"):
        pack_texture_features(contract, captures)
    assert (contract, captures) == original


@pytest.mark.parametrize("duplicate_sampler", (0, 1))
def test_duplicate_readonly_alias_pair_cannot_mask_an_omitted_pair(duplicate_sampler):
    contract = metadata(pairs=((0, 0), (0, 1)))
    captures = [captured_texture(sampler=duplicate_sampler), captured_texture(sampler=duplicate_sampler)]
    original = copy.deepcopy((contract, captures))
    with pytest.raises(ValueError, match="duplicate captured texture/sampler pair"):
        pack_texture_features(contract, captures)
    assert (contract, captures) == original


def test_readonly_alias_cannot_omit_a_used_pair_even_when_the_plane_is_uploaded():
    contract = metadata(pairs=((0, 0), (0, 1)))
    captures = [captured_texture()]
    original = copy.deepcopy((contract, captures))
    with pytest.raises(ValueError, match="match"):
        pack_texture_features(contract, captures)
    assert (contract, captures) == original


@pytest.mark.parametrize("field", ("width", "height"))
@pytest.mark.parametrize("value", (0, -1, 33, True, False, 2.0, "2", None, float("nan"),
                                   float("inf"), float("-inf"), 1 << 32))
def test_base_dimensions_require_exact_ints_within_32(field, value):
    capture = captured_texture()
    capture[field] = value
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("source_format", (None, 4, "RGBA8", "BGRA8", "RGB8", "R8", "R8G8B8A8_SNORM",
                                           "R16G16B16A16_FLOAT", "R32_FLOAT", "R8G8_UNORM"))
def test_only_canonical_typed_unorm_source_formats_are_accepted(source_format):
    capture = captured_texture()
    capture["format"] = source_format
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("encoded", (None, b"00000000", [0.0, float("nan")], 0, "", "000000", "0000000000",
                                     "gg000000", "0x000000", "00 00000", "0000000\n"))
def test_data_hex_rejects_truncation_padding_whitespace_and_nonhex(encoded):
    capture = captured_texture(1, 1)
    capture["data_hex"] = encoded
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("source_format", FORMATS)
def test_typed_source_length_is_checked_before_canonical_expansion(source_format):
    capture = captured_texture(1, 1, source_format)
    capture["data_hex"] += "00"
    with pytest.raises(ValueError, match="bytes"):
        pack_texture_features(metadata(), [capture])


def test_hex_is_case_insensitive_and_never_decoded_as_float32():
    capture = captured_texture(1, 1, contents=bytes((0, 0, 192, 127)))
    capture["data_hex"] = capture["data_hex"].upper()
    payload = pack_texture_features(metadata(), [capture])
    assert [payload[32 + component] for component in range(4)] == [fp16_word(value) for value in (0, 0, 192, 127)]


@pytest.mark.parametrize("field,value", [
    ("normalized_coords", False), ("normalized_coords", 1), ("normalized_coords", "true"),
    ("min_filter", "linear"), ("mag_filter", "linear"), ("mip_filter", "none"),
    ("mip_filter", "linear"), ("wrap_s", "repeat"), ("wrap_t", "mirrored-repeat"),
    ("min_filter", True), ("mag_filter", None),
])
def test_sampler_state_must_exactly_agree_with_shader(field, value):
    capture = captured_texture()
    capture["sampler_state"][field] = value
    with pytest.raises(ValueError, match="sampler"):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("field", list(captured_texture()["sampler_state"]))
def test_sampler_state_missing_fields_reject(field):
    capture = captured_texture()
    del capture["sampler_state"][field]
    with pytest.raises(ValueError, match="sampler"):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("state", (None, [], {}, {"normalized_coords": True}))
def test_sampler_state_requires_an_exact_object(state):
    capture = captured_texture()
    capture["sampler_state"] = state
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("field,value", (("lod_bias", float("nan")), ("min_lod", float("inf")),
                                         ("max_lod", float("-inf")), ("max_anisotropy", 4), ("compare_mode", "none")))
def test_uncontracted_sampler_state_cannot_bypass_validation(field, value):
    capture = captured_texture()
    capture["sampler_state"][field] = value
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("version,wrap_s,wrap_t", tuple(itertools.product((11, 12, 13), WRAPS, WRAPS)))
def test_each_axis_sampler_wrap_and_filter_agreement(version, wrap_s, wrap_t):
    contract = metadata(version, wrap_s=wrap_s, wrap_t=wrap_t)
    capture = captured_texture(contract=contract)
    assert pack_texture_features(layout(contract), [capture])[0] == 2
    capture["sampler_state"]["wrap_s"] = WRAPS[(WRAPS.index(wrap_s) + 1) % 3]
    with pytest.raises(ValueError, match="sampler"):
        pack_texture_features(contract, [capture])


@pytest.mark.parametrize("mipmaps", (None, {}, (), []))
def test_complete_chain_requires_explicit_mipmap_list(mipmaps):
    capture = captured_texture(4, 4)
    capture["mipmaps"] = mipmaps
    with pytest.raises(ValueError, match="chain"):
        pack_texture_features(metadata(), [capture])


def test_one_by_one_still_requires_an_explicit_empty_mipmap_list():
    capture = captured_texture(1, 1)
    capture["mipmaps"] = [captured_texture(1, 1)["mipmaps"]]
    with pytest.raises(ValueError, match="chain"):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("field", ("level", "width", "height", "format", "data_hex"))
def test_mipmap_missing_fields_reject_even_last_level(field):
    capture = captured_texture(32, 32)
    del capture["mipmaps"][-1][field]
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("extra", ("row_stride", "binding", "sampler", "offset"))
def test_mipmap_extra_fields_reject(extra):
    capture = captured_texture()
    capture["mipmaps"][0][extra] = 0
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("field,value", [
    ("level", 0), ("level", 2), ("level", True), ("level", 1.0), ("level", float("nan")),
    ("width", 2), ("height", 2), ("width", True), ("height", 1.0), ("height", float("inf")),
    ("format", "R8_UNORM"), ("format", "R32_FLOAT"), ("data_hex", "000000"), ("data_hex", "gg000000"),
])
def test_mipmap_level_index_dimension_format_and_bytes_are_all_validated(field, value):
    capture = captured_texture()
    capture["mipmaps"][0][field] = value
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


@pytest.mark.parametrize("level", range(1, 6))
def test_every_level_in_maximum_chain_is_validated_even_if_runtime_lod_would_skip_it(level):
    capture = captured_texture(32, 32)
    capture["mipmaps"][level - 1]["data_hex"] = capture["mipmaps"][level - 1]["data_hex"][:-2]
    original = copy.deepcopy(capture)
    with pytest.raises(ValueError, match="bytes"):
        pack_texture_features(metadata(), [capture])
    assert capture == original


@pytest.mark.parametrize("mutation", ("missing", "extra", "reverse", "skip", "repeat"))
def test_incomplete_or_reordered_mipmap_chains_reject(mutation):
    capture = captured_texture(8, 4)
    if mutation == "missing":
        capture["mipmaps"].pop()
    elif mutation == "extra":
        capture["mipmaps"].append(copy.deepcopy(capture["mipmaps"][-1]))
    elif mutation == "reverse":
        capture["mipmaps"].reverse()
    elif mutation == "skip":
        capture["mipmaps"][1]["level"] = 3
    else:
        capture["mipmaps"][1] = copy.deepcopy(capture["mipmaps"][0])
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [capture])


def test_all_resources_are_validated_before_returning_a_payload():
    contract = metadata(pairs=((0, 0), (3, 1)))
    captures = [captured_texture(32, 32), captured_texture(32, 32, texture=3, sampler=1)]
    captures[-1]["mipmaps"][-1]["data_hex"] = "000000"
    original = copy.deepcopy(captures)
    with pytest.raises(ValueError):
        pack_texture_features(contract, captures)
    assert captures == original


@pytest.mark.parametrize("field,value", [
    ("version", 1), ("version", 2), ("version", 3), ("version", True), ("version", 4.0),
    ("shader_abi_version", 1), ("shader_abi_version", 3), ("shader_abi_version", True),
    ("shader_abi_version", 4.0), ("supported", False), ("supported", 1),
    ("program_format", "fp16-hardware-v1"), ("program_format", None),
    ("address_units", "bytes"), ("stage", "vertex"), ("stage", None),
    ("host_features", None), ("host_features", {}), ("host_features", {"version": True}),
    ("host_features", {"version": 2}),
])
def test_full_layout_requires_supported_word_addressed_legacy_abi4_fragment(field, value):
    source = layout()
    source[field] = value
    with pytest.raises(ValueError):
        pack_texture_features(source, [captured_texture()])


@pytest.mark.parametrize("field", ("version", "shader_abi_version", "supported", "program_format",
                                   "address_units", "stage", "host_features", "bases", "uniforms"))
def test_layout_required_gate_fields_reject_when_missing(field):
    source = layout()
    del source[field]
    with pytest.raises(ValueError):
        pack_texture_features(source, [captured_texture()])


@pytest.mark.parametrize("field,value", [
    ("texture", BASE * 4), ("texture", BASE + 1), ("texture", True), ("texture", float("nan")),
    ("input", BASE), ("fragment", BASE - 65535), ("uniform", 0x10001),
    ("vertex", -1), ("vertex", 1 << 32), ("vertex", 0xFFFF0001), ("input", 0x10000 * 1.0),
])
def test_abi_bases_reject_wrong_domain_units_aliases_bounds_and_types(field, value):
    source = layout()
    source["bases"][field] = value
    with pytest.raises(ValueError):
        pack_texture_features(source, [captured_texture()])


@pytest.mark.parametrize("bases", (None, [], {}, {"texture": BASE}, {**layout()["bases"], "other": 0}))
def test_layout_requires_exact_five_bases(bases):
    source = layout()
    source["bases"] = bases
    with pytest.raises(ValueError):
        pack_texture_features(source, [captured_texture()])


def test_abi4_has_no_old_ubo15_reservation_and_other_domains_can_end_at_uint32_limit():
    source = layout()
    source["bases"]["vertex"] = 0xFFFF0000
    source["uniforms"] = [{"word_offset": 61440}, {"word_offset": 65535}]
    assert pack_texture_features(source, [captured_texture()]) == pack_texture_features(metadata(), [captured_texture()])


@pytest.mark.parametrize("uniforms", (None, {}, [None], [{}], [{"word_offset": True}],
                                      [{"word_offset": 1.0}], [{"word_offset": float("inf")}],
                                      [{"word_offset": -1}], [{"word_offset": 65536}]))
def test_layout_uniform_reflection_cannot_alias_other_domains(uniforms):
    source = layout()
    source["uniforms"] = uniforms
    with pytest.raises(ValueError):
        pack_texture_features(source, [captured_texture()])


@pytest.mark.parametrize("stage", ("vertex", "fragment"))
def test_supported_abi4_layout_without_textures_returns_empty_only_for_empty_captures(stage):
    source = layout()
    source["stage"] = stage
    del source["host_texture"]
    assert pack_texture_features(source, []) == {}
    with pytest.raises(ValueError):
        pack_texture_features(source, [captured_texture()])


def test_exact_metadata_type_rules_reject_dictionary_and_integer_subclasses():
    class CustomDict(dict):
        pass

    class CustomInt(int):
        pass

    with pytest.raises(ValueError):
        validate_texture_features_metadata(CustomDict(metadata()))
    contract = metadata()
    contract["max_size"] = CustomInt(32)
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)
    with pytest.raises(ValueError):
        pack_texture_features(metadata(), [CustomDict(captured_texture())])


def test_v14_exact_metadata_and_raw32_input_snapshot():
    contract = metadata(14, pairs=((3, 0),), lod_mode="mixed", wrap_s="repeat", wrap_t="mirrored-repeat")
    expected_contract = {
        "version": 14,
        "bindings": [{"texture": 3, "sampler": 0}],
        "max_bindings": 4,
        "max_samples": 32,
        "max_size": 32,
        "max_levels": 6,
        "per_binding_stride": 16384,
        "header_words": 32,
        "lod_mode": "mixed",
        "wrap_s": "repeat",
        "wrap_t": "mirrored-repeat",
        "filter": "linear",
        "mip_filter": "linear",
        "max_anisotropy": 4,
    }
    capture = captured_texture(1, 1, "R8G8B8_UNORM", contents=bytes((0, 255, 0)),
                               texture=3, sampler=0, contract=contract)
    source = layout(contract)
    original = copy.deepcopy((source, capture))
    validated = validate_texture_features_metadata(contract)
    assert validated == expected_contract
    assert capture["sampler_state"] == {
        "normalized_coords": True, "min_filter": "linear", "mag_filter": "linear",
        "mip_filter": "linear", "wrap_s": "repeat", "wrap_t": "mirrored-repeat", "max_anisotropy": 4,
    }
    expected_words = {offset: 0 for offset in range(36)}
    expected_words.update({0: 1, 1: 1, 2: 1, 3: 32, 33: 0x3C00, 35: 0x3C00})
    assert pack_texture_features(source, [capture]) == {
        3 * STRIDE + offset: word for offset, word in expected_words.items()
    }
    validated["bindings"][0]["sampler"] = 1
    validated["max_anisotropy"] = 3
    assert (source, capture) == original
    assert contract == expected_contract


@pytest.mark.parametrize("lod_mode,wrap_s,wrap_t", tuple(itertools.product(
    ("explicit", "implicit", "gradients", "mixed"), WRAPS, WRAPS)))
def test_v14_lod_modes_wraps_and_source_sample_limit_preserve_input_packing(lod_mode, wrap_s, wrap_t):
    contract = metadata(14, lod_mode=lod_mode, wrap_s=wrap_s, wrap_t=wrap_t)
    capture = captured_texture(8, 4, contract=contract)
    original = copy.deepcopy((contract, capture))
    assert validate_texture_features_metadata(contract) == contract
    baseline_contract = metadata(13, lod_mode=lod_mode, wrap_s=wrap_s, wrap_t=wrap_t)
    baseline_capture = captured_texture(8, 4, contract=baseline_contract)
    assert pack_texture_features(contract, [capture]) == pack_texture_features(baseline_contract, [baseline_capture])
    assert contract["max_samples"] == 32
    assert (contract, capture) == original


@pytest.mark.parametrize("source_format,texture,reverse,use_layout", tuple(itertools.product(
    FORMATS, (0, 3), (False, True), (False, True))))
def test_v14_complete_maximum_chain_readonly_aliases_match_v13_inputs(source_format, texture, reverse, use_layout):
    contract = metadata(14, pairs=((texture, 0), (texture, 1)))
    capture = captured_texture(32, 32, source_format, texture=texture, sampler=0, contract=contract)
    alias = copy.deepcopy(capture)
    alias["sampler"] = 1
    alias["data_hex"] = alias["data_hex"].upper()
    for entry in alias["mipmaps"]:
        entry["data_hex"] = entry["data_hex"].upper()
    captures = [alias, capture] if reverse else [capture, alias]
    source = SimpleNamespace(source=layout(contract)) if use_layout else contract
    original = copy.deepcopy((source, captures))
    payload = pack_texture_features(source, captures)
    baseline_contract = metadata(13, pairs=((texture, 0),))
    baseline_capture = captured_texture(32, 32, source_format, texture=texture, sampler=0, contract=baseline_contract)
    assert payload == pack_texture_features(baseline_contract, [baseline_capture])
    assert sorted(payload) == list(range(texture * STRIDE, texture * STRIDE + 5492))
    assert (source, captures) == original


@pytest.mark.parametrize("target", ("metadata", "sampler_state"))
@pytest.mark.parametrize("value", (None, True, False, 0, 1, 2, 3, 5, 8, -1, 1 << 32, 4.0, "4",
                                   float("nan"), float("inf"), float("-inf"), [], {}))
def test_v14_anisotropy_requires_exact_integer_four(target, value):
    contract = metadata(14)
    capture = captured_texture(contract=contract)
    if target == "metadata":
        contract["max_anisotropy"] = value
        with pytest.raises(ValueError):
            validate_texture_features_metadata(contract)
    else:
        capture["sampler_state"]["max_anisotropy"] = value
    original = copy.deepcopy((contract, capture))
    with pytest.raises(ValueError):
        pack_texture_features(contract, [capture])
    assert (contract, capture) == original


@pytest.mark.parametrize("target", ("metadata", "sampler_state"))
def test_v14_anisotropy_rejects_integer_subclasses(target):
    class CustomInt(int):
        pass

    contract = metadata(14)
    capture = captured_texture(contract=contract)
    if target == "metadata":
        contract["max_anisotropy"] = CustomInt(4)
    else:
        capture["sampler_state"]["max_anisotropy"] = CustomInt(4)
    original = copy.deepcopy((contract, capture))
    with pytest.raises(ValueError):
        pack_texture_features(contract, [capture])
    assert (contract, capture) == original


@pytest.mark.parametrize("target", ("metadata", "sampler_state"))
def test_v14_requires_its_extra_anisotropy_field(target):
    contract = metadata(14)
    capture = captured_texture(contract=contract)
    if target == "metadata":
        del contract["max_anisotropy"]
        with pytest.raises(ValueError):
            validate_texture_features_metadata(contract)
    else:
        del capture["sampler_state"]["max_anisotropy"]
    original = copy.deepcopy((contract, capture))
    with pytest.raises(ValueError):
        pack_texture_features(contract, [capture])
    assert (contract, capture) == original


@pytest.mark.parametrize("version,target", tuple(itertools.product((11, 12, 13), ("metadata", "sampler_state"))))
def test_old_versions_still_reject_anisotropy_extras(version, target):
    contract = metadata(version)
    capture = captured_texture(contract=contract)
    if target == "metadata":
        contract["max_anisotropy"] = 4
        with pytest.raises(ValueError):
            validate_texture_features_metadata(contract)
    else:
        capture["sampler_state"]["max_anisotropy"] = 4
    original = copy.deepcopy((contract, capture))
    with pytest.raises(ValueError):
        pack_texture_features(contract, [capture])
    assert (contract, capture) == original


@pytest.mark.parametrize("field,value", [
    ("filter", "nearest"), ("filter", "anisotropic"), ("mip_filter", "nearest"),
    ("max_samples", 128), ("max_samples", 8), ("max_size", 64), ("max_levels", 7),
    ("per_binding_stride", 4096), ("header_words", 64), ("lod_mode", "anisotropic"),
    ("version", 14.0), ("version", "14"), ("version", True), ("version", 15),
    ("tap_count", 4), ("anisotropy_mode", "ewa"),
])
def test_v14_cannot_change_other_contract_fields_or_count_expanded_taps(field, value):
    contract = metadata(14)
    contract[field] = value
    with pytest.raises(ValueError):
        validate_texture_features_metadata(contract)


@pytest.mark.parametrize("field,value", [
    ("normalized_coords", 1), ("min_filter", "nearest"), ("mag_filter", "nearest"),
    ("mip_filter", "nearest"), ("wrap_s", "repeat"), ("wrap_t", "mirrored-repeat"),
    ("tap_count", 4), ("lod_bias", 0), ("anisotropy_mode", "ewa"),
])
def test_v14_sampler_state_remains_exact_and_normalized(field, value):
    contract = metadata(14)
    capture = captured_texture(contract=contract)
    capture["sampler_state"][field] = value
    original = copy.deepcopy((contract, capture))
    with pytest.raises(ValueError, match="sampler"):
        pack_texture_features(contract, [capture])
    assert (contract, capture) == original


@pytest.mark.parametrize("mutation", ("conflict", "truncated", "sampler", "duplicate", "omitted"))
def test_v14_aliases_preserve_byte_sampler_and_pair_coverage_guards(mutation):
    contract = metadata(14, pairs=((3, 0), (3, 1)))
    captures = [captured_texture(32, 32, texture=3, sampler=0, contract=contract),
                captured_texture(32, 32, texture=3, sampler=1, contract=contract)]
    if mutation == "conflict":
        entry = captures[1]["mipmaps"][-1]
        entry["data_hex"] = "ff" + entry["data_hex"][2:]
    elif mutation == "truncated":
        entry = captures[1]["mipmaps"][-1]
        entry["data_hex"] = entry["data_hex"][:-2]
    elif mutation == "sampler":
        captures[1]["sampler_state"]["max_anisotropy"] = 3
    elif mutation == "duplicate":
        captures[1]["sampler"] = 0
    else:
        captures.pop()
    original = copy.deepcopy((contract, captures))
    with pytest.raises(ValueError):
        pack_texture_features(contract, captures)
    assert (contract, captures) == original


@pytest.mark.parametrize("field,value", [
    ("version", 1), ("version", 2), ("version", 3), ("version", True),
    ("shader_abi_version", 1), ("shader_abi_version", 2), ("shader_abi_version", 3),
    ("shader_abi_version", True), ("program_format", "fp16-hardware-v1"),
    ("address_units", "bytes"), ("stage", "vertex"), ("supported", False),
    ("host_features", None), ("host_features", {"version": 2}),
])
def test_v14_layout_remains_abi4_legacy_host_only(field, value):
    contract = metadata(14)
    capture = captured_texture(contract=contract)
    source = layout(contract)
    source[field] = value
    original = copy.deepcopy((source, capture))
    with pytest.raises(ValueError):
        pack_texture_features(source, [capture])
    assert (source, capture) == original


@pytest.mark.parametrize("texture_base", (BASE + 1, BASE + 65536, BASE * 4))
def test_v14_keeps_the_fixed_word_addressed_texture_base(texture_base):
    contract = metadata(14)
    source = layout(contract)
    source["bases"]["texture"] = texture_base
    with pytest.raises(ValueError, match="texture base"):
        pack_texture_features(source, [captured_texture(contract=contract)])
