import copy
from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
from itertools import product
import json
import math
import random

import pytest

from tools.host_pipeline_state import (
    BLEND_EQUATIONS, BLEND_FACTORS, COLOR_FORMATS, COMPARISONS, DEPTH_STENCIL_FORMATS,
    STATE_VERSION, STENCIL_OPS, PipelineState, StencilFace, blend_color, commit_fragment,
    default_state, pack_color, pack_depth_stencil, unpack_color, unpack_depth_stencil,
    validate_state,
)


FACTORS = (
    "zero", "one", "src-color", "one-minus-src-color", "dst-color", "one-minus-dst-color",
    "src-alpha", "one-minus-src-alpha", "dst-alpha", "one-minus-dst-alpha", "constant-color",
    "one-minus-constant-color", "constant-alpha", "one-minus-constant-alpha", "src-alpha-saturate",
)
EQUATIONS = ("add", "subtract", "reverse-subtract", "min", "max")
COMPARES = ("never", "less", "equal", "lequal", "greater", "notequal", "gequal", "always")
OPERATIONS = (
    "keep", "zero", "replace", "increment-clamp", "decrement-clamp", "invert",
    "increment-wrap", "decrement-wrap",
)
MAX_DEPTH = {"z16": 65535, "z24s8": 16777215}


def oracle_round(value, maximum):
    bounded = min(Fraction(1), max(Fraction(0), Fraction(value))) * maximum
    quotient, remainder = divmod(bounded.numerator, bounded.denominator)
    twice_remainder = remainder * 2
    return quotient + int(twice_remainder > bounded.denominator or
                          (twice_remainder == bounded.denominator and quotient % 2 == 1))


def oracle_decode_color(color_format, data):
    if color_format == "rgb565":
        word = data[0] + data[1] * 256
        return (Fraction(word // 2048, 31), Fraction((word // 32) % 64, 63),
                Fraction(word % 32, 31), Fraction(1))
    if color_format == "r8":
        return (Fraction(data[0], 255), Fraction(0), Fraction(0), Fraction(1))
    order = (2, 1, 0, 3) if color_format == "bgra8" else (0, 1, 2, 3)
    return tuple(Fraction(data[channel], 255) for channel in order)


def oracle_pack_color(color_format, rgba):
    if color_format == "rgb565":
        word = oracle_round(rgba[0], 31) * 2048 + oracle_round(rgba[1], 63) * 32 + oracle_round(rgba[2], 31)
        return bytes((word % 256, word // 256))
    if color_format == "r8":
        return bytes((oracle_round(rgba[0], 255),))
    order = (2, 1, 0, 3) if color_format == "bgra8" else (0, 1, 2, 3)
    return bytes(oracle_round(rgba[channel], 255) for channel in order)


def oracle_blend(state, source, destination_color):
    source = tuple(min(Fraction(1), max(Fraction(0), Fraction(channel))) for channel in source)
    destination = oracle_decode_color(state["framebuffer"]["color_format"], destination_color)
    constant = tuple(Fraction(channel) for channel in state["blend"]["constant"])
    factors = {
        "zero": (Fraction(0),) * 4,
        "one": (Fraction(1),) * 4,
        "src-color": source,
        "dst-color": destination,
        "src-alpha": (source[3],) * 4,
        "dst-alpha": (destination[3],) * 4,
        "constant-color": constant,
        "constant-alpha": (constant[3],) * 4,
        "src-alpha-saturate": (min(source[3], 1 - destination[3]),) * 3 + (Fraction(1),),
    }
    for name in ("src-color", "dst-color", "src-alpha", "dst-alpha", "constant-color", "constant-alpha"):
        factors["one-minus-" + name] = tuple(1 - channel for channel in factors[name])
    result = []
    for channel in range(4):
        suffix = "alpha" if channel == 3 else "rgb"
        equation = state["blend"][suffix + "_equation"]
        first = source[channel] * factors[state["blend"]["src_" + suffix]][channel]
        second = destination[channel] * factors[state["blend"]["dst_" + suffix]][channel]
        values = {"add": first + second, "subtract": first - second, "reverse-subtract": second - first,
                  "min": min(source[channel], destination[channel]), "max": max(source[channel], destination[channel])}
        value = values[equation] if state["blend"]["enabled"] else source[channel]
        result.append(value if state["color_mask"][channel] else destination[channel])
    return oracle_pack_color(state["framebuffer"]["color_format"], result)


def oracle_compare(comparison, incoming, stored):
    if comparison == "never":
        return False
    if comparison == "always":
        return True
    relation = (incoming > stored) - (incoming < stored)
    accepted = {"less": {-1}, "equal": {0}, "lequal": {-1, 0}, "greater": {1},
                "notequal": {-1, 1}, "gequal": {0, 1}}
    return relation in accepted[comparison]


def oracle_stencil(operation, stored, reference, mask):
    values = {"keep": stored, "zero": 0, "replace": reference,
              "increment-clamp": min(255, stored + 1), "decrement-clamp": max(0, stored - 1),
              "invert": 255 - stored, "increment-wrap": (stored + 1) % 256,
              "decrement-wrap": (stored + 255) % 256}
    updated = values[operation]
    result = 0
    for bit in range(8):
        value = updated if mask & (1 << bit) else stored
        result += ((value >> bit) % 2) << bit
    return result


def oracle_fragment(state, source, color, depth_stencil, source_depth, *, front_facing=True, killed=False):
    color_format = state["framebuffer"]["color_format"]
    depth_format = state["framebuffer"]["depth_stencil_format"]
    original_depth = None if depth_stencil is None else bytes(depth_stencil)
    result = {"color": bytes(color), "depth_stencil": original_depth, "outcome": "killed",
              "color_written": False, "depth_written": False, "stencil_written": False}
    if killed:
        return result
    word = 0 if depth_stencil is None else sum(value * 256 ** index for index, value in enumerate(depth_stencil))
    stored_depth = word % 16777216 if depth_format == "z24s8" else word
    stored_stencil = word // 16777216 if depth_format == "z24s8" else 0
    incoming_depth = 0 if depth_format == "none" else oracle_round(source_depth, MAX_DEPTH[depth_format])
    face = state["stencil"]["front" if front_facing else "back"]
    stencil_pass = not state["stencil"]["enabled"] or oracle_compare(
        face["comparison"], face["reference"] & face["read_mask"], stored_stencil & face["read_mask"],
    )
    depth_pass = not state["depth"]["enabled"] or oracle_compare(
        state["depth"]["comparison"], incoming_depth, stored_depth,
    )
    result["outcome"] = "stencil-fail" if not stencil_pass else "depth-fail" if not depth_pass else "passed"
    operation = face["fail"] if not stencil_pass else face["zfail"] if not depth_pass else face["zpass"]
    result["stencil_written"] = state["stencil"]["enabled"] and operation != "keep" and face["write_mask"] != 0
    if result["stencil_written"]:
        stored_stencil = oracle_stencil(operation, stored_stencil, face["reference"], face["write_mask"])
    accepted = result["outcome"] == "passed"
    result["depth_written"] = accepted and state["depth"]["enabled"] and state["depth"]["write"]
    if result["depth_written"]:
        stored_depth = incoming_depth
    if result["depth_written"] or result["stencil_written"]:
        updated_word = stored_depth + stored_stencil * 16777216 if depth_format == "z24s8" else stored_depth
        size = 4 if depth_format == "z24s8" else 2
        result["depth_stencil"] = bytes((updated_word // 256 ** index) % 256 for index in range(size))
    present = 1 if color_format == "r8" else 3 if color_format == "rgb565" else 4
    result["color_written"] = accepted and any(state["color_mask"][:present])
    if result["color_written"]:
        result["color"] = oracle_blend(state, source, color)
    return result


def assert_commit(state, source, color, attachment, source_depth, *, front_facing=True, killed=False):
    before = (copy.deepcopy(state), tuple(source), bytes(color), None if attachment is None else bytes(attachment))
    result = commit_fragment(state, source, color, source_depth=source_depth,
                             destination_depth_stencil=attachment, front_facing=front_facing, killed=killed)
    expected = oracle_fragment(state, source, color, attachment, source_depth, front_facing=front_facing, killed=killed)
    assert asdict(result) == expected
    assert result.passed == (expected["outcome"] == "passed")
    assert result.killed == killed
    assert (state, tuple(source), bytes(color), None if attachment is None else bytes(attachment)) == before
    return result


def test_exported_contract_and_exact_json_roundtrip():
    assert STATE_VERSION == 1
    assert BLEND_FACTORS == FACTORS
    assert BLEND_EQUATIONS == EQUATIONS
    assert COMPARISONS == COMPARES
    assert STENCIL_OPS == OPERATIONS
    assert COLOR_FORMATS == ("rgba8", "bgra8", "rgb565", "r8")
    assert DEPTH_STENCIL_FORMATS == ("none", "z16", "z24s8")
    original = default_state(blend_mode="source-over", color_format="bgra8", depth_stencil_format="z24s8")
    original["blend"]["constant"] = [0, 0.25, 1.0, 0.75]
    original["stencil"]["front"]["reference"] = 127
    original["stencil"]["back"]["reference"] = 255
    snapshot = copy.deepcopy(original)
    state = validate_state(original)
    assert type(state) is PipelineState
    assert validate_state(state) is state
    assert state.to_dict() == snapshot
    assert json.dumps(state.to_dict(), allow_nan=False) == json.dumps(snapshot, allow_nan=False)
    assert PipelineState.from_dict(json.loads(json.dumps(snapshot))) == state
    original["blend"]["constant"][0] = 1
    original["color_mask"][0] = False
    original["stencil"]["front"]["reference"] = 0
    exported = state.to_dict()
    exported["stencil"]["back"]["reference"] = 0
    assert state.to_dict() == snapshot
    with pytest.raises(FrozenInstanceError):
        state.depth_write = False
    with pytest.raises(FrozenInstanceError):
        state.stencil_front.reference = 0
    first, second = default_state(), default_state()
    first["stencil"]["front"]["reference"] = 22
    assert first["stencil"]["back"]["reference"] == second["stencil"]["front"]["reference"] == 0


class StringSubclass(str):
    pass


class DictSubclass(dict):
    pass


class StateSubclass(PipelineState):
    pass


class IntSubclass(int):
    pass


def object_at(state, path):
    for name in path:
        state = state[name]
    return state


@pytest.mark.parametrize("path", [(), ("framebuffer",), ("blend",), ("depth",), ("stencil",),
                                   ("stencil", "front"), ("stencil", "back")])
def test_every_object_rejects_missing_extra_and_non_dict_fields(path):
    original = default_state()
    for mutation in ("missing", "extra", "subclass", "wrong-key"):
        state = copy.deepcopy(original)
        target = object_at(state, path)
        if mutation == "missing":
            del target[next(iter(target))]
        elif mutation == "extra":
            target["unknown"] = False
        elif mutation == "wrong-key":
            name = next(iter(target))
            target[StringSubclass(name)] = target.pop(name)
        elif path:
            object_at(state, path[:-1])[path[-1]] = DictSubclass(target)
        else:
            state = DictSubclass(target)
        before = copy.deepcopy(state)
        with pytest.raises(ValueError):
            validate_state(state)
        assert state == before
    for invalid in (None, [], (), 0, "state"):
        state = copy.deepcopy(original)
        if path:
            object_at(state, path[:-1])[path[-1]] = invalid
        else:
            state = invalid
        with pytest.raises(ValueError):
            validate_state(state)


INVALID_FIELDS = [
    (("version",), value) for value in (True, False, 0, 2, -1, 1.0, "1", IntSubclass(1))
] + [
    (("framebuffer", name), value) for name, value in (
        ("color_format", "RGBA8"), ("color_format", "rgba16f"), ("color_format", StringSubclass("rgba8")),
        ("depth_stencil_format", None), ("depth_stencil_format", "z32"),
    )
] + [
    (("blend", name), value) for name in ("src_rgb", "dst_rgb", "src_alpha", "dst_alpha")
    for value in (None, 1, "src1-color", "source-alpha", "ONE", StringSubclass("one"))
] + [
    (("blend", name), value) for name in ("rgb_equation", "alpha_equation")
    for value in (None, "minimum", "multiply", StringSubclass("add"))
] + [
    (path, value) for path in (("blend", "enabled"), ("depth", "enabled"), ("depth", "write"), ("stencil", "enabled"))
    for value in (None, 0, 1, "false")
] + [
    (("color_mask",), value) for value in (None, (), (True,) * 4, [True] * 3, [True] * 5,
                                           [1, True, True, True], [None] * 4)
] + [
    (("blend", "constant"), value) for value in (None, (0,) * 4, [0] * 3, [0] * 5,
                                                   [True, 0, 0, 0], [Fraction(1, 2), 0, 0, 0],
                                                   [-0.01, 0, 0, 0], [1.01, 0, 0, 0],
                                                   [float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0])
] + [
    (path, value) for path in (("depth", "comparison"), ("stencil", "front", "comparison"),
                              ("stencil", "back", "comparison"))
    for value in (None, "LESS", "less-equal", StringSubclass("less"))
] + [
    (("stencil", face, name), value) for face in ("front", "back")
    for name in ("reference", "read_mask", "write_mask") for value in (-1, 256, 1.0, True, IntSubclass(1))
] + [
    (("stencil", face, name), value) for face in ("front", "back")
    for name in ("fail", "zfail", "zpass") for value in (None, "increment", StringSubclass("keep"))
] + [
    (("blend", name), "src-alpha-saturate") for name in ("dst_rgb", "dst_alpha")
]


@pytest.mark.parametrize("path,value", INVALID_FIELDS)
def test_malformed_state_is_rejected_without_mutation(path, value):
    state = default_state(depth_stencil_format="z24s8")
    object_at(state, path[:-1])[path[-1]] = value
    before = copy.deepcopy(state)
    color, depth = bytearray((23, 41, 73, 111)), bytearray((7, 8, 9, 10))
    with pytest.raises(ValueError):
        commit_fragment(state, (0, 0, 0, 0), color, source_depth=0.5,
                        destination_depth_stencil=depth, killed=True)
    assert color == bytearray((23, 41, 73, 111))
    assert depth == bytearray((7, 8, 9, 10))
    if path != ("blend", "constant") or not any(type(channel) is float and channel != channel for channel in value or []):
        assert state == before


def test_cross_attachment_validation_and_direct_dataclass_validation():
    for depth_format, depth_enabled, stencil_enabled in (("none", True, False), ("none", False, True),
                                                         ("z16", False, True), ("z16", True, True)):
        state = default_state(depth_stencil_format=depth_format)
        state["depth"]["enabled"] = depth_enabled
        state["stencil"]["enabled"] = stencil_enabled
        with pytest.raises(ValueError):
            validate_state(state)
    for changes in ({"constant": [0] * 4}, {"color_mask": [True] * 4}, {"depth_write": 1},
                    {"stencil_front": {}}, {"dst_alpha": "src-alpha-saturate"}):
        with pytest.raises(ValueError):
            PipelineState(**changes)
    with pytest.raises(ValueError):
        StencilFace(write_mask=True)
    with pytest.raises(ValueError):
        validate_state(StateSubclass())
    state = PipelineState()
    object.__setattr__(state, "depth_write", 1)
    with pytest.raises(ValueError):
        validate_state(state)
    state = PipelineState()
    object.__setattr__(state.stencil_front, "reference", 256)
    with pytest.raises(ValueError):
        validate_state(state)
    assert replace(PipelineState(), depth_enabled=True).depth_enabled
    for mode in (None, True, "additive", StringSubclass("disabled")):
        with pytest.raises(ValueError):
            default_state(blend_mode=mode)


@pytest.mark.parametrize("rgb_equation,alpha_equation", list(product(EQUATIONS, repeat=2)))
def test_all_standard_factor_pairs_and_separate_equations_against_rational_oracle(rgb_equation, alpha_equation):
    state = default_state(depth_stencil_format="none")
    state["blend"].update(enabled=True, rgb_equation=rgb_equation, alpha_equation=alpha_equation,
                            constant=[0.125, 0.375, 0.625, 0.875])
    samples = [((Fraction(1, 7), Fraction(2, 3), Fraction(5, 8), Fraction(3, 8)), bytes((19, 73, 151, 211))),
               ((-2.0, 1.5, 0.5, 0.0), bytes((255, 0, 127, 0)))]
    for source_index, destination_index in product(range(len(FACTORS)), range(len(FACTORS) - 1)):
        state["blend"].update(src_rgb=FACTORS[source_index], dst_rgb=FACTORS[destination_index],
                                src_alpha=FACTORS[(source_index + 3) % len(FACTORS)],
                                dst_alpha=FACTORS[(destination_index + 5) % (len(FACTORS) - 1)])
        for source, destination in samples:
            expected = oracle_blend(state, source, destination)
            assert blend_color(state, source, destination) == expected
            result = commit_fragment(state, source, destination, source_depth=0.5, destination_depth_stencil=None)
            assert result.color == expected and result.passed


@pytest.mark.parametrize("color_format", ("bgra8", "rgb565", "r8"))
def test_factors_observe_logical_destination_channels_and_implicit_alpha(color_format):
    state = default_state(color_format=color_format, depth_stencil_format="none")
    state["blend"].update(enabled=True, constant=[0.25, 0.75, 0.5, 0.125], alpha_equation="reverse-subtract")
    destination = oracle_pack_color(color_format, (Fraction(3, 7), Fraction(5, 9), Fraction(8, 11), Fraction(2, 3)))
    for first, second in product(FACTORS, FACTORS[:-1]):
        state["blend"].update(src_rgb=first, dst_rgb=second, src_alpha=first, dst_alpha=second)
        assert_commit(state, (0.25, 0.5, 0.75, 0.5), destination, None, 0.25)
    state["blend"].update(src_rgb="src-alpha-saturate", dst_rgb="zero", src_alpha="src-alpha-saturate",
                            dst_alpha="zero", alpha_equation="add")
    result = assert_commit(state, (1, 1, 1, 0.5), destination, None, 0)
    if color_format in ("r8", "rgb565"):
        assert result.color == bytes(len(destination))
    else:
        assert result.color[3] == 128


def test_min_max_ignore_zero_factors_and_alpha_saturate_is_one():
    state = default_state(depth_stencil_format="none")
    state["blend"].update(enabled=True, src_rgb="zero", dst_rgb="zero", src_alpha="zero", dst_alpha="zero",
                            rgb_equation="min", alpha_equation="max")
    assert blend_color(state, (0.75, 0.25, 0.5, 0.125), bytes((128, 255, 64, 192))) == bytes((128, 64, 64, 192))
    state["blend"].update(src_rgb="src-alpha-saturate", src_alpha="src-alpha-saturate",
                            rgb_equation="add", alpha_equation="add")
    assert blend_color(state, (1, 1, 1, 0.5), bytes((0, 0, 0, 204))) == bytes((51, 51, 51, 128))


@pytest.mark.parametrize("mode", ("disabled", "source-over"))
def test_legacy_mapping_matches_existing_dyadic_and_transparent_behavior(mode):
    import numpy as np
    from tools.host_blend import blend_rgba8

    state = default_state(blend_mode=mode, depth_stencil_format="none")
    for alpha, red, destination_alpha in product((0, 0.125, 0.5, 1), (-2, 0, 0.25, 0.5, 0.875, 1, 2), range(256)):
        source = (red, 0.375, 0.625, alpha)
        destination = bytes((19, 73, 151, destination_alpha))
        result = blend_color(state, source, destination)
        assert result == oracle_blend(state, source, destination)
        assert result == blend_rgba8(np.array(source, dtype=float), np.frombuffer(destination, dtype=np.uint8), mode).tobytes()


@pytest.mark.parametrize("color_format", COLOR_FORMATS)
@pytest.mark.parametrize("mode", ("disabled", "source-over"))
def test_all_color_masks_preserve_stored_bits_and_ignore_absent_channels(color_format, mode):
    destination = {"rgba8": bytes((37, 83, 129, 201)), "bgra8": bytes((129, 83, 37, 201)),
                   "rgb565": bytes((0xA5, 0x73)), "r8": bytes((37,))}[color_format]
    for mask in product((False, True), repeat=4):
        state = default_state(blend_mode=mode, color_format=color_format, depth_stencil_format="none")
        state["color_mask"] = list(mask)
        result = assert_commit(state, (0.875, 0.125, 0.5, 0.75), destination, None, 0)
        before, after = oracle_decode_color(color_format, destination), oracle_decode_color(color_format, result.color)
        for channel, enabled in enumerate(mask):
            if not enabled:
                assert before[channel] == after[channel]
        present = 1 if color_format == "r8" else 3 if color_format == "rgb565" else 4
        assert result.color_written == any(mask[:present])


@pytest.mark.parametrize("depth_format", ("z16", "z24s8"))
@pytest.mark.parametrize("comparison", COMPARES)
def test_depth_comparisons_use_quantized_incoming_depth_and_preserve_stencil(depth_format, comparison):
    state = default_state(depth_stencil_format=depth_format)
    state["depth"].update(enabled=True, comparison=comparison)
    maximum = MAX_DEPTH[depth_format]
    stencil = 171 if depth_format == "z24s8" else 0
    for stored_code in (0, 1, 2, maximum // 2, maximum - 1, maximum):
        attachment = pack_depth_stencil(depth_format, Fraction(stored_code, maximum), stencil)
        for offset in (-1, Fraction(-1, 4), 0, Fraction(1, 4), 1):
            source_depth = min(1, max(0, Fraction(stored_code + offset, maximum)))
            result = assert_commit(state, (0.5, 0.25, 0.75, 0), bytes((11, 31, 51, 71)), attachment, source_depth)
            assert unpack_depth_stencil(depth_format, result.depth_stencil)[1] == stencil
            if result.passed:
                assert result.depth_written
            else:
                assert not result.depth_written and not result.color_written


@pytest.mark.parametrize("front_facing", (True, False))
@pytest.mark.parametrize("comparison", COMPARES)
def test_exhaustive_small_stencil_comparisons_and_read_masks(front_facing, comparison):
    state = default_state(depth_stencil_format="z24s8")
    state["stencil"]["enabled"] = True
    state["depth"]["enabled"] = True
    selected = state["stencil"]["front" if front_facing else "back"]
    other = state["stencil"]["back" if front_facing else "front"]
    selected.update(comparison=comparison, fail="replace", zfail="increment-wrap", zpass="invert")
    other.update(comparison="never", fail="zero", zfail="zero", zpass="zero")
    for stored, reference, read_mask in product(range(4), repeat=3):
        selected.update(reference=reference, read_mask=read_mask)
        for depth in (0.25, 0.75):
            attachment = pack_depth_stencil("z24s8", 0.5, stored)
            result = assert_commit(state, (0.5, 0.25, 0.75, 0.5), bytes((19, 37, 73, 151)), attachment,
                                   depth, front_facing=front_facing)
            expected_pass = oracle_compare(comparison, reference & read_mask, stored & read_mask)
            assert (result.outcome != "stencil-fail") == expected_pass


@pytest.mark.parametrize("front_facing", (True, False))
@pytest.mark.parametrize("stage", ("fail", "zfail", "zpass"))
def test_exhaustive_small_stencil_operations_masks_and_fail_zfail_zpass(front_facing, stage):
    state = default_state(depth_stencil_format="z24s8")
    state["depth"].update(enabled=True, comparison="greater" if stage != "zpass" else "less")
    state["stencil"]["enabled"] = True
    face = state["stencil"]["front" if front_facing else "back"]
    face.update(comparison="never" if stage == "fail" else "always", read_mask=0,
                fail="replace", zfail="invert", zpass="zero")
    state["stencil"]["back" if front_facing else "front"].update(comparison="never", fail="zero")
    for stored, reference, write_mask, operation in product(range(4), range(4), range(4), OPERATIONS):
        face.update(reference=reference, write_mask=write_mask)
        face[stage] = operation
        attachment = pack_depth_stencil("z24s8", 0.5, stored)
        result = assert_commit(state, (0.5, 0.25, 0.75, 0), bytes((19, 37, 73, 151)), attachment,
                               0.25, front_facing=front_facing)
        assert result.outcome == {"fail": "stencil-fail", "zfail": "depth-fail", "zpass": "passed"}[stage]
        assert unpack_depth_stencil("z24s8", result.depth_stencil)[1] == oracle_stencil(operation, stored, reference, write_mask)
        assert result.depth_written == (stage == "zpass")


def test_every_stencil_byte_and_noncontiguous_write_masks_wrap_vs_clamp():
    state = default_state(depth_stencil_format="z24s8")
    state["stencil"]["enabled"] = True
    face = state["stencil"]["front"]
    face.update(reference=171, read_mask=0)
    for operation, stored, mask in product(OPERATIONS, range(256), (0, 0xAA, 255)):
        face.update(zpass=operation, write_mask=mask)
        attachment = pack_depth_stencil("z24s8", 0.5, stored)
        result = assert_commit(state, (0, 0, 0, 0), bytes(4), attachment, 0.25)
        assert unpack_depth_stencil("z24s8", result.depth_stencil) == (Fraction(8388608, 16777215),
                                                                     oracle_stencil(operation, stored, 171, mask))
        assert not result.depth_written


@pytest.mark.parametrize("mode,depth_enabled,depth_write,killed", list(product(
    ("disabled", "source-over"), (False, True), (False, True), (False, True),
)))
def test_transparent_depth_blend_stencil_and_kill_combinations(mode, depth_enabled, depth_write, killed):
    state = default_state(blend_mode=mode, depth_stencil_format="z24s8")
    state["depth"].update(enabled=depth_enabled, write=depth_write)
    state["stencil"]["front"].update(reference=23, fail="zero", zfail="increment-wrap", zpass="replace")
    color, attachment = bytearray((31, 73, 127, 191)), bytearray(pack_depth_stencil("z24s8", 0.5, 17))
    sources = ((0, 0, 0, 0), (0.75, 0.25, 1, 0), (0.5, 0.25, 0.75, 0.5))
    for source, depth, stencil in product(sources, (0.25, 0.5, 0.75), ("disabled", "pass", "fail")):
        state["stencil"]["enabled"] = stencil != "disabled"
        state["stencil"]["front"]["comparison"] = "never" if stencil == "fail" else "always"
        result = assert_commit(state, source, color, attachment, depth, killed=killed)
        if killed:
            assert result.color == color and result.depth_stencil == attachment
            assert not any((result.color_written, result.depth_written, result.stencil_written))
        elif result.passed and source[3] == 0:
            assert result.color_written
            assert result.depth_written == (depth_enabled and depth_write)
            if mode == "source-over":
                assert result.color == color
            else:
                assert result.color == oracle_pack_color("rgba8", source)


def test_write_flags_describe_permitted_writes_not_value_changes():
    state = default_state(depth_stencil_format="z24s8")
    state["depth"].update(enabled=True, comparison="equal")
    state["stencil"]["enabled"] = True
    state["stencil"]["front"].update(reference=17, zpass="replace")
    source = (0, 0, 0, 0)
    color, attachment = bytes(4), pack_depth_stencil("z24s8", 0.5, 17)
    depth = unpack_depth_stencil("z24s8", attachment)[0]
    result = assert_commit(state, source, color, attachment, depth)
    assert result.color == color and result.depth_stencil == attachment
    assert result.color_written and result.depth_written and result.stencil_written
    with pytest.raises(FrozenInstanceError):
        result.color = bytes((1, 2, 3, 4))
    state["color_mask"] = [False] * 4
    state["depth"]["write"] = False
    state["stencil"]["front"]["zpass"] = "keep"
    result = assert_commit(state, source, color, attachment, depth)
    assert not any((result.color_written, result.depth_written, result.stencil_written))


@pytest.mark.parametrize("color_format", COLOR_FORMATS)
def test_disabled_color_mask_does_not_suppress_depth_or_stencil(color_format):
    state = default_state(blend_mode="source-over", color_format=color_format, depth_stencil_format="z24s8")
    state["color_mask"] = [False] * 4
    state["depth"]["enabled"] = True
    state["stencil"]["enabled"] = True
    state["stencil"]["front"].update(reference=171, zpass="replace")
    color = pack_color(color_format, (0.5, 0.25, 0.75, 0.5))
    attachment = pack_depth_stencil("z24s8", 0.5, 17)
    result = assert_commit(state, (0, 0, 0, 0), color, attachment, 0.25)
    assert result.color == color and not result.color_written
    assert result.depth_written and result.stencil_written
    assert result.depth_stencil == pack_depth_stencil("z24s8", 0.25, 171)


@pytest.mark.parametrize("color_format,expected", [("rgba8", bytes((255, 128, 0, 64))),
                                                  ("bgra8", bytes((0, 128, 255, 64))),
                                                  ("rgb565", bytes((0, 252))), ("r8", bytes((255,)))])
def test_known_color_layouts_and_ties_to_even(color_format, expected):
    assert pack_color(color_format, (1, 0.5, 0, 0.25)) == expected
    assert unpack_color(color_format, expected) == oracle_decode_color(color_format, expected)
    for code in range(255):
        tie = Fraction(2 * code + 1, 510)
        source = (tie,) * 4
        assert pack_color(color_format, source) == oracle_pack_color(color_format, source)
    for channel in (float("inf"), float("nan"), float("-inf"), True, None, "0"):
        with pytest.raises(ValueError):
            pack_color(color_format, (channel, 0, 0, 0))
    assert pack_color(color_format, (-1e300, 1e300, 10 ** 400, Fraction(-10 ** 400))) == oracle_pack_color(
        color_format, (0, 1, 1, 0),
    )


def test_exhaustive_rgb565_r8_and_rgba8_component_roundtrips():
    for word in range(65536):
        raw = bytes((word % 256, word // 256))
        normalized = unpack_color("rgb565", raw)
        assert normalized == oracle_decode_color("rgb565", raw)
        assert pack_color("rgb565", normalized) == raw
    for channel in range(256):
        raw = bytes((channel,))
        assert pack_color("r8", unpack_color("r8", raw)) == raw
        for color_format, position in product(("rgba8", "bgra8"), range(4)):
            raw = bytes(channel if index == position else 127 for index in range(4))
            assert pack_color(color_format, unpack_color(color_format, raw)) == raw


@pytest.mark.parametrize("depth_format,expected", [("z16", bytes((0, 128))),
                                                  ("z24s8", bytes((0, 0, 128, 171)))])
def test_depth_stencil_layouts_and_exact_halfway_rounding(depth_format, expected):
    stencil = 171 if depth_format == "z24s8" else 0
    assert pack_depth_stencil(depth_format, Fraction(1, 2), stencil) == expected
    maximum = MAX_DEPTH[depth_format]
    for code in (0, 1, 2, 127, 128, maximum - 2, maximum - 1):
        tie = Fraction(2 * code + 1, 2 * maximum)
        raw = pack_depth_stencil(depth_format, tie, stencil)
        depth, stored_stencil = unpack_depth_stencil(depth_format, raw)
        assert depth == Fraction(oracle_round(tie, maximum), maximum)
        assert stored_stencil == stencil
    for invalid in (-0.01, 1.01, float("nan"), float("inf"), True, "0"):
        with pytest.raises(ValueError):
            pack_depth_stencil(depth_format, invalid)
    for invalid in (-1, 256, True, 1.0):
        with pytest.raises(ValueError):
            pack_depth_stencil(depth_format, 0.5, invalid)
    with pytest.raises(ValueError):
        pack_depth_stencil("z16", 0.5, 1)


def test_exhaustive_z16_and_seeded_z24s8_storage_roundtrips():
    for word in range(65536):
        raw = bytes((word % 256, word // 256))
        depth, stencil = unpack_depth_stencil("z16", raw)
        assert depth == Fraction(word, 65535) and stencil == 0
        assert pack_depth_stencil("z16", depth) == raw
    generator = random.Random(98435)
    words = [generator.randrange(2 ** 32) for _ in range(1024)]
    words.extend(depth + stencil * 16777216 for depth, stencil in product(
        (0, 1, 127, 128, 8388607, 8388608, 16777214, 16777215), range(256),
    ))
    for word in words:
        raw = word.to_bytes(4, "little")
        depth, stencil = unpack_depth_stencil("z24s8", raw)
        assert depth == Fraction(word % 16777216, 16777215) and stencil == word // 16777216
        assert pack_depth_stencil("z24s8", depth, stencil) == raw


@pytest.mark.parametrize("depth_format", ("z16", "z24s8"))
def test_float_depth_staging_adapter_roundtrips_exact_native_attachment_layout(depth_format):
    generator = random.Random(44104)
    maximum = MAX_DEPTH[depth_format]
    codes = [0, 1, maximum // 2, maximum - 1, maximum]
    codes.extend(generator.randrange(maximum + 1) for _ in range(2048))
    for code in codes:
        stencil = generator.randrange(256) if depth_format == "z24s8" else 0
        raw = pack_depth_stencil(depth_format, float(Fraction(code, maximum)), stencil)
        expected = code.to_bytes(2, "little") if depth_format == "z16" else bytes(
            (code % 256, (code // 256) % 256, code // 65536, stencil),
        )
        assert raw == expected
        depth, stored_stencil = unpack_depth_stencil(depth_format, raw)
        assert pack_depth_stencil(depth_format, float(depth), stored_stencil) == raw


@pytest.mark.parametrize("mode", ("disabled", "source-over"))
def test_float_halfway_neighbors_use_exact_binary_input_not_rounded_intermediates(mode):
    state = default_state(blend_mode=mode, depth_stencil_format="none")
    for code in range(255):
        nearest = float(Fraction(2 * code + 1, 510))
        for value in (math.nextafter(nearest, 0), nearest, math.nextafter(nearest, 1)):
            source = (value, 0.375, 0.625, 0.5)
            destination = bytes((code, 37, 73, 151))
            assert_commit(state, source, destination, None, 0.5)


@pytest.mark.parametrize("front_facing", (True, False))
def test_explicit_kill_suppresses_every_front_back_stencil_operation(front_facing):
    state = default_state(depth_stencil_format="z24s8")
    state["depth"].update(enabled=True, comparison="always")
    state["stencil"]["enabled"] = True
    face = state["stencil"]["front" if front_facing else "back"]
    face.update(reference=171, write_mask=255)
    color, attachment = bytearray((19, 37, 73, 151)), bytearray(pack_depth_stencil("z24s8", 0.5, 127))
    for operation, stage in product(OPERATIONS, ("fail", "zfail", "zpass")):
        face.update(comparison="never" if stage == "fail" else "always")
        face[stage] = operation
        state["depth"]["comparison"] = "never" if stage == "zfail" else "always"
        result = assert_commit(state, (0.75, 0.25, 0.5, 0), color, attachment, 0.25,
                               front_facing=front_facing, killed=True)
        assert result.color == color and result.depth_stencil == attachment
        assert not any((result.color_written, result.depth_written, result.stencil_written))


BAD_FRAGMENT_FIELDS = [
    ("source_rgba", value) for value in (None, (), [0] * 3, [0] * 5, "rgba", [True, 0, 0, 0],
                                       [float("nan"), 0, 0, 0], [0, float("inf"), 0, 0],
                                       [0, 0, 0, float("-inf")], ["0", 0, 0, 0])
] + [
    ("destination_color", value) for value in (None, b"", bytes(3), bytes(5), [0] * 4, memoryview(bytes(4)))
] + [
    ("destination_depth_stencil", value) for value in (None, b"", bytes(3), bytes(5), [0] * 4)
] + [
    ("source_depth", value) for value in (None, True, -0.01, 1.01, float("nan"), float("inf"), "0")
] + [
    (name, value) for name in ("front_facing", "killed") for value in (None, 0, 1, "false")
]


@pytest.mark.parametrize("field,value", BAD_FRAGMENT_FIELDS)
@pytest.mark.parametrize("outcome", ("killed", "stencil-fail", "depth-fail", "passed"))
def test_malformed_fragment_never_mutates_even_on_suppressed_paths(field, value, outcome):
    state = default_state(depth_stencil_format="z24s8")
    state["stencil"]["enabled"] = True
    state["stencil"]["front"].update(comparison="never" if outcome == "stencil-fail" else "always",
                                        fail="replace", zfail="replace", zpass="replace", reference=191)
    state["depth"].update(enabled=True, comparison="never" if outcome == "depth-fail" else "always")
    color, attachment = bytearray((31, 73, 127, 191)), bytearray((17, 23, 41, 83))
    arguments = {"source_rgba": [0.5, 0.25, 0.75, 0], "destination_color": color,
                 "source_depth": 0.25, "destination_depth_stencil": attachment,
                 "front_facing": True, "killed": outcome == "killed"}
    arguments[field] = value
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        commit_fragment(state, **arguments)
    assert state == before
    assert color == bytearray((31, 73, 127, 191))
    assert attachment == bytearray((17, 23, 41, 83))


def test_no_attachment_and_standalone_pack_blend_validation():
    state = default_state(depth_stencil_format="none")
    assert_commit(state, (0, 0, 0, 0), bytearray((19, 37, 73, 151)), None, 0.5)
    with pytest.raises(ValueError):
        commit_fragment(state, (0, 0, 0, 0), bytes(4), source_depth=0.5, destination_depth_stencil=b"")
    for invalid in (None, "RGBA8", StringSubclass("rgba8")):
        with pytest.raises(ValueError):
            pack_color(invalid, (0, 0, 0, 0))
        with pytest.raises(ValueError):
            unpack_color(invalid, bytes(4))
    for invalid in (None, "none", "Z16", StringSubclass("z16")):
        with pytest.raises(ValueError):
            pack_depth_stencil(invalid, 0.5)
        with pytest.raises(ValueError):
            unpack_depth_stencil(invalid, bytes(2))
    for color_format, size in (("rgba8", 4), ("bgra8", 4), ("rgb565", 2), ("r8", 1)):
        for invalid in (None, bytes(size - 1), bytes(size + 1), [0] * size):
            with pytest.raises(ValueError):
                unpack_color(color_format, invalid)
    for depth_format, size in (("z16", 2), ("z24s8", 4)):
        for invalid in (None, bytes(size - 1), bytes(size + 1), [0] * size):
            with pytest.raises(ValueError):
                unpack_depth_stencil(depth_format, invalid)
    color = bytearray((19, 37, 73, 151))
    with pytest.raises(ValueError):
        blend_color(state, (0, 0, float("nan"), 0), color)
    assert color == bytearray((19, 37, 73, 151))


@pytest.mark.parametrize("mode", ("disabled", "source-over"))
def test_repeated_commits_quantize_each_fragment_and_stage_all_side_effects(mode):
    state = default_state(blend_mode=mode, depth_stencil_format="z24s8")
    state["depth"].update(enabled=True, comparison="lequal")
    state["stencil"]["enabled"] = True
    state["stencil"]["front"]["zpass"] = "increment-wrap"
    color = bytes((19, 37, 73, 151))
    attachment = pack_depth_stencil("z24s8", 1, 254)
    expected_color, expected_depth = color, attachment
    draws = [((0.25, 0.5, 0.75, 0.125), 0.5, False), ((0, 0, 0, 0), 0.25, True),
             ((0.75, 0.25, 0.5, 0), 0.25, False), ((1, 1, 1, 1), 0.75, False),
             ((0.5, 0.75, 0.25, 0.5), 0.25, False)]
    for source, depth, killed in draws:
        expected = oracle_fragment(state, source, expected_color, expected_depth, depth, killed=killed)
        expected_color, expected_depth = expected["color"], expected["depth_stencil"]
        result = commit_fragment(validate_state(state), source, color, source_depth=depth,
                                 destination_depth_stencil=attachment, killed=killed)
        assert asdict(result) == expected
        color, attachment = result.color, result.depth_stencil
    assert unpack_depth_stencil("z24s8", attachment)[1] == 1
