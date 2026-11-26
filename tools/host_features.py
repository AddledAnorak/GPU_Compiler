"""Strict opt-in ABI4 host contracts, never hardware addresses or operations."""

from __future__ import annotations

from tools.mgcdriver import ProgramFormat


DRAW_FEATURES = {
    "version": 1,
    "shader_abi_version": 4,
    "max_instances": 256,
    "max_expanded_vertices": 65536,
    "indexing": "base-instance-plus-floor-local-instance-over-divisor",
}
BUILTINS = {
    "gl_InstanceID": ([65535], "int", 32, "raw32", "vertex"),
    "gl_VertexID": ([65520], "int", 32, "raw32", "vertex"),
    "gl_BaseInstance": ([65521], "uint", 32, "raw32", "vertex"),
    "gl_DrawID": ([65522], "int", 32, "raw32", "vertex"),
    "gl_FragCoord": ([65524, 65525, 65526, 65527], "float", 32, "low_fp16", "fragment"),
    "gl_FrontFacing": ([65528], "bool", 1, "raw32", "fragment"),
}
DERIVATIVE_MAILBOX = {
    "request_word_offset": 65496,
    "operand_word_offset": 65497,
    "response_word_offset": 65498,
    "operation_ids": {"dx": 1, "dy": 2},
    "semantics": "fine-quad",
}


def _exact(value, expected, name):
    if type(value) is not type(expected):
        raise ValueError(f"unsupported {name} type")
    if isinstance(expected, dict):
        if set(value) != set(expected):
            raise ValueError(f"unsupported {name} keys")
        for key, item in expected.items():
            _exact(value[key], item, f"{name}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(f"unsupported {name} length")
        for supplied, item in zip(value, expected):
            _exact(supplied, item, name)
    elif value != expected:
        raise ValueError(f"unsupported {name} value")


def validate_shader_features(source):
    if source.get("version") != 4:
        if "host_features" in source or source.get("shader_abi_version") == 4:
            raise ValueError("host features require explicit layout4/ABI4")
        return None
    if (source.get("supported") is not True or source.get("shader_abi_version") != 4 or
            type(source.get("shader_abi_version")) is not int or
            source.get("program_format") != ProgramFormat.LEGACY.value or
            source.get("address_units") != "words" or source.get("stage") not in ("vertex", "fragment") or
            "host_instancing" in source):
        raise ValueError("host features require supported legacy layout4/ABI4")
    features = source.get("host_features")
    expected_keys = {"version", "max_instances", "max_expanded_vertices", "reserved_input_words",
                     "reserved_output_words", "builtins", "uses_derivatives", "derivative_mailbox",
                     "spill", "fault_word_offset"}
    if type(features) is not dict or set(features) != expected_keys:
        raise ValueError("host features require an exact versioned shader contract")
    for key, expected in {"version": 1, "max_instances": 256, "max_expanded_vertices": 65536,
                          "reserved_input_words": [65504, 65535], "reserved_output_words": [57344, 65534],
                          "fault_word_offset": 65499}.items():
        _exact(features[key], expected, f"host features {key}")
    builtins = features["builtins"]
    if type(builtins) is not list or len(builtins) > len(BUILTINS):
        raise ValueError("invalid host builtin list")
    used = set()
    for builtin in builtins:
        if type(builtin) is not dict or builtin.get("name") not in BUILTINS:
            raise ValueError("unsupported host builtin")
        name = builtin["name"]
        offsets, kind, size, storage, stage = BUILTINS[name]
        if name in used or source["stage"] != stage:
            raise ValueError("duplicate or wrong-stage host builtin")
        if name == "gl_FragCoord":
            supplied_offsets = builtin.get("word_offsets")
            if (type(supplied_offsets) is not list or not supplied_offsets or
                    any(type(offset) is not int or offset not in offsets for offset in supplied_offsets) or
                    supplied_offsets != sorted(set(supplied_offsets))):
                raise ValueError("fragment coordinates require an ascending used-component subset")
            offsets = supplied_offsets
        _exact(builtin, {"name": name, "word_offsets": offsets, "type": kind,
                         "bit_size": size, "storage": storage}, "host builtin")
        used.add(name)
    if type(features["uses_derivatives"]) is not bool:
        raise ValueError("host derivative use must be explicitly boolean")
    if features["uses_derivatives"]:
        if source["stage"] != "fragment":
            raise ValueError("host derivatives require a fragment shader")
        _exact(features["derivative_mailbox"], DERIVATIVE_MAILBOX, "host derivative mailbox")
    elif features["derivative_mailbox"] is not None:
        raise ValueError("unused host derivative mailbox must be null")
    spill = features["spill"]
    if type(spill) is not dict or set(spill) != {"word_offset", "max_words", "words"}:
        raise ValueError("invalid host spill contract")
    _exact(spill["word_offset"], 57344, "host spill offset")
    _exact(spill["max_words"], 4096, "host spill capacity")
    if type(spill["words"]) is not int or not 0 <= spill["words"] <= 4096:
        raise ValueError("host spill words exceed the declared capacity")
    return features


def validate_draw_features(artifact):
    if artifact.get("schema") != "mgc-host-draw-v4":
        if "host_features" in artifact or artifact.get("shader_abi_version") == 4:
            raise ValueError("host features require version4 captured resources")
        return False
    if artifact.get("program_format") != ProgramFormat.LEGACY.value:
        raise ValueError("host features cannot execute a hardware program")
    _exact(artifact.get("host_features"), DRAW_FEATURES, "captured host features")
    _exact(artifact.get("shader_abi_version"), 4, "captured host ABI")
    return True


class VertexPayload(dict):
    def __init__(self, values, builtins):
        super().__init__(values)
        self.builtins = builtins
