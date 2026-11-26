"""Checked compiler reflection for provisional host shader invocation."""

from __future__ import annotations

from dataclasses import dataclass

from tools.mgcdriver import ProgramFormat
from tools.host_instancing import HOST_INSTANCE_ID_WORD, validate_shader_instancing


def bounded_integer(value, name, maximum=65535, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class ShaderLayout:
    source: dict

    def __post_init__(self):
        if not isinstance(self.source, dict) or type(self.source.get("version")) is not int or self.source.get("version") not in (1, 2, 3, 4):
            raise ValueError("unsupported compiler layout version")
        if self.source.get("supported") is not True:
            raise ValueError(f"compiler layout is unavailable: {self.source.get('errors', [])}")
        target = ProgramFormat(self.source.get("program_format"))
        if type(self.source.get("shader_abi_version")) is not int:
            raise ValueError("shader ABI version must be an integer")
        if target == ProgramFormat.C_INTERPRETER and self.source["version"] != 1:
            raise ValueError("current C target requires layout version1")
        if self.source.get("shader_abi_version") != (5 if target == ProgramFormat.C_INTERPRETER else
                                                   2 if target == ProgramFormat.HARDWARE else
                                                   4 if self.source["version"] == 4 else
                                                   3 if self.source["version"] == 3 else 1):
            raise ValueError("compiler layout target/ABI mismatch")
        if self.source.get("address_units") != ("words" if target == ProgramFormat.LEGACY else "bytes"):
            raise ValueError("compiler layout address units mismatch")
        if self.source.get("stage") not in ("vertex", "fragment"):
            raise ValueError("unsupported reflected shader stage")
        validate_shader_instancing(self.source)
        if self.source["version"] == 4 and "host_texture" in self.source:
            from tools.host_texture_features import validate_texture_features_metadata

            if self.source["stage"] != "fragment":
                raise ValueError("extended host textures require a fragment shader")
            validate_texture_features_metadata(self.source["host_texture"])
        elif self.source["version"] == 2 or (self.source["version"] == 3 and "host_texture" in self.source):
            from tools.host_texture import host_texture_metadata_contract
            texture = self.source.get("host_texture")
            if type(texture) is not dict:
                raise ValueError("unsupported host texture layout contract")
            expected = host_texture_metadata_contract(texture)
            if (target != ProgramFormat.LEGACY or self.source["stage"] != "fragment" or
                    not isinstance(texture, dict) or set(texture) != set(expected) or
                    any(type(texture[name]) is not type(value) or texture[name] != value
                        for name, value in expected.items())):
                raise ValueError("unsupported host texture layout contract")
        elif "host_texture" in self.source:
            raise ValueError("host texture requires compiler layout version2")
        if "uses_discard" in self.source and not isinstance(self.source["uses_discard"], bool):
            raise ValueError("discard metadata must be explicitly boolean")
        if "fragment_status" in self.source:
            status = self.source["fragment_status"]
            expected = {"version": 1, "word_offset": 65535, "pending": 0, "discarded": 1, "committed": 2}
            if (target != ProgramFormat.LEGACY or self.source["stage"] != "fragment" or
                    not isinstance(status, dict) or set(status) != set(expected) or
                    any(type(status[name]) is not int or status[name] != value for name, value in expected.items())):
                raise ValueError("unsupported fragment completion status contract")
        bases = self.source.get("bases")
        domains = {"input", "uniform", "vertex", "fragment"}
        if self.source["version"] == 4:
            domains.add("texture")
        if not isinstance(bases, dict) or set(bases) != domains:
            raise ValueError("compiler layout requires all four ABI bases")
        scale = self.stride
        spans = []
        for name, base in bases.items():
            bounded_integer(base, f"{name} base", (1 << 32) - 65536 * scale)
            if target != ProgramFormat.LEGACY and base % 4:
                raise ValueError("hardware layout bases must be four-byte aligned")
            spans.append((base, base + 65536 * scale))
        if any(max(left[0], right[0]) < min(left[1], right[1])
               for index, left in enumerate(spans) for right in spans[index + 1:]):
            raise ValueError("compiler layout ABI regions overlap")
        for category in ("inputs", "outputs", "uniforms"):
            entries = self.source.get(category)
            if not isinstance(entries, list) or len(entries) > 4096:
                raise ValueError("invalid or oversized shader layout entries")
            seen = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("layout entry must be an object")
                offset = bounded_integer(entry.get("word_offset"), "scalar offset")
                if offset in seen:
                    raise ValueError("duplicate reflected scalar offset")
                if category == "inputs" and self.host_instancing and offset == HOST_INSTANCE_ID_WORD:
                    raise ValueError("ordinary input overlaps reserved host instance-ID word")
                if category == "outputs" and self.fragment_status and offset == self.fragment_status["word_offset"]:
                    raise ValueError("output overlaps reserved fragment completion word")
                if self.host_features and ((category == "inputs" and (offset >= 65504 or offset == 65498)) or
                                           (category == "outputs" and offset >= 57344)):
                    raise ValueError("ordinary scalar overlaps reserved host feature storage")
                if category == "uniforms" and self.host_texture and not self.host_features and offset >= 15 * 4096:
                    raise ValueError("ordinary uniform overlaps reserved host texture storage")
                seen.add(offset)
                if entry.get("type") not in ("float", "int", "uint") or entry.get("bit_size") not in (16, 32):
                    raise ValueError("unsupported or ambiguous reflected scalar type")
                if category != "uniforms":
                    bounded_integer(entry.get("location"), "semantic location", 255)
                    bounded_integer(entry.get("component"), "semantic component", 3)
                    if entry.get("interpolation") not in ("smooth", "flat", "noperspective"):
                        raise ValueError("unsupported reflected interpolation")

    @property
    def target(self):
        return ProgramFormat(self.source["program_format"])

    @property
    def stride(self):
        return 1 if self.target == ProgramFormat.LEGACY else 4

    def entries(self, category):
        return self.source[category]

    @property
    def fragment_status(self):
        return self.source.get("fragment_status")

    @property
    def host_texture(self):
        return self.source.get("host_texture")

    @property
    def host_instancing(self):
        return self.source.get("host_instancing") or self.host_features

    @property
    def host_features(self):
        return self.source.get("host_features")

    def base(self, domain):
        return self.source["bases"][domain]


def link_varyings(vertex: ShaderLayout, fragment: ShaderLayout):
    if vertex.source["stage"] != "vertex" or fragment.source["stage"] != "fragment":
        raise ValueError("stage linkage requires a vertex/fragment pair")
    if vertex.target != fragment.target or vertex.source["shader_abi_version"] != fragment.source["shader_abi_version"]:
        raise ValueError("linked stages must use the same explicit target and ABI")
    outputs = {}
    for entry in vertex.entries("outputs"):
        key = (entry["location"], entry["component"])
        if key in outputs:
            raise ValueError("ambiguous vertex semantic output")
        outputs[key] = entry
    mapping = {}
    for entry in fragment.entries("inputs"):
        key = (entry["location"], entry["component"])
        if key not in outputs or key[0] == 0:
            raise ValueError(f"unmatched or unsupported fragment input semantic {key}")
        output = outputs[key]
        if (output["type"], output["bit_size"]) != (entry["type"], entry["bit_size"]):
            raise ValueError("linked varying types differ")
        if entry["type"] != "float" and entry["interpolation"] != "flat":
            raise ValueError("integer varyings require explicit flat interpolation")
        mapping[entry["word_offset"]] = output["word_offset"]
    return mapping
