"""Loads docs/isa/isa.yaml and exposes it as convenient lookup structures.

This is the single point of contact with the spec data for both the
assembler/disassembler (this package) and the emulator (tools/mgcsim).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_ISA_YAML = REPO_ROOT / "docs" / "isa" / "isa.yaml"


@dataclass
class FieldDef:
    name: str
    hi_bit: int
    lo_bit: int
    width: int

    def extract(self, word: int) -> int:
        mask = (1 << self.width) - 1
        return (word >> self.lo_bit) & mask

    def insert(self, word: int, value: int) -> int:
        mask = (1 << self.width) - 1
        if value < 0 or value > mask:
            raise ValueError(
                f"value {value} does not fit in field {self.name} "
                f"({self.width} bits, max {mask})"
            )
        word &= ~(mask << self.lo_bit)
        word |= (value & mask) << self.lo_bit
        return word


@dataclass
class InstrDef:
    mnemonic: str
    channel: int | None
    opcode: int | None
    subop: int | None
    operands: list[str]
    status: str
    latency_class: str | None = None
    control_flow: bool = False
    note: str = ""


@dataclass
class ChannelDef:
    channel: int
    opcode: int
    bin: str
    meaning: str
    mnemonics: list[str]


class ISA:
    """Parsed view of isa.yaml."""

    def __init__(self, path: pathlib.Path = DEFAULT_ISA_YAML):
        self.path = pathlib.Path(path)
        with open(self.path) as f:
            self.raw = yaml.safe_load(f)

        self.instruction_width_bits: int = self.raw["instruction_width_bits"]
        self.fields: dict[str, FieldDef] = {}
        for f_ in self.raw["fields"]:
            if not isinstance(f_, dict) or "name" not in f_:
                continue
            hi, lo = f_["bits"]
            self.fields[f_["name"]] = FieldDef(f_["name"], hi, lo, f_["width"])

        self.channels: dict[int, ChannelDef] = {}
        self.opcode_by_channel: dict[int, int] = {}
        for _, entry in self.raw["opcodes"].items():
            if not isinstance(entry, dict) or "channel" not in entry:
                continue
            ch = entry["channel"]
            cd = ChannelDef(
                channel=ch,
                opcode=int(entry["bin"], 2),
                bin=entry["bin"],
                meaning=entry["meaning"],
                mnemonics=entry.get("mnemonics", []),
            )
            self.channels[ch] = cd
            self.opcode_by_channel[ch] = cd.opcode

        self.instructions: dict[str, InstrDef] = {}
        for mnem, spec in self.raw["instructions"].items():
            channel = spec.get("channel")
            opcode = self.opcode_by_channel.get(channel) if channel is not None else None
            subop = None
            if channel is not None and mnem in self.channels[channel].mnemonics:
                subop = self.channels[channel].mnemonics.index(mnem)
            self.instructions[mnem] = InstrDef(
                mnemonic=mnem,
                channel=channel,
                opcode=opcode,
                subop=subop,
                operands=list(spec.get("operands", [])),
                status=spec.get("status", "guess"),
                latency_class=spec.get("latency_class"),
                control_flow=spec.get("control_flow", False),
                note=spec.get("note", ""),
            )

        self.latency_classes: dict[str, dict] = self.raw.get("latency_classes", {})

        self.constants_by_index: dict[int, dict] = {}
        self.constants_by_name: dict[str, int] = {}
        _NAME_ALIASES = {
            "-inf": "neg_inf", "negative infinity": "neg_inf",
            "-1.0": "neg_one", "0.0": "zero", "1.0": "one", "0.5": "half",
            "pi approximation": "pi", "positive infinity": "pos_inf",
            "quiet nan": "nan",
        }
        for entry in self.raw["fp16_constant_bank"]["entries"]:
            idx = entry["index"]
            if not isinstance(idx, int):
                continue
            self.constants_by_index[idx] = entry
            meaning = str(entry.get("meaning", "")).lower()
            alias = _NAME_ALIASES.get(meaning)
            if alias:
                self.constants_by_name[alias] = idx

        self.register_count: int = self.raw["scalar_register_count"]
        self.direct_register_range = tuple(self.raw["direct_register_range"])
        self.condition_registers: list[str] = self.raw["condition_registers"]["names"]
        self.vector_register_count: int = self.raw["vector_registers"]["assumed_count"]
        self.vector_lanes: int = self.raw["vector_registers"]["assumed_lanes"]

        cft = self.raw.get("control_flow_target", {})
        self.control_flow_target_width: int = cft.get("width", 22)

    def channel_mnemonics(self, channel: int) -> list[str]:
        return self.channels[channel].mnemonics

    def mnemonic_for(self, channel: int, subop: int) -> str | None:
        mnems = self.channels[channel].mnemonics
        if 0 <= subop < len(mnems):
            return mnems[subop]
        return None
