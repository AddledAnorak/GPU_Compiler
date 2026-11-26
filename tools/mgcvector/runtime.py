"""Bounded functional vector execution, never a scalar or board fallback."""

from __future__ import annotations

import copy
import math
import re
import struct
import threading
from types import MappingProxyType

from tools.mgcdriver import Driver, FenceStatus, HardwareSimulationBackend, Program, ProgramFormat
from tools.mgchw import assemble_hardware, decode_word, disassemble_hardware, parse_source
from tools.shader_layout import ShaderLayout

VECTOR_TARGET = "fp16-hardware-vector-experimental-v1"

_MAX_PROGRAM_WORDS = 262144
_MAX_STEPS = 1000000
_ABI_BYTES = 262144
_MAX_MEMORY_BYTES = 32 << 20
_WAIT_SECONDS = 5.0
_CONTRACT = {
    "version": 1,
    "lanes": 4,
    "lane_mode": "broadcast-scalar",
    "scratch_word_offset": 32768,
    "status_word_offset": 65535,
    "pending": 0,
    "failed": 1,
    "committed": 2,
    "approximate_fp16_required": True,
    "vector_spacing_nops": 16,
    "rtl_timing_validated": False,
}
_OPERATIONS = {
    "preload_low", "load_high", "copy", "load", "store", "nop",
    "vbcast", "vget_lane0_i32", "vfadd16", "vfsub16", "vfmul16", "vfmax16",
    "icmp_ge", "icmp_gt", "icmp_lt", "jump",
}
_EXPANSIONS = {"movi_u16": 2, "movi_i32": 2, "movi_f16": 2, "wait_load": 4, "wait_fadd": 6}
_SOURCE_OPERATIONS = _OPERATIONS | set(_EXPANSIONS)


class VectorExecutionError(ValueError):
    """Invalid vector invocation or failed execution; contains no output payload."""


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise VectorExecutionError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _layout_snapshot(layout):
    source = layout.source if isinstance(layout, ShaderLayout) else layout
    if not isinstance(source, dict):
        raise VectorExecutionError("vector layout must be a dict or ShaderLayout")
    source = copy.deepcopy(source)
    if (type(source.get("version")) is not int or source["version"] != 1 or
            source.get("program_format") != ProgramFormat.HARDWARE.value or
            type(source.get("shader_abi_version")) is not int or source["shader_abi_version"] != 2 or
            source.get("address_units") != "bytes" or source.get("compiler_target") != VECTOR_TARGET):
        raise VectorExecutionError("vector reflection requires version 1, hardware ABI 2 bytes and compiler_target")
    contract = source.get("vector_contract")
    if not isinstance(contract, dict) or set(contract) != set(_CONTRACT) | {"scratch_words"}:
        raise VectorExecutionError("vector_contract must contain exactly the supported keys")
    for name, value in _CONTRACT.items():
        if type(contract[name]) is not type(value) or contract[name] != value:
            raise VectorExecutionError(f"unsupported vector_contract {name}")
    _integer(contract["scratch_words"], "scratch_words", 1, 16384)
    checked = ShaderLayout(source)
    for category in ("inputs", "uniforms", "outputs"):
        for entry in checked.entries(category):
            if entry["type"] != "float" or type(entry["bit_size"]) is not int:
                raise VectorExecutionError("vector reflection supports only floating-point scalars")
            if category == "outputs" and entry["word_offset"] >= _CONTRACT["scratch_word_offset"]:
                raise VectorExecutionError("reflected outputs must stay below scratch word offset 32768")
    return checked


def _program(assembly, stage):
    header = f"; mgc {VECTOR_TARGET} backend, ABI=2, stage={stage}"
    if (not isinstance(assembly, str) or len(assembly) > 16 << 20 or
            not re.match(re.escape(header) + r"(?=\s|,|$)", assembly)):
        raise VectorExecutionError("assembly provenance does not match the vector compiler target, ABI and stage")
    instructions, _ = parse_source(assembly)
    if not instructions:
        raise VectorExecutionError("vector program must contain 1..262144 words")
    for instruction in instructions:
        if instruction.mnemonic not in _SOURCE_OPERATIONS:
            raise VectorExecutionError(f"unsupported vector instruction {instruction.mnemonic}")
    word_count = instructions[-1].pc + _EXPANSIONS.get(instructions[-1].mnemonic, 1)
    _integer(word_count, "program words", 1, _MAX_PROGRAM_WORDS)
    words = assemble_hardware(assembly)
    _integer(len(words), "program words", 1, _MAX_PROGRAM_WORDS)
    for word in words:
        if decode_word(word).mnemonic not in _OPERATIONS:
            raise VectorExecutionError("program contains an instruction outside the vector allowlist")
    if assemble_hardware("\n".join(disassemble_hardware(words))) != words:
        raise VectorExecutionError("vector hardware program does not round-trip canonically")
    return Program.from_words(words, format=ProgramFormat.HARDWARE)


def _payload(values, offsets, name):
    if not isinstance(values, dict):
        raise VectorExecutionError(f"{name} must be a word_offset-to-float dict")
    values = dict(values)
    for offset in values:
        _integer(offset, f"{name} word_offset", 0, 65535)
    missing = set(offsets) - set(values)
    extra = set(values) - set(offsets)
    if missing or extra:
        raise VectorExecutionError(f"{name} must match reflection exactly: missing={sorted(missing)}, extra={sorted(extra)}")
    packed = {}
    for offset, value in values.items():
        if not isinstance(value, float) or not math.isfinite(value):
            raise VectorExecutionError(f"{name}[{offset}] requires a finite float, not a raw integer payload")
        try:
            half = struct.pack("<e", value)
        except (OverflowError, struct.error) as error:
            raise VectorExecutionError(f"{name}[{offset}] does not encode as finite FP16") from error
        if int.from_bytes(half, "little") & 0x7C00 == 0x7C00:
            raise VectorExecutionError(f"{name}[{offset}] does not encode as finite FP16")
        packed[offset] = half + bytes(2)
    return packed


class VectorShaderExecutor:
    """One broadcast-scalar shader invocation at a time on managed simulation."""

    def __init__(self, assembly, layout, *, allow_approximate_fp16=False, max_steps=1000000):
        if allow_approximate_fp16 is not True:
            raise VectorExecutionError("vector execution requires explicit allow_approximate_fp16=True; RTL timing is unverified")
        self._max_steps = _integer(max_steps, "max_steps", 1, _MAX_STEPS)
        self._lock = threading.RLock()
        try:
            self.layout = _layout_snapshot(layout)
            self._stage = self.layout.source["stage"]
            self._offsets = {category: tuple(entry["word_offset"] for entry in self.layout.entries(category))
                             for category in ("inputs", "uniforms", "outputs")}
            bases = dict(self.layout.source["bases"])
            memory_size = max(bases.values()) + _ABI_BYTES
            if memory_size > _MAX_MEMORY_BYTES:
                raise VectorExecutionError("vector ABI aperture exceeds the bounded 32 MB memory budget")
            self._program = _program(assembly, self._stage)
            scratch_words = self.layout.source["vector_contract"]["scratch_words"]
        except VectorExecutionError:
            raise
        except Exception as error:
            raise VectorExecutionError(f"invalid vector artifact: {error}") from error
        self.driver = Driver(HardwareSimulationBackend(memory_size=memory_size, quantum=4096,
                                                     allow_approximate_fp16=True))
        regions = {}
        self._starts = {}
        try:
            for category, domain in (("inputs", "input"), ("uniforms", "uniform"), ("outputs", self._stage)):
                offsets = self._offsets[category]
                if offsets:
                    start, end = min(offsets), max(offsets) + 1
                    self._starts[domain] = start
                    regions[domain] = self.driver.allocate((end - start) * 4, alignment=4,
                                                           address=bases[domain] + start * 4)
            regions["scratch"] = self.driver.allocate(scratch_words * 4, alignment=4,
                                                      address=bases[self._stage] + _CONTRACT["scratch_word_offset"] * 4)
            regions["status"] = self.driver.allocate(4, alignment=4,
                                                     address=bases[self._stage] + _CONTRACT["status_word_offset"] * 4)
        except Exception as error:
            self.driver.close()
            raise VectorExecutionError(f"vector buffer allocation failed: {error}") from error
        self.buffers = MappingProxyType(regions)

    @property
    def program(self):
        return self._program

    @property
    def max_steps(self):
        return self._max_steps

    def execute(self, inputs, uniforms):
        with self._lock:
            fence = None
            try:
                payloads = (("input", _payload(inputs, self._offsets["inputs"], "inputs")),
                            ("uniform", _payload(uniforms, self._offsets["uniforms"], "uniforms")))
                for buffer in self.buffers.values():
                    buffer.upload(bytes(buffer.size))
                for domain, payload in payloads:
                    for offset, value in payload.items():
                        self.buffers[domain].upload(value, offset=(offset - self._starts[domain]) * 4)
                fence = self.driver.submit(self._program, buffers=tuple(self.buffers.values()),
                                           max_steps=self._max_steps, execution_timeout=_WAIT_SECONDS)
                result = fence.wait(timeout=_WAIT_SECONDS)
                if fence.status != FenceStatus.COMPLETED:
                    raise VectorExecutionError("vector fence did not complete")
                _integer(result.steps, "host_steps", 0, self._max_steps)
                status = int.from_bytes(self.buffers["status"].download(), "little")
                if status != _CONTRACT["committed"]:
                    raise VectorExecutionError(f"vector status is not committed (status={status})")
                output = self.buffers.get(self._stage)
                words = [] if output is None else [word for (word,) in struct.iter_unpack("<I", output.download())]
                for word in words:
                    if word >> 16 or word & 0x7C00 == 0x7C00:
                        raise VectorExecutionError("vector output requires finite low16 FP16 with upper16 zero")
                raw_outputs = {offset: words[offset - self._starts[self._stage]]
                               for offset in self._offsets["outputs"]}
                outputs = {offset: struct.unpack("<e", word.to_bytes(2, "little"))[0]
                           for offset, word in raw_outputs.items()}
                return {
                    "outputs": outputs, "raw_outputs": raw_outputs, "fence": fence.status.value,
                    "host_steps": result.steps, "program_format": self._program.format.value,
                    "compiler_target": VECTOR_TARGET, "status": "committed",
                    "fpga_execution": False, "rtl_timing_validated": False, "approximate_fp16": True,
                }
            except Exception as error:
                if fence is not None and not fence.done:
                    fence.cancel()
                if isinstance(error, VectorExecutionError):
                    raise
                raise VectorExecutionError(f"vector execution failed: {error}") from error

    def close(self):
        with self._lock:
            self.driver.close()

    def __enter__(self):
        if self.driver.closed:
            raise VectorExecutionError("vector executor is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
