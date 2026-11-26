"""Reflected single-invocation shader execution using managed C resources."""

from __future__ import annotations

import copy
import hashlib
import struct
import threading

from tools.mgcdriver import Driver, FenceStatus, Program, ProgramFormat
from tools.shader_layout import ShaderLayout, link_varyings
from .backend import CInterpreterBackend
from .encoding import assemble_c, disassemble_c, fp16_bits, integer


class CShaderExecutor:
    def __init__(self, assembly, layout, *, max_steps=1000000):
        self._lock = threading.RLock()
        self.max_steps = integer(max_steps, 1, 10000000, "instruction budget")
        self.layout = ShaderLayout(copy.deepcopy(layout.source if isinstance(layout, ShaderLayout) else layout))
        if self.layout.target != ProgramFormat.C_INTERPRETER:
            raise ValueError("C execution requires reflected c-interpreter-v1/ABI5")
        if self.layout.source.get("uses_discard"):
            raise ValueError("C executor has no discard/output-valid contract")
        header = f"; mgc c-interpreter-v1 backend, ABI=5, stage=MESA_SHADER_{self.layout.source['stage'].upper()}"
        if not isinstance(assembly, str) or not assembly or assembly.splitlines()[0] != header:
            raise ValueError("C shader assembly header/stage/ABI does not match reflection")
        words = assemble_c(assembly)
        if assemble_c("\n".join(disassemble_c(words))) != words:
            raise ValueError("C machine-code roundtrip failed")
        self.program = Program.from_words(words, format=ProgramFormat.C_INTERPRETER)
        self.backend = CInterpreterBackend()
        self.driver = Driver(self.backend)
        self.buffers = {}
        self.entries = {category: {entry["word_offset"]: entry for entry in self.layout.entries(category)}
                        for category in ("inputs", "uniforms", "outputs")}
        try:
            count = sum(map(len, self.entries.values()))
            if count > 4096:
                raise ValueError("C reflected resource count exceeds 4096 words")
            for category, entries in self.entries.items():
                domain = "input" if category == "inputs" else "uniform" if category == "uniforms" else self.layout.source["stage"]
                for offset, entry in entries.items():
                    if entry["type"] not in ("float", "int", "uint", "bool"):
                        raise ValueError("unsupported C reflected scalar type")
                    self.buffers[category, offset] = self.driver.allocate(4, alignment=4,
                        address=self.layout.base(domain) + offset * 4)
        except Exception:
            self.close()
            raise

    def _payload(self, values, category):
        if not isinstance(values, dict) or any(type(offset) is not int for offset in values):
            raise ValueError(f"{category} must map exact integer reflected word offsets to values")
        if set(values) != set(self.entries[category]):
            raise ValueError(f"{category} must supply exactly the reflected offsets {sorted(self.entries[category])}")
        result = {}
        for offset, value in values.items():
            kind = self.entries[category][offset]["type"]
            bits = fp16_bits(value) if kind == "float" else integer(value, -(1 << 31), (1 << 32) - 1, kind) & 0xffffffff
            result[offset] = bits.to_bytes(4, "little")
        return result

    def execute(self, inputs, uniforms):
        with self._lock:
            return self._execute(inputs, uniforms)

    def _execute(self, inputs, uniforms):
        payloads = {category: self._payload(values, category)
                    for category, values in (("inputs", inputs), ("uniforms", uniforms))}
        for buffer in self.buffers.values():
            buffer.upload(bytes(4))
        for category, values in payloads.items():
            for offset, contents in values.items():
                self.buffers[category, offset].upload(contents)
        fence = self.driver.submit(self.program, buffers=tuple(self.buffers.values()),
                                   max_steps=self.max_steps, execution_timeout=10.0)
        result = fence.wait(15.0)
        if fence.status != FenceStatus.COMPLETED:
            raise ValueError("C shader did not complete")
        raw = {offset: int.from_bytes(self.buffers["outputs", offset].download(), "little")
               for offset in self.entries["outputs"]}
        outputs = {}
        for offset, bits in raw.items():
            kind = self.entries["outputs"][offset]["type"]
            if kind == "float":
                if bits >> 16:
                    raise ValueError("C float output is not low16 FP16")
                outputs[offset] = struct.unpack("<e", bits.to_bytes(2, "little"))[0]
            else:
                outputs[offset] = bits - (1 << 32) if kind == "int" and bits & (1 << 31) else bits
        return {"outputs": outputs, "raw_outputs": raw, "host_steps": result.steps,
                "fence": fence.status.value, "completion": "stop_core-and-irq",
                "program_format": self.program.format.value, "shader_abi_version": 5,
                "program_words": len(self.program.words),
                "program_sha256": hashlib.sha256(self.program.to_bytes()).hexdigest(),
                "fpga_execution": False, "rtl_timing_validated": False,
                "c_build": copy.deepcopy(self.backend.build_evidence)}

    def close(self):
        with self._lock:
            try:
                self.driver.close()
            finally:
                self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def execute_c_pair(compiled, vertex_inputs, vertex_uniforms, fragment_uniforms, *, max_steps=1000000):
    layouts = {stage: ShaderLayout(compiled["layouts"][stage]) for stage in ("vs", "fs")}
    mapping = link_varyings(layouts["vs"], layouts["fs"])
    with CShaderExecutor(compiled["assembly"]["vs"], layouts["vs"], max_steps=max_steps) as executor:
        vertex = executor.execute(vertex_inputs, vertex_uniforms)
    fragment_inputs = {destination: vertex["outputs"][source] for destination, source in mapping.items()}
    with CShaderExecutor(compiled["assembly"]["fs"], layouts["fs"], max_steps=max_steps) as executor:
        fragment = executor.execute(fragment_inputs, fragment_uniforms)
    return {"vertex": vertex, "fragment": fragment, "varying_map": mapping,
            "pipeline": "GLSL -> Mesa NIR -> current-C scalar ASM -> machine words -> supplied C",
            "rasterization": False, "interpolation": False, "fpga_execution": False,
            "rtl_timing_validated": False, "numerics": "C-contract FP16, not full GLSL FP32 conformance"}
