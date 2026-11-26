"""Optional bounded cross-check against the separate real-binary simulator."""

import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np

from .scene import DATA_BASE, FRAMEBUFFER_BYTES, Frame


def run_hardware_prefix(frame: Frame, output: Path, *, max_steps: int = 2000000,
                        allow_approximate_fp16: bool = False) -> dict:
    if frame.index != 0 or frame.first_triangle != 0:
        raise ValueError("binary check currently supports a prefix of frame 0 only")
    if max_steps <= 0:
        raise ValueError("hardware instruction budget must be positive")
    if frame.shading == "ceramic" and not allow_approximate_fp16:
        raise ValueError("Utah binary requires explicit --allow-approximate-hardware-fp16; RTL is unverified")
    repository = Path(__file__).resolve().parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    hardware = importlib.import_module("tools.mgchw")
    stem = frame.source_path.name.removesuffix(".data.mem")
    program_path = frame.source_path.parent / f"{stem}.program.mem"
    assembly_path = frame.source_path.parent / f"{stem}.asm"
    words = hardware.load_mem(program_path)
    data = hardware.load_mem(frame.source_path)
    memory = hardware.ByteMemory()
    memory.map_region(0, 2 * FRAMEBUFFER_BYTES, name="framebuffer banks")
    memory.map_region(0x90000000, 4, name="observed display control")
    memory.load_words(data, base=DATA_BASE)
    cpu = hardware.HardwareCPU(memory=memory, allow_approximate_fp16=allow_approximate_fp16)
    cpu.load_program(words)
    loops = []
    for pc, instruction in enumerate(cpu.program[:-1]):
        branch = cpu.program[pc + 1]
        if (instruction.mnemonic == "icmp_gt" and instruction.source_a == 27
                and branch.mnemonic == "jump" and branch.invert
                and branch.condition == instruction.destination):
            loops.append(pc)
    if not loops:
        raise ValueError("cannot identify the packet-count loop in the real binary")
    skip_pcs = {}
    source_equivalent = None
    if assembly_path.exists():
        source = assembly_path.read_text(encoding="ascii")
        source_equivalent = hardware.assemble_hardware(source) == words
        if source_equivalent:
            _, labels = hardware.parse_source(source)
            if "startup_marker_hold_outer" in labels:
                skip_pcs[labels["startup_marker_hold_outer"]] = labels["frame_00"]
    loop_pc = loops[0]
    cpu.run(max_steps=max_steps, stop_pc=loop_pc, skip_pcs=skip_pcs)
    startup_steps = cpu.steps
    if cpu.registers[26] != frame.header_address + 4:
        raise ValueError("binary first packet pointer disagrees with the selected data")
    stride = 15 if frame.shading == "ceramic" else 13
    for completed in range(len(frame.triangles)):
        if cpu.steps >= max_steps:
            raise hardware.StepLimitExceeded("host instruction budget exhausted before next packet")
        cpu.step()
        if cpu.steps >= max_steps:
            raise hardware.StepLimitExceeded("host instruction budget exhausted inside packet loop")
        cpu.run(max_steps=max_steps - cpu.steps, stop_pc=loop_pc)
        expected_pointer = frame.header_address + 4 + (completed + 1) * stride * 4
        if cpu.registers[26] != expected_pointer:
            raise ValueError("binary packet pointer did not advance by the inferred stride")
    bank = np.fromiter((memory.read32(frame.bank * FRAMEBUFFER_BYTES + pixel * 4)
                        for pixel in range(FRAMEBUFFER_BYTES // 4)), dtype="<u4")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bank.tobytes())
    return {
        "engine": "hardware-binary-functional-simulator", "actual_gpu_execution": False,
        "simulator": "tools.mgchw.HardwareCPU", "program": str(program_path),
        "program_sha256": hashlib.sha256(program_path.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(frame.source_path.read_bytes()).hexdigest(),
        "program_words": len(words), "assembly_binary_equal": source_equivalent,
        "frame": frame.index, "bank": frame.bank, "triangles_completed": len(frame.triangles),
        "steps": cpu.steps, "startup_steps": startup_steps, "max_steps": max_steps,
        "stop_pc": cpu.pc, "triangle_loop_pc": loop_pc,
        "stop_reason": "selected prefix complete; stopped before the next triangle-count test",
        "final_packet_byte_pointer": cpu.registers[26],
        "original_count_register_remaining": cpu.registers[27],
        "program_or_register_patches": False,
        "host_only_pacing_skips": [{"from_pc": origin, "to_pc": target} for origin, target in skip_pcs.items()],
        "approximate_fp16_allowed": allow_approximate_fp16,
        "instruction_counts": dict(sorted(cpu.instruction_counts.items())),
        "dump": str(output), "dump_u32le_sha256": hashlib.sha256(bank.tobytes()).hexdigest(),
        "limits": "functional instruction semantics; no RTL latency/timing or FPGA execution",
    }
