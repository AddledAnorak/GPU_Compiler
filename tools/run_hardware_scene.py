"""Submit a validated first-frame binary prefix through the managed CPU driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from tools.mgcdriver import Driver, HardwareSimulationBackend, Program, ProgramFormat
from tools.mgchw import assemble_hardware, load_mem, parse_source
from tools.mgchw.cli import DEFAULT_EXAMPLES
from tools.mgcreference import load_scene, render, rgb565_to_rgb8
from tools.mgcreference.artifacts import write_png

SCENES = {"bucky12": "bucky_12frame_dualbuf_clean", "utah": "utah_12frame_dualbuf_shader"}


def run_scene(scene: str, output: Path, *, allow_approximate_fp16: bool = False) -> dict:
    if scene not in SCENES:
        raise ValueError("only source/binary-matched Bucky12 and Utah are supported by this runner")
    if scene == "utah" and not allow_approximate_fp16:
        raise ValueError("Utah needs explicit --allow-approximate-fp16; RTL color packing remains unknown")
    stem = SCENES[scene]
    source = (DEFAULT_EXAMPLES / f"{stem}.asm").read_text()
    words = load_mem(DEFAULT_EXAMPLES / f"{stem}.program.mem")
    if assemble_hardware(source) != words:
        raise ValueError("supplied assembly and binary no longer match")
    labels = parse_source(source)[1]
    stop_pc = labels["frame_done_00"]
    entry = labels["frame_00"] if scene == "utah" else 0
    program = Program.from_words(words[:stop_pc], format=ProgramFormat.HARDWARE)
    data = np.asarray(load_mem(DEFAULT_EXAMPLES / f"{stem}.data.mem"), dtype="<u4").tobytes()
    frame = load_scene(scene)[0]
    backend = HardwareSimulationBackend(memory_size=1 << 32, quantum=4096,
                                        allow_approximate_fp16=allow_approximate_fp16)
    with Driver(backend) as driver:
        framebuffer = driver.allocate(0x80000, alignment=4, address=0)
        packets = driver.allocate(len(data), alignment=4, address=0x100000)
        control = driver.allocate(4, alignment=4, address=0x90000000)
        packets.upload(data)
        started = time.perf_counter()
        fence = driver.submit(program, buffers=(framebuffer, packets, control), entry=entry,
                              max_steps=10000000, execution_timeout=30.0)
        execution = fence.wait(35.0)
        seconds = time.perf_counter() - started
        raw = framebuffer.download(offset=frame.bank * 0x40000, size=256 * 256 * 4)
    actual = np.frombuffer(raw, dtype="<u4").reshape(256, 256)
    expected = render(frame)
    mismatches = int(np.count_nonzero(actual != expected.pixels))
    support_mismatches = int(np.count_nonzero((actual != 0) != (expected.pixels != 0)))
    output.mkdir(parents=True, exist_ok=True)
    (output / "framebuffer.u32le.bin").write_bytes(raw)
    write_png(output / "frame.png", rgb565_to_rgb8(actual))
    report = {
        "pipeline": "supplied hardware-format binary prefix -> managed host driver -> functional HardwareCPU",
        "actual_fpga_execution": False, "scene": scene, "frame": 0,
        "original_program_words": len(words), "submitted_prefix_words": len(program.words),
        "entry_pc": entry, "entry_explanation": "frame_00 bypasses Utah startup markers/hold only" if entry else "original entry",
        "completion": "end of validated first-frame prefix, before display swap/animation holds",
        "fence": fence.status.value, "host_instructions": execution.steps, "host_seconds": seconds,
        "approximate_fp16": allow_approximate_fp16,
        "framebuffer_sha256": hashlib.sha256(raw).hexdigest(),
        "reference_sha256": expected.sha256, "color_mismatches": mismatches,
        "nonzero_support_mismatches": support_mismatches,
        "pixel_exact_reference_match": mismatches == 0,
        "shader_packing_hardware_equivalence": "unverified" if allow_approximate_fp16 else "not applicable",
    }
    (output / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    if mismatches and not allow_approximate_fp16:
        raise ValueError(f"integer binary/reference mismatch: {mismatches} pixels; see {output}")
    if support_mismatches:
        raise ValueError(f"binary/reference coverage mismatch: {support_mismatches} pixels; see {output}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, default="bucky12")
    parser.add_argument("--output", type=Path, default=Path("artifacts/driver-scene"))
    parser.add_argument("--allow-approximate-fp16", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_scene(args.scene, args.output,
                                   allow_approximate_fp16=args.allow_approximate_fp16), indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(json.dumps({"status": "error", "error": type(error).__name__, "message": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
