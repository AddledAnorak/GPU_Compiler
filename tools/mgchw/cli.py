from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import zlib
from pathlib import Path

from .cpu import ByteMemory, HardwareCPU
from .encoding import assemble_hardware, disassemble_hardware, load_mem, parse_source

DEFAULT_EXAMPLES = Path(__file__).resolve().parents[2] / "hardware-examples/aakash-2026-09-28"


def verify_examples(directory: Path = DEFAULT_EXAMPLES) -> dict:
    results = {"format": "fp16-hardware-v1", "binary_roundtrips": [], "source_pairs": [], "files": []}
    for path in sorted(directory.glob("*.program.mem")):
        words = load_mem(path)
        rebuilt = assemble_hardware("\n".join(disassemble_hardware(words)))
        results["binary_roundtrips"].append({"file": path.name, "words": len(words), "matches": words == rebuilt})
    for path in sorted(directory.glob("*.asm")):
        words = load_mem(path.with_suffix(".program.mem"))
        generated = assemble_hardware(path.read_text())
        differences = [{"word": index, "source": f"{left:08x}", "binary": f"{right:08x}"}
                       for index, (left, right) in enumerate(zip(generated, words)) if left != right]
        results["source_pairs"].append({
            "file": path.name, "source_words": len(generated), "binary_words": len(words),
            "matches": generated == words, "overlapping_word_mismatches": len(differences),
            "first_differences": differences[:8],
        })
    for path in sorted(directory.iterdir()):
        if path.suffix in {".asm", ".mem"}:
            contents = path.read_bytes()
            results["files"].append({"file": path.name, "bytes": len(contents),
                                     "sha256": hashlib.sha256(contents).hexdigest()})
    return results


def framebuffer_png(memory: ByteMemory, base: int, width: int = 256, height: int = 256) -> bytes:
    if width <= 0 or height <= 0:
        raise ValueError("framebuffer dimensions must be positive")
    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)
        for column in range(width):
            pixel = memory.read32(base + (row * width + column) * 4) & 65535
            scanlines.extend((((pixel >> 11) & 31) * 255 // 31,
                              ((pixel >> 5) & 63) * 255 // 63, (pixel & 31) * 255 // 31))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(scanlines))) + chunk(b"IEND", b""))


def _main() -> int:
    parser = argparse.ArgumentParser(description="Observed GPU binary encoding and functional host execution")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    verify.add_argument("--report", type=Path)
    verify.add_argument("--strict-sources", action="store_true")
    execute = commands.add_parser("execute")
    execute.add_argument("program", type=Path)
    execute.add_argument("--data", type=Path, required=True)
    execute.add_argument("--source", type=Path)
    execute.add_argument("--stop-label", default="frame_done_00")
    execute.add_argument("--stop-pc", type=int)
    execute.add_argument("--max-steps", type=int, default=10000000)
    execute.add_argument("--skip-startup-hold", action="store_true")
    execute.add_argument("--allow-approximate-fp16", action="store_true")
    execute.add_argument("--framebuffer-base", type=lambda text: int(text, 0), default=0x40000)
    execute.add_argument("--png", type=Path)
    execute.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        results = verify_examples(args.examples)
        failed = any(not entry["matches"] for entry in results["binary_roundtrips"])
        failed |= args.strict_sources and any(not entry["matches"] for entry in results["source_pairs"])
    else:
        words = load_mem(args.program)
        labels = {}
        if args.source:
            if assemble_hardware(args.source.read_text()) != words:
                parser.error("source and binary differ; use an explicit --stop-pc without --source")
            labels = parse_source(args.source.read_text())[1]
        stop_pc = args.stop_pc
        if stop_pc is None:
            if args.stop_label not in labels:
                parser.error("provide a matching --source with --stop-label, or --stop-pc")
            stop_pc = labels[args.stop_label]
        skips = {}
        if args.skip_startup_hold:
            if "startup_marker_hold_outer" not in labels or "frame_00" not in labels:
                parser.error("startup hold skip requires validated Utah source labels")
            skips[labels["startup_marker_hold_outer"]] = labels["frame_00"]
        memory = ByteMemory()
        memory.map_region(0, 0x80000, name="sample framebuffer banks")
        memory.map_region(0x90000000, 4, name="sample display-control literal, not a host AXI mapping")
        memory.load_words(load_mem(args.data))
        cpu = HardwareCPU(memory=memory, allow_approximate_fp16=args.allow_approximate_fp16)
        started = time.perf_counter()
        cpu.run(words, max_steps=args.max_steps, stop_pc=stop_pc, skip_pcs=skips)
        elapsed = time.perf_counter() - started
        image = framebuffer_png(memory, args.framebuffer_base)
        if args.png:
            args.png.parent.mkdir(parents=True, exist_ok=True)
            args.png.write_bytes(image)
        results = {
            "execution": "functional CPU model of supplied hardware binary, not FPGA",
            "program": str(args.program), "pc": cpu.pc, "host_instructions": cpu.steps,
            "host_seconds": elapsed, "host_only_skips": cpu.instruction_counts["host-only-skip"],
            "approximate_fp16": args.allow_approximate_fp16,
            "rgb565_png_sha256": hashlib.sha256(image).hexdigest(),
            "instruction_counts": dict(cpu.instruction_counts),
        }
        failed = False
    output = json.dumps(results, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n")
    return int(failed)


def main() -> int:
    try:
        return _main()
    except (ValueError, RuntimeError, OSError) as error:
        print(json.dumps({"status": "error", "error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
