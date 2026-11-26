"""Render recovered precomputed scenes on the CPU, never instruction binaries."""

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .mgcreference import load_scene, render, rgb565_to_rgb8, subset_frame
    from .mgcreference.artifacts import compare_framebuffer, export_frame, write_json, write_png
    from .mgcreference.scene import DEFAULT_DATASET, SCENES, inventory
else:
    from mgcreference import load_scene, render, rgb565_to_rgb8, subset_frame
    from mgcreference.artifacts import compare_framebuffer, export_frame, write_json, write_png
    from mgcreference.scene import DEFAULT_DATASET, SCENES, inventory


def selected_frames(specification: str, count: int) -> list[int]:
    if specification == "all":
        return list(range(count))
    try:
        indices = [int(index) for index in specification.split(",")]
    except ValueError as error:
        raise ValueError("frames must be 'all' or comma-separated integer indices") from error
    if len(set(indices)) != len(indices) or any(index < 0 or index >= count for index in indices):
        raise ValueError(f"frame indices must be unique and within 0..{count - 1}")
    return indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", choices=SCENES, default="utah")
    parser.add_argument("--frames", default="0", help="all, or comma-separated zero-based indices")
    parser.add_argument("--triangles", type=int, help="positive count; default is a full frame")
    parser.add_argument("--first-triangle", type=int, default=0)
    parser.add_argument("--renderer", choices=("numpy", "scalar"), default="numpy")
    parser.add_argument("--shading", choices=("auto", "coverage", "ceramic", "flat"), default="auto")
    parser.add_argument("--scale", type=int, default=3, help="nearest-neighbor preview only, 1..8")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference"))
    parser.add_argument("--inspect", action="store_true", help="inventory all 13 files and decode all frame headers")
    comparison_group = parser.add_mutually_exclusive_group()
    comparison_group.add_argument("--compare-dump", type=Path, help="compare one selected frame against a raw u32le bank")
    comparison_group.add_argument("--compare-hardware-prefix", action="store_true",
                                  help="execute the separate HardwareCPU on real binary up to this frame-0 prefix")
    parser.add_argument("--hardware-max-steps", type=int, default=2000000)
    parser.add_argument("--allow-approximate-hardware-fp16", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.inspect:
            value = inventory(args.dataset)
            write_json(args.output / "inventory.json", value)
            print(f"CPU reference inventory: {len(value['files'])} source files")
            for scene, frames in value["scenes"].items():
                print(f"{scene}: {len(frames)} frames, counts {[frame['available_triangles'] for frame in frames]}")
            return 0
        if not 1 <= args.scale <= 8:
            raise ValueError("scale must be 1..8")
        frames = load_scene(args.scene, args.dataset)
        indices = selected_frames(args.frames, len(frames))
        if (args.compare_dump or args.compare_hardware_prefix) and len(indices) != 1:
            raise ValueError("hardware/dump comparison requires exactly one selected frame")
        metadata = []
        previews = []
        failed_comparison = False
        hardware_report = None
        for index in indices:
            frame = subset_frame(frames[index], args.first_triangle, args.triangles)
            result = render(frame, args.renderer, args.shading)
            if args.compare_hardware_prefix:
                from mgcreference.hardware import run_hardware_prefix

                dump = args.output / f"hardware-binary-{frame.scene}-prefix.u32le.bin"
                hardware_report = run_hardware_prefix(
                    frame, dump, max_steps=args.hardware_max_steps,
                    allow_approximate_fp16=args.allow_approximate_hardware_fp16,
                )
                comparison = compare_framebuffer(result, dump)
                comparison["dump_producer"] = "tools.mgchw.HardwareCPU executing the recovered .program.mem"
                hardware_report["comparison"] = comparison
                write_json(args.output / "hardware-prefix-comparison.json", hardware_report)
                failed_comparison = not comparison["equal"]
                print(f"Real-binary functional simulator: {hardware_report['steps']} instructions, "
                      f"{comparison['mismatched_pixels']} mismatched pixels; not FPGA execution")
            if args.compare_dump:
                raw_output = args.output / f"{frame.scene}-frame-{frame.index:02d}.framebuffer.u32le.bin"
                if args.compare_dump.resolve() == raw_output.resolve():
                    raise ValueError("output would overwrite the comparison input; choose another output directory")
                comparison = compare_framebuffer(result, args.compare_dump)
                write_json(args.output / "dump-comparison.json", comparison)
                failed_comparison = not comparison["equal"]
                print(f"External dump comparison: {comparison['mismatched_pixels']} mismatched pixels")
            record = export_frame(args.output, frame, result, args.scale)
            metadata.append(record)
            previews.append(rgb565_to_rgb8(result.pixels))
            print(f"CPU reference {args.scene} frame {index}: {result.triangle_count} triangles, "
                  f"{result.covered_fragments} fragments, {int(result.touched.sum())} visible pixels, "
                  f"sha256={result.sha256}")
        if len(previews) > 1:
            columns = min(4, len(previews))
            rows = (len(previews) + columns - 1) // columns
            sheet = np.zeros((rows * 256, columns * 256, 3), dtype=np.uint8)
            for slot, preview in enumerate(previews):
                row, column = divmod(slot, columns)
                sheet[row * 256:(row + 1) * 256, column * 256:(column + 1) * 256] = preview
            write_png(args.output / f"{args.scene}-contact-sheet.png", sheet)
        write_json(args.output / f"{args.scene}-manifest.json", {
            "engine": "cpu-reference", "hardware_execution": False, "frames": metadata,
            "hardware_binary_comparison": hardware_report,
        })
        return 1 if failed_comparison else 0
    except (ValueError, OSError, RuntimeError, ImportError, FloatingPointError) as error:
        print(f"reference_demo: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
