"""Measure wall time of CPU packet rendering, not GPU throughput or FPGA FPS."""

import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

import numpy as np

if __package__:
    from .mgcreference import load_scene, render, subset_frame
    from .mgcreference.artifacts import write_json
    from .mgcreference.scene import DEFAULT_DATASET, SCENES, frame_info
    from .reference_demo import selected_frames
else:
    from mgcreference import load_scene, render, subset_frame
    from mgcreference.artifacts import write_json
    from mgcreference.scene import DEFAULT_DATASET, SCENES, frame_info
    from reference_demo import selected_frames


def benchmark(frame, *, renderer="numpy", shading="auto", warmup=2, repeats=5) -> dict:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    for _ in range(warmup):
        render(frame, renderer, shading)
    samples = []
    expected_hash = None
    for _ in range(repeats):
        start = perf_counter_ns()
        result = render(frame, renderer, shading)
        elapsed = perf_counter_ns() - start
        digest = result.sha256
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError("nondeterministic framebuffer between timed runs")
        expected_hash = digest
        samples.append(elapsed)
    median_ns = statistics.median(samples)
    return {
        **frame_info(frame), **result.summary(), "warmup": warmup, "repeats": repeats,
        "elapsed_ns": samples, "median_ms": median_ns / 1e6,
        "min_ms": min(samples) / 1e6, "max_ms": max(samples) / 1e6,
        "mean_ms": statistics.mean(samples) / 1e6,
        "stdev_ms": statistics.stdev(samples) / 1e6 if repeats > 1 else 0.0,
        "triangles_per_second_at_median": len(frame.triangles) * 1e9 / median_ns,
        "covered_fragments_per_second_at_median": result.covered_fragments * 1e9 / median_ns,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=["utah", "gears", "bucky12"])
    parser.add_argument("--frames", default="0")
    parser.add_argument("--triangles", type=int)
    parser.add_argument("--first-triangle", type=int, default=0)
    parser.add_argument("--renderer", choices=("numpy", "scalar"), default="numpy")
    parser.add_argument("--shading", choices=("auto", "coverage", "ceramic", "flat"), default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark/reference.json"))
    args = parser.parse_args(argv)
    try:
        if args.warmup < 0 or args.repeats < 1:
            raise ValueError("warmup must be nonnegative and repeats positive")
        records = []
        loading = {}
        for scene in args.scenes:
            start = perf_counter_ns()
            frames = load_scene(scene, args.dataset)
            loading[scene] = perf_counter_ns() - start
            for index in selected_frames(args.frames, len(frames)):
                frame = subset_frame(frames[index], args.first_triangle, args.triangles)
                record = benchmark(frame, renderer=args.renderer, shading=args.shading,
                                   warmup=args.warmup, repeats=args.repeats)
                records.append(record)
                print(f"CPU-only {scene} frame {index}: {record['triangles']} triangles, "
                      f"median {record['median_ms']:.3f} ms ({args.repeats} repeats)")
        write_json(args.output, {
            "engine": "cpu-reference", "hardware_execution": False,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            "environment": {
                "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                "machine": platform.machine(), "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "gc_enabled": gc.isenabled(), "timer": "time.perf_counter_ns (monotonic wall time)",
            },
            "methodology": {
                "timed": "render only: fresh framebuffer, coverage, shading, stores and counters",
                "excluded": "parsing, assembly verification, PNG/PPM/raw export, hashing, warmups",
                "loading_ns_separately": loading,
                "correctness": "raw u32le framebuffer SHA256 checked after each timed render",
                "scope": "Python/NumPy CPU reference; no Mesa, mgcsim, RTL, FPGA, driver or GPU utilization",
                "caveat": "warm-cache sequential samples, normal OS scheduling and GC; not hardware performance",
            },
            "results": records,
        })
        return 0
    except (ValueError, OSError, FloatingPointError) as error:
        print(f"benchmark_reference: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
