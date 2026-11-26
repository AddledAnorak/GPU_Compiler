"""Safe simulation CLI; deliberately contains no hardware-opening option."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (
    Driver, DriverError, HardwareSimulationBackend, Program, ProgramFormat,
    SimulationBackend, load_mem,
)


def _number(text):
    try:
        return int(text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer (decimal or 0x hexadecimal)") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Submit tagged legacy programs to a bounded host simulator")
    parser.add_argument("program", type=Path)
    parser.add_argument("--format", required=True, choices=[format.value for format in ProgramFormat])
    parser.add_argument("--backend", choices=("legacy-simulation", "hardware-reference"),
                        default="legacy-simulation", help="reference simulation is not physical hardware")
    parser.add_argument("--kind", choices=("mem", "binary"))
    parser.add_argument("--byteorder", choices=("little", "big"), default="little")
    parser.add_argument("--memory-size", type=_number, default=1 << 20)
    parser.add_argument("--buffer-size", type=_number)
    parser.add_argument("--buffer-address", type=_number, default=0)
    parser.add_argument("--data", type=Path, help="optional dense/sparse .mem buffer contents")
    parser.add_argument("--output", type=Path, help="write complete buffer as raw little-endian bytes")
    parser.add_argument("--entry", type=_number, default=0, help="instruction word index")
    parser.add_argument("--max-steps", type=_number, default=1_000_000)
    parser.add_argument("--execution-timeout", type=float, default=5.0)
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    fence = None
    try:
        program = Program.load(args.program, format=args.format, byteorder=args.byteorder, kind=args.kind)
        backend_class = SimulationBackend if args.backend == "legacy-simulation" else HardwareSimulationBackend
        backend = backend_class(memory_size=args.memory_size)
        backend.validate(program)
        data = b"" if args.data is None else b"".join(word.to_bytes(4, "little") for word in load_mem(args.data))
        if args.buffer_size is None and not data and args.output is not None:
            raise DriverError("--output requires --buffer-size or --data")
        with Driver(backend) as driver:
            buffer = None
            if args.buffer_size is not None or data:
                size = args.buffer_size if args.buffer_size is not None else len(data)
                buffer = driver.allocate(size, alignment=4, address=args.buffer_address)
                buffer.upload(data)
            fence = driver.submit(program, buffers=() if buffer is None else (buffer,), entry=args.entry,
                                  max_steps=args.max_steps, execution_timeout=args.execution_timeout)
            result = fence.wait(args.wait_timeout)
            if args.output is not None:
                args.output.write_bytes(buffer.download())
            print(json.dumps({"status": fence.status.value, "format": program.format.value,
                              "backend": result.backend, "steps": result.steps,
                              "registers": result.registers}))
        return 0
    except (DriverError, OSError, UnicodeError) as error:
        print(json.dumps({"status": "error", "fence_status": None if fence is None else fence.status.value,
                          "error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
