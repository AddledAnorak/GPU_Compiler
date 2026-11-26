"""Assemble programs or run assembly and raw words in Emu."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import struct
import sys

from . import ASSUMPTIONS, AssemblyError, Emulator, ExecutionFault, Memory, assemble


def read_mem(path: Path) -> list[int]:
    words = []
    with path.open(encoding="ascii") as source:
        for line_number, line in enumerate(source, 1):
            word = line.split("//", 1)[0].strip()
            if not word:
                continue
            if re.fullmatch(r"[0-9a-fA-F]{8}", word) is None:
                raise ValueError(f"{path}:{line_number}: expected one eight-digit hex word")
            words.append(int(word, 16))
    if not words:
        raise ValueError(f"{path}: no words")
    return words


def _number(text):
    return int(text, 0)


def _region(text):
    base, separator, size = text.partition(":")
    try:
        if not separator:
            raise ValueError
        return _number(base), _number(size)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected BASE:SIZE in bytes") from error


def _read_assembly(path, *, sample_source):
    try:
        return assemble(path.read_text(encoding="utf-8"), sample_source=sample_source)
    except AssemblyError as error:
        raise AssemblyError(f"{path}:{error}") from error


def _sample_source_option(parser):
    parser.add_argument("--sample-source", action="store_true",
                        help="use Akash's next-PC jump spelling and recovered graphics macros")


def _assemble_main(argv):
    parser = argparse.ArgumentParser(prog="python -m tools.mgcemu assemble",
                                     description="Translate Emu assembly to hexadecimal words.")
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="create a .program.mem file")
    _sample_source_option(parser)
    args = parser.parse_args(argv)
    try:
        words = _read_assembly(args.source, sample_source=args.sample_source)
        with args.output.open("x", encoding="ascii") as output:
            output.writelines(f"{word:08x}\n" for word in words)
    except (OSError, ValueError) as error:
        print(f"mgcemu: {error}", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["assemble"]:
        return _assemble_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Use 'assemble SOURCE -o OUTPUT' to emit machine words.")
    parser.add_argument("program", type=Path, help=".asm or hexadecimal .program.mem file")
    parser.add_argument("--data", type=Path, help="read-only hexadecimal .data.mem file")
    parser.add_argument("--data-base", type=_number, default=0x100000, help="data byte base")
    parser.add_argument("--map", action="append", type=_region, default=[], metavar="BASE:SIZE",
                        help="map an initially zero writable data region")
    parser.add_argument("--allow", action="append", choices=ASSUMPTIONS, default=[])
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--stop-pc", type=_number, help="inspect state before running this word PC")
    parser.add_argument("--timing-check", action="store_true",
                        help="reject reads before the ISA's documented safe result spacing")
    parser.add_argument("--trace", type=Path, help="create a JSON Lines instruction trace")
    parser.add_argument("--dump", type=Path, help="create a little-endian uint32 data dump")
    parser.add_argument("--dump-region", type=_region, metavar="BASE:SIZE")
    _sample_source_option(parser)
    args = parser.parse_args(argv)
    if (args.dump is None) != (args.dump_region is None):
        parser.error("--dump and --dump-region must be used together")
    is_assembly = args.program.suffix.lower() == ".asm"
    if args.sample_source and not is_assembly:
        parser.error("--sample-source applies to .asm input only")
    try:
        memory = Memory()
        for base, size in args.map:
            memory.map(base, size)
        if args.data is not None:
            memory.map_words(args.data_base, read_mem(args.data))
        words = (_read_assembly(args.program, sample_source=args.sample_source)
                 if is_assembly else read_mem(args.program))
        emulator = Emulator(words, memory=memory, allow=args.allow, timing_check=args.timing_check)
        if args.trace is not None:
            with args.trace.open("x", encoding="utf-8") as output:
                result = emulator.run(max_steps=args.max_steps, stop_pc=args.stop_pc,
                                      trace=lambda step: output.write(json.dumps(asdict(step)) + "\n"))
        else:
            result = emulator.run(max_steps=args.max_steps, stop_pc=args.stop_pc)
        if args.dump is not None:
            base, size = args.dump_region
            if base < 0 or base % 4 or size <= 0 or size % 4:
                raise ValueError("dump region must be nonempty and word aligned")
            payload = b"".join(struct.pack("<I", memory.read(addr)) for addr in range(base, base + size, 4))
            with args.dump.open("xb") as output:
                output.write(payload)
        print(json.dumps({**asdict(result), "flags": emulator.flags, "irq": emulator.irq,
                          "registers": emulator.registers, "vectors": emulator.vectors}))
    except (OSError, ValueError, ExecutionFault) as error:
        print(f"mgcemu: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
