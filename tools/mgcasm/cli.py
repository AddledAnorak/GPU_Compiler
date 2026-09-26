"""Command-line front end: mgcasm assemble|disassemble <file>."""

from __future__ import annotations

import argparse
import struct
import sys

from .assembler import assemble
from .disassembler import disassemble


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mgcasm")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("assemble", help="assembly text -> binary image")
    pa.add_argument("infile")
    pa.add_argument("-o", "--outfile", default=None, help="default: stdout as hex")

    pd = sub.add_parser("disassemble", help="binary image -> assembly text")
    pd.add_argument("infile")

    args = p.parse_args(argv)

    if args.cmd == "assemble":
        with open(args.infile) as f:
            words = assemble(f.read())
        if args.outfile:
            with open(args.outfile, "wb") as f:
                for w in words:
                    f.write(struct.pack("<I", w))
        else:
            for w in words:
                print(f"{w:#010x}")
        return 0

    if args.cmd == "disassemble":
        with open(args.infile, "rb") as f:
            data = f.read()
        words = [w[0] for w in struct.iter_unpack("<I", data)]
        for line in disassemble(words):
            print(line)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
