"""Bounded 32-bit program and Verilog-style memory image loading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .errors import ProgramFormatError, ValidationError

MAX_PROGRAM_WORDS = 1_048_576
ADDRESS_LIMIT = 1 << 32


class ProgramFormat(str, Enum):
    LEGACY = "legacy-mgcsim-v1"
    HARDWARE = "fp16-hardware-v1"
    C_INTERPRETER = "c-interpreter-v1"


def integer(value: int, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValidationError(f"{name} must be <= {maximum}")
    return value


def _byteorder(value: str) -> str:
    if value not in ("little", "big"):
        raise ValidationError("byteorder must be 'little' or 'big'")
    return value


def parse_mem(text: str, *, max_words: int = MAX_PROGRAM_WORDS) -> tuple[int, ...]:
    """Parse eight-hex-digit words; @ addresses are word indices, not bytes."""
    integer(max_words, "max_words", 1, MAX_PROGRAM_WORDS)
    if not isinstance(text, str):
        raise ValidationError("memory image must be text")
    if len(text) > max_words * 32 + 4096:
        raise ValidationError("memory image text exceeds loading limit")
    def strip_comment(match):
        comment = match.group()
        if comment in ("/*", "*/"):
            raise ValidationError("unterminated or unmatched memory image block comment")
        return " " + "\n" * comment.count("\n")

    text = re.sub(r"//[^\n]*|#[^\n]*|;[^\n]*|/\*.*?\*/|/\*|\*/", strip_comment, text, flags=re.DOTALL)
    words: list[int] = []
    cursor = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        for token in line.split():
            if re.fullmatch(r"@[0-9a-fA-F]{1,8}", token):
                address = int(token[1:], 16)
                if address < cursor or address >= max_words:
                    raise ValidationError(f"line {line_number}: backward or oversized @ address")
                cursor = address
                continue
            if not re.fullmatch(r"[0-9a-fA-F]{8}", token):
                raise ValidationError(f"line {line_number}: expected a 32-bit hex word, got {token!r}")
            if cursor >= max_words:
                raise ValidationError("memory image exceeds max_words")
            words.extend([0] * (cursor - len(words)))
            words.append(int(token, 16))
            cursor += 1
    if not words:
        raise ValidationError("memory image contains no words")
    if cursor != len(words):
        raise ValidationError("trailing @ address has no data")
    return tuple(words)


def load_mem(path: str | Path, *, max_words: int = MAX_PROGRAM_WORDS) -> tuple[int, ...]:
    integer(max_words, "max_words", 1, MAX_PROGRAM_WORDS)
    with Path(path).open("r", encoding="ascii") as stream:
        text = stream.read(max_words * 32 + 4097)
    return parse_mem(text, max_words=max_words)


@dataclass(frozen=True)
class Program:
    words: tuple[int, ...]
    format: ProgramFormat

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "format", ProgramFormat(self.format))
        except (ValueError, TypeError) as error:
            raise ProgramFormatError(f"unknown program format: {self.format!r}") from error
        words = []
        try:
            for word in self.words:
                if len(words) == MAX_PROGRAM_WORDS:
                    raise ValidationError("program exceeds word limit")
                integer(word, "program word", 0, ADDRESS_LIMIT - 1)
                words.append(word)
        except TypeError as error:
            raise ValidationError("program words must be iterable") from error
        if not words or len(words) > MAX_PROGRAM_WORDS:
            raise ValidationError("program must contain 1..1048576 words")
        object.__setattr__(self, "words", tuple(words))

    @classmethod
    def from_words(cls, words: Iterable[int], *, format: ProgramFormat | str) -> Program:
        return cls(words, format)

    @classmethod
    def from_binary(
        cls, data: bytes, *, format: ProgramFormat | str, byteorder: str = "little"
    ) -> Program:
        _byteorder(byteorder)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValidationError("program binary must be bytes-like")
        data = bytes(data)
        if not data or len(data) % 4 or len(data) > MAX_PROGRAM_WORDS * 4:
            raise ValidationError("program binary must contain complete bounded 32-bit words")
        return cls(tuple(int.from_bytes(data[offset:offset + 4], byteorder)
                         for offset in range(0, len(data), 4)), format)

    @classmethod
    def from_mem(cls, text: str, *, format: ProgramFormat | str) -> Program:
        return cls(parse_mem(text), format)

    @classmethod
    def load(
        cls, path: str | Path, *, format: ProgramFormat | str,
        byteorder: str = "little", kind: str | None = None,
    ) -> Program:
        path = Path(path)
        kind = kind or ({".mem": "mem", ".bin": "binary"}.get(path.suffix.lower()))
        if kind == "mem":
            return cls(load_mem(path), format)
        if kind != "binary":
            raise ValidationError("use .mem/.bin or specify kind='mem'/'binary'")
        with path.open("rb") as stream:
            data = stream.read(MAX_PROGRAM_WORDS * 4 + 1)
        return cls.from_binary(data, format=format, byteorder=byteorder)

    @property
    def size_bytes(self) -> int:
        return len(self.words) * 4

    def to_bytes(self, byteorder: str = "little") -> bytes:
        _byteorder(byteorder)
        return b"".join(word.to_bytes(4, byteorder) for word in self.words)
