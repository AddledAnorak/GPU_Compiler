"""Explicitly mapped, aligned word data access using byte addresses."""

from dataclasses import dataclass

from .isa import WORD_MASK


class MemoryFault(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Region:
    base: int
    end: int
    writable: bool


class Memory:
    def __init__(self):
        self._regions: list[Region] = []
        self._words: dict[int, int] = {}

    def map(self, base: int, size: int, *, writable: bool = True) -> None:
        if (type(base) is not int or type(size) is not int or base < 0 or size <= 0
                or base % 4 or size % 4 or base + size > WORD_MASK + 1):
            raise MemoryFault("region must be aligned and fit in the 32-bit address space")
        if any(base < region.end and region.base < base + size for region in self._regions):
            raise MemoryFault("memory regions overlap")
        self._regions.append(Region(base, base + size, writable))

    def map_words(self, base, words, *, writable: bool = False) -> None:
        words = tuple(words)
        if any(type(word) is not int or not 0 <= word <= WORD_MASK for word in words):
            raise MemoryFault("initial data must contain unsigned 32-bit words")
        self.map(base, len(words) * 4, writable=writable)
        self._words.update((base + index * 4, word) for index, word in enumerate(words) if word)

    def _check(self, address, write=False):
        if type(address) is not int or address % 4:
            raise MemoryFault(f"unaligned word address {address!r}")
        for region in self._regions:
            if region.base <= address < region.end:
                if write and not region.writable:
                    raise MemoryFault(f"write to read-only byte address 0x{address:x}")
                return
        raise MemoryFault(f"unmapped byte address 0x{address:x}")

    def read(self, address: int) -> int:
        self._check(address)
        return self._words.get(address, 0)

    def write(self, address: int, value: int) -> None:
        self._check(address, True)
        if type(value) is not int or not 0 <= value <= WORD_MASK:
            raise MemoryFault("stored data must be an unsigned 32-bit word")
        self._words[address] = value

    def write_lanes(self, writes: tuple[tuple[int, int], ...]) -> None:
        if len({address for address, _ in writes}) != len(writes):
            raise MemoryFault("masked store has colliding active lanes")
        for address, value in writes:
            self._check(address, True)
            if type(value) is not int or not 0 <= value <= WORD_MASK:
                raise MemoryFault("stored data must be an unsigned 32-bit word")
        self._words.update(writes)
