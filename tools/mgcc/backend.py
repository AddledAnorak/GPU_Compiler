"""Bounded, explicit managed execution of the unmodified supplied C interpreter."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from tools.mgcdriver import ExecutionResult, ProgramFormat, SimulationBackend
from tools.mgcdriver.errors import ExecutionTimeoutError, ProgramFormatError, ResourceError, SubmissionError
from tools.mgcdriver.program import integer
from .encoding import CEncodingError, MAX_WORDS, disassemble_c

ROOT = Path(__file__).resolve().parents[2]
PINNED = {
    "include/mgc_fp16_isa.h": "4bd6ce19016a7471db24f473c2aafdb28257b6e19b9407afe8e949a37c0022e5",
    "include/mgc_interpreter.h": "8a4c52ab7bddec80a41489533d7311807f10a9346642b4ec21c298cfe771c33e",
    "src/mgc_interpreter.c": "8d3d04ff3364d88b86f108f93402847c35fe95f0ab762b12105ac1a98a96c847",
    "src/mgc_fp16_math.c": "51a31619ee040e27b5bda3bc5610557b6c670fb1c976b4848ce864f7ef0a5d2d",
}


class NativeBuffer(ctypes.Structure):
    _fields_ = [("address", ctypes.c_uint32), ("size", ctypes.c_uint32),
                ("data", ctypes.POINTER(ctypes.c_ubyte))]


class CInterpreterBackend(SimulationBackend):
    format = ProgramFormat.C_INTERPRETER
    name = "supplied-c-interpreter"

    def __init__(self, *, memory_base=0x40000, memory_size=5 << 20, quantum=256):
        super().__init__(memory_base=memory_base, memory_size=memory_size, quantum=quantum)
        self._temporary = tempfile.TemporaryDirectory(prefix="mgcc-native-")
        self._handle = None
        try:
            folder = Path(self._temporary.name)
            hashes = {}
            for relative in PINNED:
                source = ROOT / "interpreter" / relative
                contents = source.read_bytes()
                digest = hashlib.sha256(contents).hexdigest()
                if relative in PINNED and digest != PINNED[relative]:
                    raise SubmissionError(f"supplied C contract changed: {relative}; re-audit before execution")
                hashes[relative] = digest
                destination = folder / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents)
            shim = Path(__file__).with_name("native.c")
            shutil.copyfile(shim, folder / "native.c")
            library = folder / "libmgcc.so"
            command = [os.environ.get("CC", "cc"), "-std=c11", "-O0", "-g", "-fPIC", "-shared",
                       "-Wall", "-Wextra", "-Werror", "-I", str(folder / "include"),
                       str(folder / "native.c"), str(folder / "src/mgc_interpreter.c"),
                       str(folder / "src/mgc_fp16_math.c"), "-lm", "-o", str(library)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if completed.returncode:
                raise SubmissionError(f"fresh supplied-C build failed: {completed.stdout}{completed.stderr}")
            self.build_evidence = {"command": command, "source_sha256": hashes,
                                   "shim_sha256": hashlib.sha256(shim.read_bytes()).hexdigest(),
                                   "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                                   "original_sources_modified": False,
                                   "negative_fp16_signed_shift_issue_unresolved": True}
            self._library = ctypes.CDLL(str(library))
            signatures = {
                "mgcc_create": ([ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                                 ctypes.POINTER(NativeBuffer), ctypes.c_uint32,
                                 ctypes.c_uint32, ctypes.c_uint64], ctypes.c_void_p),
                "mgcc_poll": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
                "mgcc_steps": ([ctypes.c_void_p], ctypes.c_uint64),
                "mgcc_register": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_uint32),
                "mgcc_fault_address": ([ctypes.c_void_p], ctypes.c_uint32),
                "mgcc_destroy": ([ctypes.c_void_p], None),
            }
            for name, (arguments, result) in signatures.items():
                function = getattr(self._library, name)
                function.argtypes, function.restype = arguments, result
        except Exception:
            self._temporary.cleanup()
            raise

    def validate(self, program):
        if program.format != self.format:
            raise ProgramFormatError(f"{self.name} requires {self.format.value}")
        try:
            disassemble_c(program.words)
        except CEncodingError as error:
            raise ProgramFormatError(f"invalid current C machine code: {error}") from error

    def start(self, program, buffers, entry, max_steps):
        self.validate(program)
        integer(max_steps, "max_steps", 1, (1 << 63) - 1)
        if self._handle:
            raise SubmissionError("C interpreter already has an active submission")
        if len(program.words) > MAX_WORDS or len(buffers) > 4096:
            raise SubmissionError("C submission exceeds program/resource limits")
        if any(buffer.address < len(program.words) * 4 for buffer in buffers):
            raise ResourceError("C program fetch region overlaps a submitted data resource")
        self._program_array = (ctypes.c_uint32 * len(program.words))(*program.words)
        self._views = [(ctypes.c_ubyte * buffer.size).from_buffer(buffer._data) for buffer in buffers]
        self._buffer_array = (NativeBuffer * len(buffers))(*[
            NativeBuffer(buffer.address, buffer.size, view) for buffer, view in zip(buffers, self._views)
        ])
        self._handle = self._library.mgcc_create(self._program_array, len(program.words),
                                                self._buffer_array, len(buffers), entry, max_steps)
        if not self._handle:
            self.quiesce()
            raise SubmissionError("C interpreter rejected execution setup")

    def poll(self):
        if not self._handle:
            raise SubmissionError("no active C submission")
        status = self._library.mgcc_poll(self._handle, self.quantum)
        if status == 0:
            return None
        if status == 1:
            return ExecutionResult(self.name, self._library.mgcc_steps(self._handle),
                                   tuple(self._library.mgcc_register(self._handle, index) for index in range(64)))
        if status == -1:
            address = self._library.mgcc_fault_address(self._handle)
            raise ResourceError(f"C shader access at {address:#x} is unaligned, read-only program memory, or outside submitted resources")
        if status == -3:
            raise ExecutionTimeoutError("C submission exceeded its instruction budget")
        if status == -4:
            raise SubmissionError("C float-to-integer exceptional input has no defined portable result")
        raise SubmissionError("C program escaped its loaded instruction image without stop_core" if status == -2 else "C execution failed")

    def quiesce(self):
        if self._handle:
            self._library.mgcc_destroy(self._handle)
            self._handle = None
        self._views = []
        self._buffer_array = None
        self._program_array = None
        return True

    def finish(self, buffers):
        self.quiesce()

    def close(self):
        self.quiesce()
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
