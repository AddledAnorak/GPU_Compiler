"""Managed byte buffers and FIFO submissions with cooperative fences."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from enum import Enum

from .backends import ExecutionResult, MMIOBackend, SimulationBackend
from .errors import (
    DriverClosedError, ExecutionTimeoutError, FenceTimeoutError, HardwareBusyError,
    ResourceBusyError, ResourceError, SubmissionCancelledError, SubmissionError,
    ValidationError,
)
from .program import ADDRESS_LIMIT, Program, integer


def _seconds(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be finite seconds")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValidationError(f"{name} must be finite seconds") from error
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValidationError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} finite seconds")
    return value


class FenceStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


_TERMINAL = {FenceStatus.COMPLETED, FenceStatus.FAILED, FenceStatus.TIMED_OUT, FenceStatus.CANCELLED}


class Buffer:
    """An owned, aligned byte range; host access is forbidden while pinned."""

    def __init__(self, owner, identifier, address, size, alignment):
        self._owner = owner
        self._id = identifier
        self._address = address
        self._size = size
        self._alignment = alignment
        self._data = bytearray(size) if isinstance(owner.backend, SimulationBackend) else bytearray()
        self._pins = 0
        self._alive = True

    @property
    def id(self):
        return self._id

    @property
    def address(self):
        return self._address

    @property
    def size(self):
        return self._size

    @property
    def alignment(self):
        return self._alignment

    @property
    def legacy_word_address(self):
        return self._address // 4

    @property
    def alive(self):
        return self._alive

    @property
    def in_use(self):
        return bool(self._pins)

    def upload(self, data, *, offset=0):
        self._owner.upload(self, data, offset=offset)

    def download(self, *, offset=0, size=None):
        return self._owner.download(self, offset=offset, size=size)

    def close(self):
        self._owner.free(self)

    def __enter__(self):
        self._owner._buffer(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._alive:
            self.close()


class Fence:
    """Completion is distinct from retirement if hardware cannot be stopped."""

    def __init__(self, owner, identifier, program, buffers, entry, max_steps, execution_timeout):
        self._owner = owner
        self.id = identifier
        self._program = program
        self._buffers = buffers
        self._entry = entry
        self._max_steps = max_steps
        self._execution_timeout = execution_timeout
        self._status = FenceStatus.QUEUED
        self._result = None
        self._error = None
        self._retired = False
        self.submitted_at = time.monotonic()
        self.started_at = None
        self.finished_at = None

    @property
    def status(self):
        with self._owner._lock:
            return self._status

    @property
    def done(self):
        return self.status in _TERMINAL

    @property
    def retired(self):
        with self._owner._lock:
            return self._retired

    @property
    def error(self):
        with self._owner._lock:
            return self._error

    def poll(self):
        if self.done:
            return self.status
        self._owner.poll()
        return self.status

    def wait(self, timeout=None) -> ExecutionResult:
        if timeout is not None:
            timeout = _seconds(timeout, "timeout", allow_zero=True)
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.done:
            if deadline is not None and time.monotonic() >= deadline:
                raise FenceTimeoutError(f"fence {self.id} wait expired; submission remains live")
            self._owner.poll()
            if not self.done and isinstance(self._owner.backend, MMIOBackend):
                remaining = 0.001 if deadline is None else max(0, min(0.001, deadline - time.monotonic()))
                time.sleep(remaining)
        if self.error is not None:
            raise self.error
        return self._result

    def cancel(self):
        return self._owner.cancel(self)


class Driver:
    """Single queue, explicitly progressed by poll()/wait(); no background worker."""

    def __init__(self, backend=None):
        self.backend = SimulationBackend() if backend is None else backend
        if not isinstance(self.backend, (SimulationBackend, MMIOBackend)):
            raise ValidationError("backend must be SimulationBackend or MMIOBackend")
        if getattr(self.backend, "_driver_owner", None) is not None:
            raise ResourceError("a backend may be owned by only one live driver")
        self.backend._driver_owner = self
        self._lock = threading.RLock()
        self._buffers = {}
        self._queue = deque()
        self._active = None
        self._quarantined = None
        self._closed = False
        self._next_buffer = 1
        self._next_fence = 1

    @property
    def closed(self):
        return self._closed

    @property
    def allocated_bytes(self):
        with self._lock:
            return sum(buffer.size for buffer in self._buffers.values())

    @property
    def pending_count(self):
        with self._lock:
            return len(self._queue) + int(self._active is not None)

    @property
    def needs_recovery(self):
        return self._quarantined is not None

    def _open(self):
        if self._closed:
            raise DriverClosedError("driver is closed")
        if self._quarantined is not None:
            raise HardwareBusyError("hardware state is unsafe; resources stay pinned until recover()")

    def _buffer(self, buffer, *, host_access=False):
        self._open()
        if not isinstance(buffer, Buffer) or self._buffers.get(buffer.id) is not buffer:
            raise ResourceError("buffer is freed, foreign, or not owned by this driver")
        if host_access and buffer.in_use:
            raise ResourceBusyError(f"buffer {buffer.id} is pinned by a submission")
        return buffer

    def allocate(self, size, *, alignment=64, address=None):
        with self._lock:
            self._open()
            integer(size, "size", 1, self.backend.memory_size)
            integer(alignment, "alignment", 4, 1 << 31)
            if alignment & (alignment - 1):
                raise ValidationError("alignment must be a power of two")
            occupied = sorted(self._buffers.values(), key=lambda buffer: buffer.address)
            if address is None:
                address = (self.backend.memory_base + alignment - 1) & -alignment
                for buffer in occupied:
                    if address + size <= buffer.address:
                        break
                    if address < buffer.address + buffer.size:
                        address = (buffer.address + buffer.size + alignment - 1) & -alignment
            integer(address, "address", 0, ADDRESS_LIMIT - 1)
            if address % alignment:
                raise ValidationError("requested address does not satisfy alignment")
            if (address < self.backend.memory_base
                    or address + size > self.backend.memory_base + self.backend.memory_size):
                raise ResourceError("allocation does not fit the managed memory window")
            if any(address < buffer.address + buffer.size and buffer.address < address + size
                   for buffer in occupied):
                raise ResourceError("allocation overlaps a live buffer")
            buffer = Buffer(self, self._next_buffer, address, size, alignment)
            self.backend.write(buffer, 0, bytes(size))
            self._buffers[buffer.id] = buffer
            self._next_buffer += 1
            return buffer

    def free(self, buffer):
        with self._lock:
            buffer = self._buffer(buffer, host_access=True)
            self.backend.write(buffer, 0, bytes(buffer.size))
            del self._buffers[buffer.id]
            buffer._alive = False
            buffer._data.clear()

    @staticmethod
    def _range(buffer, offset, size):
        integer(offset, "offset", 0, buffer.size)
        integer(size, "size", 0, buffer.size)
        if offset + size > buffer.size:
            raise ResourceError("transfer exceeds buffer bounds")

    def upload(self, buffer, data, *, offset=0):
        with self._lock:
            buffer = self._buffer(buffer, host_access=True)
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise ValidationError("upload data must be bytes-like")
            data = bytes(data)
            self._range(buffer, offset, len(data))
            self.backend.write(buffer, offset, data)

    def download(self, buffer, *, offset=0, size=None):
        with self._lock:
            buffer = self._buffer(buffer, host_access=True)
            integer(offset, "offset", 0, buffer.size)
            size = buffer.size - offset if size is None else size
            self._range(buffer, offset, size)
            return self.backend.read(buffer, offset, size)

    def submit(self, program, *, buffers=(), entry=0, max_steps=1_000_000, execution_timeout=5.0):
        with self._lock:
            self._open()
            if not isinstance(program, Program):
                raise ValidationError("submit requires a validated, format-tagged Program")
            self.backend.validate(program)
            integer(entry, "entry", 0, len(program.words) - 1)
            integer(max_steps, "max_steps", 1, 100_000_000)
            execution_timeout = _seconds(execution_timeout, "execution_timeout")
            try:
                buffers = tuple(buffers)
            except TypeError as error:
                raise ResourceError("buffers must be an iterable of owned Buffer objects") from error
            for buffer in buffers:
                self._buffer(buffer)
            if len({buffer.id for buffer in buffers}) != len(buffers):
                raise ResourceError("submission resource list contains duplicate buffers")
            fence = Fence(self, self._next_fence, program, buffers, entry, max_steps, execution_timeout)
            self._next_fence += 1
            for buffer in buffers:
                buffer._pins += 1
            self._queue.append(fence)
            return fence

    def _retire(self, fence):
        if not fence._retired:
            for buffer in fence._buffers:
                buffer._pins -= 1
            fence._retired = True
            fence._buffers = ()
            fence._program = None

    def _complete(self, fence, status, error=None, result=None):
        fence._status = status
        fence._error = error
        fence._result = result
        fence.finished_at = time.monotonic()
        if self._active is fence:
            self._active = None
        safe = True
        if status != FenceStatus.COMPLETED:
            try:
                safe = self.backend.quiesce()
                if safe:
                    self.backend.finish(fence._buffers)
            except Exception:
                safe = False
        if safe:
            self._retire(fence)
        else:
            self._quarantined = fence

    def poll(self):
        with self._lock:
            self._open()
            if self._active is None:
                if not self._queue:
                    return None
                self._active = self._queue.popleft()
                self._active._status = FenceStatus.RUNNING
                self._active.started_at = time.monotonic()
                try:
                    self.backend.start(self._active._program, self._active._buffers,
                                       self._active._entry, self._active._max_steps)
                except Exception as error:
                    self._fail(self._active, error)
                    return None
            fence = self._active
            try:
                if time.monotonic() - fence.started_at >= fence._execution_timeout:
                    raise ExecutionTimeoutError("submission execution deadline expired")
                result = self.backend.poll()
                if result is not None:
                    self.backend.finish(fence._buffers)
                    self._complete(fence, FenceStatus.COMPLETED, result=result)
                elif time.monotonic() - fence.started_at >= fence._execution_timeout:
                    raise ExecutionTimeoutError("submission execution deadline expired")
                return result
            except Exception as error:
                self._fail(fence, error)
                return None

    def _fail(self, fence, error):
        if isinstance(error, ExecutionTimeoutError):
            status = FenceStatus.TIMED_OUT
        else:
            status = FenceStatus.FAILED
        if not isinstance(error, SubmissionError):
            wrapped = SubmissionError(f"{type(error).__name__}: {error}")
            wrapped.__cause__ = error
            error = wrapped
        self._complete(fence, status, error=error)

    def cancel(self, fence):
        with self._lock:
            if not isinstance(fence, Fence) or fence._owner is not self:
                raise ResourceError("fence is not owned by this driver")
            if fence.done:
                return False
            if self._closed:
                raise DriverClosedError("driver is closed")
            if fence is self._active:
                self._complete(fence, FenceStatus.CANCELLED,
                               error=SubmissionCancelledError(f"fence {fence.id} cancelled"))
            else:
                self._queue.remove(fence)
                fence._status = FenceStatus.CANCELLED
                fence._error = SubmissionCancelledError(f"fence {fence.id} cancelled")
                fence.finished_at = time.monotonic()
                self._retire(fence)
            return True

    def recover(self, *, external_reset_confirmed=False):
        """Retire quarantined resources only after a board-owned stop/reset proves idle."""
        with self._lock:
            if self._closed:
                raise DriverClosedError("driver is closed")
            if self._quarantined is None:
                return
            if getattr(self.backend, "_uncertain", False) and external_reset_confirmed is not True:
                raise HardwareBusyError("uncertain launch requires an externally confirmed stop/reset")
            if not self.backend.idle():
                raise HardwareBusyError("hardware still reports busy; recovery is not safe")
            fence = self._quarantined
            self.backend.finish(fence._buffers)
            self._retire(fence)
            self._quarantined = None

    def close(self):
        with self._lock:
            if self._closed:
                return
            for fence in tuple(self._queue):
                self.cancel(fence)
            if self._active is not None:
                self.cancel(self._active)
            if self._quarantined is not None:
                raise HardwareBusyError("cannot close: hardware resources are quarantined; recover first")
            for buffer in tuple(self._buffers.values()):
                self.free(buffer)
            self._closed = True
            self.backend._driver_owner = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
