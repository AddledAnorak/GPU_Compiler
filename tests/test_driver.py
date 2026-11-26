"""Driver-only tests: all MMIO addresses refer to an in-memory fake, never a board."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

from tools.mgcasm import assemble
from tools.mgcdriver import (
    Driver, DriverClosedError, ExecutionTimeoutError, FenceStatus, FenceTimeoutError,
    HardwareBusyError, HardwareSafetyError, HardwareSimulationBackend, MMIOBackend, MMIOConfig, Program,
    ProgramFormat, ProgramFormatError, ResourceBusyError, ResourceError,
    SimulationBackend, SubmissionCancelledError, SubmissionError, ValidationError,
    load_mem, parse_mem,
)


def legacy(source="stop_core"):
    return Program.from_words(assemble(source), format=ProgramFormat.LEGACY)


def hardware():
    return Program.from_words([0xF8000000], format=ProgramFormat.HARDWARE)


def reference(source):
    from tools.mgchw import assemble_hardware

    return Program.from_words(assemble_hardware(source), format=ProgramFormat.HARDWARE)


@pytest.fixture
def driver():
    with Driver(SimulationBackend(memory_size=4096, quantum=2)) as device:
        yield device


@pytest.mark.parametrize("format", list(ProgramFormat))
@pytest.mark.parametrize("byteorder", ["little", "big"])
def test_program_binary_roundtrip(format, byteorder, tmp_path):
    words = [0, 0x12345678, 0xFFFFFFFF]
    program = Program.from_words(words, format=format)
    words[0] = 17
    assert program.words == (0, 0x12345678, 0xFFFFFFFF)
    assert program.size_bytes == 12
    assert Program.from_binary(program.to_bytes(byteorder), format=format, byteorder=byteorder) == program
    path = tmp_path / "shader.bin"
    path.write_bytes(program.to_bytes(byteorder))
    assert Program.load(path, format=format, byteorder=byteorder) == program


@pytest.mark.parametrize("words", [[], [-1], [1 << 32], [True], [1.5], ["1234"]])
def test_program_invalid_words(words):
    with pytest.raises(ValidationError):
        Program.from_words(words, format=ProgramFormat.LEGACY)


@pytest.mark.parametrize("data", [b"", b"\x00", b"12345", "1234", 4])
def test_program_invalid_binary(data):
    with pytest.raises(ValidationError):
        Program.from_binary(data, format=ProgramFormat.LEGACY)


def test_program_format_is_required_and_known():
    with pytest.raises(TypeError):
        Program.from_words([0])
    with pytest.raises(ProgramFormatError):
        Program.from_words([0], format="auto")
    with pytest.raises(ValidationError):
        legacy().to_bytes("native")
    with pytest.raises(ValidationError):
        Program.from_binary(b"1234", format=ProgramFormat.LEGACY, byteorder="native")


def test_mem_comments_sparse_addresses_and_crlf(tmp_path):
    text = "// header\r\n00000001 01234567 # comment\n/* block\ncomment */ @4\nFFFFFFFF ; end\n"
    assert parse_mem(text) == (1, 0x1234567, 0, 0, 0xFFFFFFFF)
    path = tmp_path / "shader.mem"
    path.write_text(text)
    assert load_mem(path) == parse_mem(text)
    assert Program.load(path, format=ProgramFormat.HARDWARE).words == parse_mem(text)


def test_mem_comment_precedence_and_line_diagnostics():
    assert parse_mem("// /* not a block\n00000001 /* // inside block */\n") == (1,)
    with pytest.raises(ValidationError, match="line 4"):
        parse_mem("/* first\nsecond\nthird */\nbad\n")


def test_direct_program_constructor_is_bounded(monkeypatch):
    import tools.mgcdriver.program as module

    monkeypatch.setattr(module, "MAX_PROGRAM_WORDS", 3)
    consumed = []

    def words():
        for value in range(100):
            consumed.append(value)
            yield value

    with pytest.raises(ValidationError, match="exceeds word limit"):
        Program(words(), ProgramFormat.LEGACY)
    assert consumed == [0, 1, 2, 3]
    with pytest.raises(ValidationError):
        Program(None, ProgramFormat.LEGACY)


@pytest.mark.parametrize("text", [
    "", "// only comment", "0", "0x12345678", "gggggggg", "123456789", "0000000x",
    "00000001 @0 00000002", "@ffffffff 00000000", "00000001 @2", "/* missing", "*/",
    "@-1 00000000", "@1g 00000000", "@100000000 00000000",
])
def test_mem_rejects_malformed_or_unsafe_images(text):
    with pytest.raises(ValidationError):
        parse_mem(text)


def test_mem_loading_limits_and_kind(tmp_path):
    with pytest.raises(ValidationError):
        parse_mem("00000001 00000002", max_words=1)
    with pytest.raises(ValidationError):
        parse_mem("@2 00000001", max_words=2)
    with pytest.raises(ValidationError):
        parse_mem(" " * 4200, max_words=1)
    path = tmp_path / "unknown.shader"
    path.write_bytes(b"\x00" * 4)
    with pytest.raises(ValidationError):
        Program.load(path, format=ProgramFormat.LEGACY)
    assert Program.load(path, format=ProgramFormat.LEGACY, kind="binary").words == (0,)


def test_allocation_alignment_fixed_addresses_reuse_and_zeroing(driver):
    first = driver.allocate(7)
    second = driver.allocate(4, alignment=128)
    assert first.address == 0 and second.address == 128
    assert first.size == 7 and second.legacy_word_address == 32
    assert first.download() == bytes(7)
    first.upload(b"abcdefg")
    first.close()
    assert not first.alive
    replacement = driver.allocate(7)
    assert replacement.address == 0 and replacement.download() == bytes(7)
    fixed = driver.allocate(4, alignment=4, address=1000)
    assert fixed.address == 1000
    assert driver.allocated_bytes == 15


@pytest.mark.parametrize("size,alignment", [(0, 4), (-1, 4), (True, 4), (3.5, 4), (1, 0), (1, 2), (1, 12), (1, True)])
def test_allocation_invalid_parameters(driver, size, alignment):
    with pytest.raises(ValidationError):
        driver.allocate(size, alignment=alignment)


def test_allocation_bounds_overlap_and_exhaustion(driver):
    driver.allocate(4096, alignment=4)
    with pytest.raises(ResourceError):
        driver.allocate(1)
    with pytest.raises(ResourceError):
        driver.allocate(4, alignment=4, address=4)
    with pytest.raises(ValidationError):
        driver.allocate(1, address=1)
    with Driver(SimulationBackend(memory_base=16, memory_size=16)) as offset_driver:
        with pytest.raises(ResourceError):
            offset_driver.allocate(4, alignment=4, address=0)
        assert offset_driver.allocate(16, alignment=4).address == 16


def test_upload_download_are_byte_addressed_and_bounded(driver):
    buffer = driver.allocate(8)
    buffer.upload(memoryview(b"abc"), offset=1)
    assert buffer.download(offset=1, size=3) == b"abc"
    assert buffer.download() == b"\x00abc\x00\x00\x00\x00"
    buffer.upload(b"", offset=8)
    assert buffer.download(offset=8) == b""
    with pytest.raises(ResourceError):
        buffer.upload(b"12", offset=7)
    with pytest.raises(ResourceError):
        buffer.download(offset=7, size=2)
    with pytest.raises(ValidationError):
        buffer.upload("abc")
    with pytest.raises(ValidationError):
        buffer.download(offset=-1)


def test_gpu_word_must_fit_fully_inside_a_buffer(driver):
    buffer = driver.allocate(3)
    fence = driver.submit(legacy("movi_i32 0, r1\nload r1, r2\nstop_core"), buffers=[buffer])
    with pytest.raises(SubmissionError, match="outside submitted"):
        fence.wait(1)
    assert fence.retired and buffer.download() == bytes(3)


def test_legacy_vector_stores_use_contiguous_byte_words(driver):
    buffer = driver.allocate(16)
    source = """
        movi_i32 7, r1
        movi_i32 0, r2
        vbcast r1, v0
        v4cmp_ge_zero v0
        vstore_mask v0, r2
        stop_core
    """
    driver.submit(legacy(source), buffers=[buffer]).wait(1)
    assert buffer.download() == (7).to_bytes(4, "little") * 4


def test_foreign_freed_buffer_and_context_lifetime(driver):
    with Driver() as other:
        foreign = other.allocate(4)
        with pytest.raises(ResourceError):
            driver.upload(foreign, b"1234")
        with pytest.raises(ResourceError):
            driver.submit(legacy(), buffers=[foreign])
    with driver.allocate(4) as buffer:
        assert buffer.alive
    assert not buffer.alive
    with pytest.raises(ResourceError):
        buffer.download()
    with pytest.raises(ResourceError):
        buffer.close()


def test_submission_reads_writes_the_same_managed_bytes(driver):
    buffer = driver.allocate(12, alignment=4, address=256)
    buffer.upload((21).to_bytes(4, "little"))
    program = legacy(f"""
        movi_i32 {buffer.legacy_word_address}, r1
        load r1, r2
        wait_load
        add r2, r2, r3
        wait_short
        movi_i32 {buffer.legacy_word_address + 1}, r4
        store r3, r4
        stop_core
    """)
    fence = driver.submit(program, buffers=[buffer])
    assert fence.status == FenceStatus.QUEUED and buffer.in_use
    result = fence.wait(1)
    assert fence.status == FenceStatus.COMPLETED and fence.retired
    assert not buffer.in_use and result.registers[3] == 42
    assert int.from_bytes(buffer.download(offset=4, size=4), "little") == 42
    assert result.steps == len(program.words)


def test_queue_shared_buffer_pins_fifo_and_wait_on_later(driver):
    buffer = driver.allocate(4)
    first = driver.submit(legacy("movi_i32 17, r1\nstop_core"), buffers=[buffer])
    second = driver.submit(legacy("movi_i32 23, r1\nstop_core"), buffers=[buffer])
    assert driver.pending_count == 2
    first.wait(1)
    assert buffer.in_use and second.status == FenceStatus.QUEUED
    for operation in (lambda: buffer.upload(b"1234"), lambda: buffer.download(), buffer.close):
        with pytest.raises(ResourceBusyError):
            operation()
    assert second.wait(1).registers[1] == 23
    assert not buffer.in_use and driver.pending_count == 0


def test_host_wait_timeout_does_not_cancel_or_unpin(driver):
    buffer = driver.allocate(4)
    fence = driver.submit(legacy(), buffers=[buffer])
    with pytest.raises(FenceTimeoutError):
        fence.wait(0)
    assert fence.status == FenceStatus.QUEUED and buffer.in_use
    assert fence.wait(1).steps == 1


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True, "1", 10 ** 1000])
def test_invalid_wait_timeout(driver, timeout):
    fence = driver.submit(legacy())
    with pytest.raises(ValidationError):
        fence.wait(timeout)


def test_instruction_budget_terminal_timeout_releases_resources(driver):
    buffer = driver.allocate(4)
    fence = driver.submit(legacy("loop: goto loop"), buffers=[buffer], max_steps=3)
    assert fence.poll() == FenceStatus.RUNNING
    with pytest.raises(ExecutionTimeoutError, match="max_steps=3"):
        fence.wait(1)
    assert fence.status == FenceStatus.TIMED_OUT and fence.retired and not buffer.in_use
    assert driver.submit(legacy(), max_steps=1).wait(1).steps == 1


def test_execution_wall_deadline_releases_simulation_resources(driver):
    buffer = driver.allocate(4)
    fence = driver.submit(legacy("loop: goto loop"), buffers=[buffer], execution_timeout=1e-12)
    with pytest.raises(ExecutionTimeoutError, match="deadline"):
        fence.wait(1)
    assert fence.retired and not buffer.in_use


@pytest.mark.parametrize("source,match", [
    ("movi_i32 1, r1", "program counter"),
    ("movi_f16 2.0, r1\nfmul16 r1, r1, r2\ncopy r2, r3\nstop_core", "LatencyViolation"),
    ("movi_i32 0, r1\nload r1, r2\nwait_load\nstop_core", "outside submitted"),
])
def test_execution_errors_are_reported_on_fence(driver, source, match):
    fence = driver.submit(legacy(source))
    with pytest.raises(SubmissionError, match=match):
        fence.wait(1)
    assert fence.status == FenceStatus.FAILED and fence.retired
    assert fence.error is not None


def test_shader_cannot_access_live_but_unsubmitted_buffer(driver):
    buffer = driver.allocate(4)
    buffer.upload(b"abcd")
    fence = driver.submit(legacy("movi_i32 0, r1\nload r1, r2\nstop_core"))
    with pytest.raises(SubmissionError, match="outside submitted"):
        fence.wait(1)
    assert buffer.download() == b"abcd"


def test_cancellation_of_queued_and_running_submissions(driver):
    buffer = driver.allocate(4)
    running = driver.submit(legacy("loop: goto loop"), buffers=[buffer])
    queued = driver.submit(legacy(), buffers=[buffer])
    running.poll()
    assert queued.cancel() and queued.retired
    assert buffer.in_use
    assert running.cancel() and running.retired
    assert not buffer.in_use and not running.cancel()
    with pytest.raises(SubmissionCancelledError):
        running.wait(1)


def test_close_cancels_pending_invalidates_buffers_and_preserves_fence_results():
    driver = Driver()
    buffer = driver.allocate(4)
    completed = driver.submit(legacy())
    completed.wait(1)
    pending = driver.submit(legacy(), buffers=[buffer])
    driver.close()
    driver.close()
    assert driver.closed and not buffer.alive and pending.retired
    assert completed.poll() == FenceStatus.COMPLETED
    assert completed.wait(0).steps == 1
    with pytest.raises(SubmissionCancelledError):
        pending.wait(0)
    with pytest.raises(DriverClosedError):
        driver.allocate(4)
    with pytest.raises(DriverClosedError):
        buffer.download()


def test_backend_exclusive_ownership():
    backend = SimulationBackend()
    driver = Driver(backend)
    with pytest.raises(ResourceError):
        Driver(backend)
    driver.close()
    with Driver(backend):
        pass


@pytest.mark.parametrize("keyword,value", [
    ("entry", -1), ("entry", 1), ("entry", True), ("max_steps", 0),
    ("max_steps", True), ("max_steps", 100_000_001), ("execution_timeout", 0),
    ("execution_timeout", float("inf")), ("execution_timeout", True),
])
def test_submission_validation_before_pinning(driver, keyword, value):
    buffer = driver.allocate(4)
    with pytest.raises(ValidationError):
        driver.submit(legacy(), buffers=[buffer], **{keyword: value})
    assert not buffer.in_use and driver.pending_count == 0


def test_submission_type_format_duplicates_and_foreign_fence(driver):
    buffer = driver.allocate(4)
    with pytest.raises(ProgramFormatError):
        driver.submit(hardware(), buffers=[buffer])
    with pytest.raises(ValidationError):
        driver.submit([0])
    with pytest.raises(ResourceError):
        driver.submit(legacy(), buffers=[buffer, buffer])
    with Driver() as other:
        with pytest.raises(ResourceError):
            driver.cancel(other.submit(legacy()))
    assert not buffer.in_use


@pytest.mark.parametrize("parameters", [
    {"memory_base": -1}, {"memory_base": 1}, {"memory_size": 0},
    {"memory_base": (1 << 32) - 4, "memory_size": 8},
    {"quantum": 0}, {"quantum": 4097}, {"quantum": True},
])
def test_simulation_configuration_validation(parameters):
    with pytest.raises(ValidationError):
        SimulationBackend(**parameters)


def test_hardware_reference_uses_byte_addresses_and_managed_memory():
    with Driver(HardwareSimulationBackend(memory_size=4096, quantum=2)) as driver:
        buffer = driver.allocate(8, alignment=4, address=256)
        buffer.upload((21).to_bytes(4, "little"))
        program = reference(f"""
            movi_i32 {buffer.address}, r1
            load r1, r2
            wait_load
            add r2, r2, r3
            movi_i32 {buffer.address + 4}, r4
            store r3, r4
        """)
        result = driver.submit(program, buffers=[buffer], max_steps=len(program.words)).wait(1)
        assert result.backend == "hardware-reference-simulation"
        assert result.steps == len(program.words) and len(result.registers) == 32
        assert result.registers[3] == 42
        assert int.from_bytes(buffer.download(offset=4, size=4), "little") == 42
        assert driver.backend._cpu is None


def test_hardware_reference_sparse_sample_control_range_is_explicit():
    with Driver(HardwareSimulationBackend(memory_size=0x90000004)) as driver:
        control = driver.allocate(4, alignment=4, address=0x90000000)
        program = reference("movi_u16 1, r1\nmovi_i32 2415919104, r2\nstore r1, r2")
        driver.submit(program, buffers=[control]).wait(1)
        assert control.download() == (1).to_bytes(4, "little")
        assert driver.allocated_bytes == 4


def test_hardware_reference_rejects_legacy_and_has_no_default_mappings():
    with Driver(HardwareSimulationBackend()) as driver:
        with pytest.raises(ProgramFormatError):
            driver.submit(legacy())
        buffer = driver.allocate(4)
        fence = driver.submit(reference("movi_u16 0, r1\nload r1, r2"))
        with pytest.raises(SubmissionError, match="MemoryAccessError"):
            fence.wait(1)
        assert fence.retired and buffer.download() == bytes(4)


def test_hardware_reference_fifo_jobs_share_original_buffer_bytes():
    with Driver(HardwareSimulationBackend(quantum=2)) as driver:
        buffer = driver.allocate(4)
        first = driver.submit(reference("movi_u16 7, r1\nstore r1, r0"), buffers=[buffer])
        second = driver.submit(reference("load r0, r1\nadd r1, r1, r2\nstore r2, r0"), buffers=[buffer])
        assert second.wait(1).registers[2] == 14 and first.retired
        assert int.from_bytes(buffer.download(), "little") == 14


def test_hardware_reference_simd_bounds_are_checked_before_any_lane_store():
    with Driver(HardwareSimulationBackend()) as driver:
        buffer = driver.allocate(12)
        source = """
            movi_u16 0, r0
            movi_u16 4, r1
            vset4 r0, r1, v0
            movi_u16 42, r2
            movi_u16 15, r3
            vstore_mask_r r2, v0, r3
        """
        fence = driver.submit(reference(source), buffers=[buffer])
        with pytest.raises(SubmissionError, match="MemoryAccessError"):
            fence.wait(1)
        assert buffer.download() == bytes(12) and fence.retired


@pytest.mark.parametrize("budget", [1, 3, 10])
def test_hardware_reference_cycle_budget_and_cancellation(budget):
    with Driver(HardwareSimulationBackend(quantum=2)) as driver:
        buffer = driver.allocate(4)
        fence = driver.submit(reference("forever: jump forever\nnop\nnop"), buffers=[buffer], max_steps=budget)
        with pytest.raises(ExecutionTimeoutError):
            fence.wait(1)
        assert fence.status == FenceStatus.TIMED_OUT and fence.retired and not buffer.in_use
        running = driver.submit(reference("forever: jump forever"), buffers=[buffer])
        assert running.poll() == FenceStatus.RUNNING
        running.cancel()
        assert running.retired and not buffer.in_use


def test_hardware_reference_eof_at_exact_budget_and_entry_offset():
    with Driver(HardwareSimulationBackend()) as driver:
        program = reference("copy r0, r1\ncopy r0, r2")
        assert driver.submit(program, entry=1, max_steps=1).wait(1).steps == 1


def test_hardware_reference_approximation_is_off_unless_explicitly_enabled():
    program = reference("movi_f16 1.0, r0\nvbcast r0, v0\nvrgb565_ceramic v0, v1")
    with Driver(HardwareSimulationBackend()) as driver:
        with pytest.raises(SubmissionError, match="allow_approximate_fp16"):
            driver.submit(program).wait(1)
    with Driver(HardwareSimulationBackend(allow_approximate_fp16=True)) as driver:
        assert driver.submit(program).wait(1).backend == "hardware-reference-simulation"
    with pytest.raises(ValidationError):
        HardwareSimulationBackend(allow_approximate_fp16=1)


def test_hardware_reference_bad_encoding_fails_fence_and_releases_resources():
    with Driver(HardwareSimulationBackend()) as driver:
        buffer = driver.allocate(4)
        program = Program.from_words([0xFFFFFFFF], format=ProgramFormat.HARDWARE)
        fence = driver.submit(program, buffers=[buffer])
        with pytest.raises(SubmissionError, match="HardwareEncodingError"):
            fence.wait(1)
        assert fence.status == FenceStatus.FAILED and fence.retired and not buffer.in_use


def config(**changes):
    values = dict(memory_base=0x1000, memory_size=0x1000, program_base=0x8000,
                  program_capacity=0x1000, control_offset=0, status_offset=4,
                  entry_offset=8, word_count_offset=12, start_value=1,
                  busy_mask=1, done_mask=2, error_mask=4,
                  acknowledge_offset=16, acknowledge_value=7)
    values.update(changes)
    return MMIOConfig(**values)


class FakeMMIO:
    def __init__(self, configuration, *, complete_after=1):
        self.config = configuration
        self.events = []
        self.memory = {}
        self.status = 0
        self.complete_after = complete_after
        self.polls = 0
        self.started = False
        self.clear_ack = True
        self.accept_start = True
        self.abort_works = True
        self.fail_start = False

    def read32(self, offset):
        self.events.append(("read32", offset))
        if self.started:
            self.polls += 1
            if self.complete_after is not None and self.polls >= self.complete_after:
                self.status = self.config.done_mask
        return self.status

    def write32(self, offset, value):
        self.events.append(("write32", offset, value))
        if offset == self.config.acknowledge_offset and self.clear_ack:
            self.status = 0
        if offset == self.config.control_offset:
            self.started = True
            self.status = self.config.busy_mask if self.accept_start else 0
            if self.fail_start:
                raise OSError("uncertain write")
        if offset == self.config.abort_offset and self.abort_works:
            self.status = 0
            self.started = False

    def write(self, address, data):
        self.events.append(("write", address, bytes(data)))
        self.memory.update({address + offset: byte for offset, byte in enumerate(data)})

    def read(self, address, size):
        self.events.append(("read", address, size))
        return bytes(self.memory.get(address + offset, 0) for offset in range(size))

    def sync_for_device(self, address, size):
        self.events.append(("sync_for_device", address, size))

    def sync_for_cpu(self, address, size):
        self.events.append(("sync_for_cpu", address, size))

    def barrier(self):
        self.events.append(("barrier",))


@pytest.mark.parametrize("field,value", [
    ("memory_base", 1), ("program_base", 1), ("memory_size", 0),
    ("program_capacity", 5), ("busy_mask", 0), ("done_mask", 0),
    ("error_mask", 1), ("done_mask", 1), ("byteorder", "native"),
    ("program_base", 0x1800), ("memory_base", (1 << 32) - 4),
    ("entry_offset", 0), ("start_value", 0), ("status_offset", True),
    ("abort_offset", 20), ("abort_value", 1),
])
def test_mmio_invalid_configuration(field, value):
    with pytest.raises(ValidationError):
        config(**{field: value})


def test_mmio_requires_explicit_opt_in_and_complete_transport():
    transport = FakeMMIO(config())
    with pytest.raises(HardwareSafetyError):
        MMIOBackend(config(), transport)
    assert transport.events == []
    with pytest.raises(ValidationError):
        MMIOBackend(config(), object(), allow_hardware=True)


def test_mmio_submission_order_data_lifetime_and_completion():
    configuration = config(byteorder="big")
    transport = FakeMMIO(configuration, complete_after=2)
    with Driver(MMIOBackend(configuration, transport, allow_hardware=True)) as driver:
        buffer = driver.allocate(8)
        buffer.upload(b"abcdefgh")
        program = Program.from_words([1, 2], format=ProgramFormat.HARDWARE)
        fence = driver.submit(program, buffers=[buffer], entry=1)
        result = fence.wait(1)
        assert result.backend == "configured-mmio" and result.registers is None
        assert fence.retired and buffer.download() == b"abcdefgh"
        assert ("write32", configuration.entry_offset, 4) in transport.events
        assert ("write32", configuration.word_count_offset, 2) in transport.events
        upload = transport.events.index(("write", configuration.program_base, program.to_bytes("big")))
        kick = transport.events.index(("write32", configuration.control_offset, configuration.start_value))
        assert upload < kick
        assert ("sync_for_device", configuration.program_base, 8) in transport.events[upload:kick]
        assert ("barrier",) in transport.events[upload:kick]
        assert ("sync_for_cpu", buffer.address, buffer.size) in transport.events[kick:]


def test_mmio_format_capacity_and_busy_are_checked_before_io():
    configuration = config(program_capacity=4)
    transport = FakeMMIO(configuration)
    driver = Driver(MMIOBackend(configuration, transport, allow_hardware=True))
    with pytest.raises(ProgramFormatError):
        driver.submit(legacy())
    with pytest.raises(ValidationError):
        driver.submit(Program.from_words([0, 0], format=ProgramFormat.HARDWARE))
    assert transport.events == []
    transport.status = configuration.busy_mask
    with pytest.raises(HardwareSafetyError):
        driver.allocate(4)
    assert not any(event[0] == "write" for event in transport.events)
    transport.status = 0
    driver.close()


def test_mmio_stale_done_is_not_misreported_as_this_submission():
    configuration = config()
    transport = FakeMMIO(configuration, complete_after=None)
    transport.status = configuration.done_mask
    transport.clear_ack = False
    with Driver(MMIOBackend(configuration, transport, allow_hardware=True)) as driver:
        fence = driver.submit(hardware())
        with pytest.raises(SubmissionError, match="acknowledgement"):
            fence.wait(1)
        assert fence.status == FenceStatus.FAILED and fence.retired
        assert not any(event[0] == "write" for event in transport.events)


def test_mmio_timeout_quarantines_until_external_stop():
    configuration = config()
    transport = FakeMMIO(configuration, complete_after=None)
    driver = Driver(MMIOBackend(configuration, transport, allow_hardware=True))
    buffer = driver.allocate(4)
    fence = driver.submit(hardware(), buffers=[buffer], execution_timeout=1e-12)
    with pytest.raises(ExecutionTimeoutError):
        fence.wait(1)
    assert fence.done and not fence.retired and buffer.in_use and driver.needs_recovery
    with pytest.raises(HardwareBusyError):
        buffer.close()
    with pytest.raises(HardwareBusyError):
        driver.recover()
    with pytest.raises(HardwareBusyError):
        driver.close()
    assert not driver.closed and buffer.alive
    transport.started = False
    transport.status = 0
    driver.recover()
    assert fence.retired and not buffer.in_use and not driver.needs_recovery
    driver.close()


def test_mmio_configured_abort_allows_safe_cancellation():
    configuration = config(abort_offset=20, abort_value=99)
    transport = FakeMMIO(configuration, complete_after=None)
    with Driver(MMIOBackend(configuration, transport, allow_hardware=True)) as driver:
        buffer = driver.allocate(4)
        fence = driver.submit(hardware(), buffers=[buffer])
        assert fence.poll() == FenceStatus.RUNNING
        assert fence.cancel() and fence.retired and not buffer.in_use
        assert ("write32", 20, 99) in transport.events


def test_mmio_host_wait_timeout_does_not_abort_or_release():
    configuration = config(abort_offset=20, abort_value=99)
    transport = FakeMMIO(configuration, complete_after=None)
    with Driver(MMIOBackend(configuration, transport, allow_hardware=True)) as driver:
        buffer = driver.allocate(4)
        fence = driver.submit(hardware(), buffers=[buffer])
        fence.poll()
        with pytest.raises(FenceTimeoutError):
            fence.wait(0)
        assert fence.status == FenceStatus.RUNNING and buffer.in_use
        assert ("write32", 20, 99) not in transport.events
        fence.cancel()


def test_mmio_short_reads_and_cache_failure_preserve_lifetime(monkeypatch):
    configuration = config()
    transport = FakeMMIO(configuration, complete_after=1)
    driver = Driver(MMIOBackend(configuration, transport, allow_hardware=True))
    buffer = driver.allocate(4)
    monkeypatch.setattr(transport, "read", lambda address, size: b"")
    with pytest.raises(SubmissionError, match="short"):
        buffer.download()

    def fail_sync(address, size):
        raise OSError("cache maintenance unavailable")

    monkeypatch.setattr(transport, "sync_for_cpu", fail_sync)
    fence = driver.submit(hardware(), buffers=[buffer])
    with pytest.raises(SubmissionError, match="cache maintenance"):
        fence.wait(1)
    assert driver.needs_recovery and buffer.in_use
    with pytest.raises(OSError, match="cache maintenance"):
        driver.recover()
    assert not fence.retired
    monkeypatch.setattr(transport, "sync_for_cpu", lambda address, size: None)
    driver.recover()
    driver.close()


@pytest.mark.parametrize("fail_start", [False, True])
def test_mmio_unacknowledged_or_failed_launch_is_not_assumed_safe(fail_start):
    configuration = config()
    transport = FakeMMIO(configuration, complete_after=None)
    transport.accept_start = False
    transport.fail_start = fail_start
    driver = Driver(MMIOBackend(configuration, transport, allow_hardware=True))
    buffer = driver.allocate(4)
    fence = driver.submit(hardware(), buffers=[buffer])
    with pytest.raises(SubmissionError):
        fence.wait(1)
    assert driver.needs_recovery and buffer.in_use and not fence.retired
    with pytest.raises(HardwareBusyError, match="externally confirmed"):
        driver.recover()
    transport.started = False
    driver.recover(external_reset_confirmed=True)
    driver.close()


def test_mmio_error_status_is_fence_failure_and_remains_pinned_if_busy():
    configuration = config(abort_offset=20, abort_value=1)
    transport = FakeMMIO(configuration, complete_after=None)
    transport.abort_works = False
    driver = Driver(MMIOBackend(configuration, transport, allow_hardware=True))
    buffer = driver.allocate(4)
    fence = driver.submit(hardware(), buffers=[buffer])
    fence.poll()
    transport.status = configuration.busy_mask | configuration.error_mask
    with pytest.raises(SubmissionError, match="status indicates error"):
        fence.wait(1)
    assert not fence.retired and driver.needs_recovery
    transport.started = False
    transport.status = 0
    driver.recover()
    driver.close()


def test_cli_valid_submission_mem_and_binary(tmp_path):
    program = legacy("movi_i32 42, r1\nstop_core")
    for extension, contents in (("mem", "\n".join(f"{word:08x}" for word in program.words).encode()),
                                ("bin", program.to_bytes())):
        path = tmp_path / f"shader.{extension}"
        path.write_bytes(contents)
        completed = run_cli(path, "--format", ProgramFormat.LEGACY.value)
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report["status"] == "completed" and report["registers"][1] == 42


def run_cli(*arguments):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "tools"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "mgcdriver", *map(str, arguments)],
                          cwd=ROOT, env=environment, capture_output=True, text=True, timeout=10)


def test_cli_data_upload_and_download(tmp_path):
    path = tmp_path / "shader.bin"
    path.write_bytes(legacy("movi_i32 7, r1\nmovi_i32 0, r2\nstore r1, r2\nstop_core").to_bytes())
    data = tmp_path / "data.mem"
    data.write_text("12345678\n")
    output = tmp_path / "result.bin"
    completed = run_cli(path, "--format", ProgramFormat.LEGACY.value, "--data", data, "--output", output)
    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == (7).to_bytes(4, "little")


@pytest.mark.parametrize("failure", ["missing", "format", "malformed", "loop", "zero-wait"])
def test_cli_reports_errors_and_has_no_hardware_access(tmp_path, failure):
    path = tmp_path / "shader.mem"
    program = legacy("loop: goto loop") if failure == "loop" else legacy()
    if failure != "missing":
        path.write_text("bad" if failure == "malformed" else "\n".join(f"{word:08x}" for word in program.words))
    format = ProgramFormat.HARDWARE if failure == "format" else ProgramFormat.LEGACY
    arguments = [path, "--format", format.value, "--max-steps", "3"]
    if failure == "zero-wait":
        arguments += ["--wait-timeout", "0"]
    completed = run_cli(*arguments)
    assert completed.returncode == 1 and json.loads(completed.stderr)["status"] == "error"


def test_cli_requires_format_and_rejects_hardware_option(tmp_path):
    assert run_cli(tmp_path / "shader.bin").returncode == 2
    assert run_cli("--hardware").returncode == 2


def test_cli_direct_script_entry_point(tmp_path):
    path = tmp_path / "shader.bin"
    path.write_bytes(legacy().to_bytes())
    completed = subprocess.run([sys.executable, str(ROOT / "tools/run_submission.py"), str(path),
                                "--format", ProgramFormat.LEGACY.value], cwd=tmp_path,
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "completed"


def test_canonical_imports_work_without_other_tests_or_tools_pythonpath():
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    code = """
from tools.mgcasm import assemble
from tools.mgchw import assemble_hardware
from tools.mgcdriver import Driver, HardwareSimulationBackend, Program
with Driver() as driver:
    program = Program.from_words(assemble('stop_core'), format='legacy-mgcsim-v1')
    assert driver.submit(program).wait(1).steps == 1
with Driver(HardwareSimulationBackend()) as driver:
    program = Program.from_words(assemble_hardware('copy r0, r1'), format='fp16-hardware-v1')
    assert driver.submit(program).wait(1).steps == 1
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr


def test_cli_hardware_reference_is_explicit_and_can_write_output(tmp_path):
    program = reference("movi_u16 42, r1\nstore r1, r0")
    path = tmp_path / "hardware.bin"
    path.write_bytes(program.to_bytes())
    output = tmp_path / "result.bin"
    completed = run_cli(path, "--format", ProgramFormat.HARDWARE.value,
                        "--backend", "hardware-reference", "--buffer-size", "4", "--output", output)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["backend"] == "hardware-reference-simulation"
    assert output.read_bytes() == (42).to_bytes(4, "little")
    rejected = run_cli(path, "--format", ProgramFormat.HARDWARE.value)
    assert rejected.returncode == 1 and json.loads(rejected.stderr)["error"] == "ProgramFormatError"


def test_imported_hardware_images_load_but_never_execute_as_legacy(driver):
    samples = ROOT / "hardware-examples/aakash-2026-09-28"
    programs = sorted(samples.glob("*.program.mem"))
    if not programs:
        pytest.skip("hardware examples are supplied by the parent task")
    for path in programs:
        program = Program.load(path, format=ProgramFormat.HARDWARE)
        assert program.size_bytes > 4
        with pytest.raises(ProgramFormatError):
            driver.submit(program)
