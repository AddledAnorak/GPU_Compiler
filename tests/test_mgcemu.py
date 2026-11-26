import ctypes
import json
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from tools.mgcemu import AssumptionRequired, Emulator, ExecutionFault, Memory, StepLimit, TimingViolation, decode_word
from tools.mgcemu.__main__ import main, read_mem
from tools.mgcreference.raster import render
from tools.mgcreference.scene import load_scene

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "hardware-examples/aakash-2026-09-28"
BUCKY = EXAMPLES / "bucky_12frame_dualbuf_clean"
GEARS = EXAMPLES / "mesa_gears_hd"
UTAH = EXAMPLES / "utah_12frame_dualbuf_shader"


@pytest.mark.usefixtures("local_hardware_examples")
def test_original_program_forms_are_modeled_and_utah_requires_opt_in():
    forms = {decode_word(word).form for path in EXAMPLES.glob("*.program.mem")
             for word in read_mem(path)}
    assert len(forms) == 27 and all(form.modeled for form in forms)
    word = read_mem(UTAH.with_suffix(".program.mem"))[417]
    assert word == 0x80051C25
    emulator = Emulator([word], allow=("sample-vector",))
    with pytest.raises(AssumptionRequired, match="sample-vector-fp16"):
        emulator.step()
    assert (emulator.pc, emulator.steps) == (0, 0)
    emulator = Emulator([word], allow=("sample-vector-fp16",))
    emulator.vectors[5] = (0x3C00, 0x3C00, 0xBC00, 0x8000)
    emulator.vectors[1] = (0x1600, 0xBC00, 0x3800, 0)
    assert emulator.step().vector_changes == ((5, (0x3C02, 0, 0xB800, 0)),)


@pytest.mark.usefixtures("local_hardware_examples")
def test_branch_rules_are_explicit_for_the_two_observed_selectors():
    sample = read_mem(BUCKY.with_suffix(".program.mem"))[37]
    assert sample == 0xAC3FDFFB
    current = sample & ~(15 << 10)
    program = [0] * 37 + [sample, current]
    emulator = Emulator(program)
    emulator.pc, emulator.flags = 37, 1
    with pytest.raises(AssumptionRequired, match="sample-next-pc") as caught:
        emulator.step()
    assert (caught.value.pc, caught.value.word) == (37, sample)
    emulator.flags = 0
    assert emulator.step().next_pc == 38
    emulator.flags = 1
    assert emulator.step().next_pc == 33
    emulator = Emulator(program, allow=("sample-next-pc",))
    emulator.pc, emulator.flags = 37, 1
    trace = emulator.step()
    assert (trace.operation, trace.next_pc, trace.assumptions) == ("jump_next", 33, ("sample-next-pc",))
    assert emulator.assumptions_used == {"sample-next-pc"}


def test_prefix_consumption_and_predicates_are_visible_in_trace():
    emulator = Emulator([0x00100610, 0x80210043, 0x00100610, 0x84210043, 0x80210043, 0xC0000000])
    emulator.registers[1:3] = [2, 3]
    emulator.registers[33:35] = [10, 20]
    trace = []
    stopped = emulator.run(max_steps=6, trace=trace.append)
    assert (stopped.reason, stopped.steps, emulator.irq) == ("stop_core", 6, True)
    assert trace[0].bank_after == 7
    assert trace[1].register_changes == ((35, 30),)
    assert not trace[3].executed and trace[3].bank_after == 0
    assert trace[4].register_changes == ((3, 5),)


def test_optional_numeric_assumptions_do_not_hide_undefined_casts():
    emulator = Emulator([0x20210043], allow=("negative-fp16-sign",))
    emulator.registers[1:3] = [0xBC00, 0x4000]
    assert emulator.step().register_changes == ((3, 0x3C00),)
    assert emulator.assumptions_used == {"negative-fp16-sign"}
    strict = Emulator([0x20210043])
    strict.registers[1:3] = [0x8000, 0]
    with pytest.raises(AssumptionRequired, match="negative-fp16-sign"):
        strict.step()
    strict.registers[1:3] = [0x7E00, 0]
    with pytest.raises(AssumptionRequired, match="arithmetic-nan"):
        strict.step()
    strict = Emulator([0x58210003], allow=("arithmetic-nan",))
    strict.registers[1] = 0x7C00
    with pytest.raises(ExecutionFault, match="conversion .* is undefined"):
        strict.step()


def test_scalar_fp16_follows_c_numeric_rules():
    emulator = Emulator([0x20210043, 0x30200004, 0x18210025])
    emulator.registers[1:3] = [0x3C00, 0x1600]
    assert emulator.step().register_changes == ((3, 0x3C01),)
    assert emulator.step().register_changes == ((4, 0x7C00),)
    emulator.registers[1] = 0x8001
    assert emulator.step().register_changes == ((5, 0x8000),)
    assert not emulator.assumptions_used


def test_scalar_add_override_is_explicit_and_does_not_change_subtract():
    emulator = Emulator([0x20210043, 0x20210443], allow=("sample-fadd-rne",))
    emulator.registers[1:3] = [0x3C00, 0x1600]
    assert emulator.step().register_changes == ((3, 0x3C02),)
    subtraction = emulator.step()
    assert subtraction.register_changes == ((3, 0x3BFD),)
    assert subtraction.assumptions == ()
    assert emulator.assumptions_used == {"sample-fadd-rne"}


@pytest.mark.parametrize("fp16_scale, green", [(False, 0), (True, 0x20)])
@pytest.mark.usefixtures("local_hardware_examples")
def test_sample_rgb565_packing_modes_include_clamping(fp16_scale, green):
    word = read_mem(UTAH.with_suffix(".program.mem"))[419]
    assert word == 0x80052C07
    allow = ["sample-vector-fp16"]
    if fp16_scale:
        allow.append("sample-rgb565-fp16-scale")
    emulator = Emulator([word], allow=allow)
    emulator.vectors[5] = (0x2410, 0x3800, 0x7C00, 0xBC00)
    assert emulator.step().vector_changes == ((7, (green, 0x7BEF, 0xFFFF, 0)),)
    emulator = Emulator([word], allow=allow)
    emulator.vectors[5] = (0, 0x7E00, 0, 0)
    with pytest.raises(ExecutionFault, match="NaN RGB565 color"):
        emulator.step()
    assert emulator.pc == 0 and not emulator.assumptions_used


def test_masked_store_is_preflighted_and_ignores_inactive_bad_lanes():
    memory = Memory()
    memory.map(0x100, 8)
    emulator = Emulator([0xC8240F20], memory=memory, allow=("sample-vector",))
    emulator.registers[4] = 0xDADA
    emulator.registers[25] = 0b0101
    emulator.vectors[0] = (0x100, 0x105, 0x108, 0x10C)
    with pytest.raises(ExecutionFault, match="unmapped byte address 0x108"):
        emulator.step()
    assert memory.read(0x100) == 0
    assert not emulator.assumptions_used
    emulator.registers[25] = 0b0011
    emulator.vectors[0] = (0x100, 0x100, 0x108, 0x10C)
    with pytest.raises(ExecutionFault, match="colliding active lanes"):
        emulator.step()
    emulator.registers[25] = 0b0001
    assert emulator.step().memory_writes == ((0x100, 0xDADA),)


def test_sample_vector_store_uses_distinct_raw_lane_values_and_byte_addresses():
    memory = Memory()
    memory.map(0x100, 16)
    emulator = Emulator([0xC8071720], memory=memory, allow=("sample-vector",))
    emulator.vectors[0] = (0x100, 0x104, 0x108, 0x10C)
    emulator.vectors[7] = (0x11223344, 0xFFFF4321, 0xCAFEBABE, 0x89ABCDEF)
    emulator.registers[25] = 0b1010
    assert emulator.step().memory_writes == ((0x104, 0xFFFF4321), (0x10C, 0x89ABCDEF))
    assert [memory.read(address) for address in range(0x100, 0x110, 4)] == [0, 0xFFFF4321, 0, 0x89ABCDEF]


def test_pause_is_not_halt_and_bad_fetch_and_budget_are_faults():
    emulator = Emulator([0, 0])
    assert emulator.run(max_steps=1, stop_pc=1).reason == "inspection_pc"
    assert not emulator.stopped and not emulator.irq
    with pytest.raises(ExecutionFault, match="fetch outside program"):
        emulator.run(max_steps=2)
    looping = Emulator([0xA8000000])
    with pytest.raises(StepLimit, match=r"instruction limit \(3\)"):
        looping.run(max_steps=3)
    assert (looping.pc, looping.steps) == (0, 3)
    unknown = Emulator([0xB8000000 | (7 << 10)])
    with pytest.raises(ExecutionFault, match="selector 7"):
        unknown.step()


def test_cli_writes_trace_and_dump(tmp_path, capsys):
    program = tmp_path / "prog.mem"
    trace = tmp_path / "trace.jsonl"
    dump = tmp_path / "dump.bin"
    program.write_text("080048c0\n08000021\n08004000\n08000022\na0210040\nc0000000\n")
    assert main([str(program), "--max-steps", "6", "--map", "0x100:4", "--trace", str(trace),
                 "--dump", str(dump), "--dump-region", "0x100:4"]) == 0
    result = json.loads(capsys.readouterr().out)
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert (result["reason"], result["steps"], result["irq"]) == ("stop_core", 6, True)
    assert events[-2]["memory_writes"] == [[0x100, 0x123]]
    assert dump.read_bytes() == struct.pack("<I", 0x123)


@pytest.mark.parametrize("nframes", [6, 12])
@pytest.mark.usefixtures("local_hardware_examples")
def test_original_bucky_frame_zero_from_reset(nframes):
    bucky = EXAMPLES / f"bucky_{nframes}frame_dualbuf_clean"
    memory = Memory()
    memory.map_words(0, [0xDEADBEEF] * 0x20000, writable=True)
    memory.map_words(0x100000, read_mem(bucky.with_suffix(".data.mem")))
    memory.map(0x90000000, 4)
    emulator = Emulator(read_mem(bucky.with_suffix(".program.mem")), memory=memory,
                        allow=("sample-next-pc", "sample-vector"), timing_check=True)
    result = emulator.run(max_steps=2_000_000, stop_pc=329)
    reference = render(load_scene(f"bucky{nframes}")[0], renderer="scalar")
    frame = b"".join(struct.pack("<I", memory.read(address)) for address in range(0x40000, 0x80000, 4))
    assert (result.reason, result.pc, result.steps) == ("inspection_pc", 329, 1_973_860)
    assert result.assumptions_used == ("sample-next-pc", "sample-vector")
    assert not emulator.stopped and not emulator.irq
    assert memory.read(0x90000000) == 1
    assert all(memory.read(address) == 0 for address in range(0, 0x40000, 4))
    assert reference.triangle_count == 526 and frame == reference.framebuffer_bytes
    assert emulator.registers[26] == 0x100004 + 526 * 13 * 4


@pytest.mark.usefixtures("local_hardware_examples")
def test_original_gears_frame_zero_from_reset():
    memory = Memory()
    memory.map_words(0, [0xDEADBEEF] * 0x20000, writable=True)
    memory.map_words(0x100000, read_mem(GEARS.with_suffix(".data.mem")))
    memory.map_words(0x90000000, [0xDEADBEEF], writable=True)
    emulator = Emulator(read_mem(GEARS.with_suffix(".program.mem")), memory=memory,
                        allow=("sample-next-pc", "sample-vector"), timing_check=True)
    result = emulator.run(max_steps=5_000_000, stop_pc=581)
    frame = b"".join(struct.pack("<I", memory.read(address)) for address in range(0, 0x40000, 4))
    reference = render(load_scene("gears")[0])
    assert reference.triangle_count == 4858 and frame == reference.framebuffer_bytes
    assert (result.reason, result.pc, result.steps) == ("inspection_pc", 581, 4_575_359)
    assert memory.read(0x90000000) == 0


@pytest.mark.usefixtures("local_hardware_examples")
def test_original_utah_frame_zero_from_verified_frame_entry():
    memory = Memory()
    memory.map(0, 0x80000)
    memory.map_words(0x100000, read_mem(UTAH.with_suffix(".data.mem")))
    memory.map(0x90000000, 4)
    allow = ("sample-fadd-rne", "sample-next-pc", "sample-rgb565-fp16-scale",
             "sample-vector", "sample-vector-fp16")
    emulator = Emulator(read_mem(UTAH.with_suffix(".program.mem")), memory=memory, allow=allow, timing_check=True)
    emulator.pc = 123
    result = emulator.run(max_steps=3_000_000, stop_pc=501)
    frame = b"".join(struct.pack("<I", memory.read(address)) for address in range(0x40000, 0x80000, 4))
    reference = render(load_scene("utah")[0])
    assert reference.triangle_count == 2345 and frame == reference.framebuffer_bytes
    assert (result.reason, result.pc, result.steps) == ("inspection_pc", 501, 2_188_028)
    assert result.assumptions_used == allow
    assert memory.read(0x90000000) == 1
    assert not emulator.stopped and not emulator.irq


@pytest.mark.usefixtures("local_hardware_examples")
def test_utah_sample_add_pair_fails_with_one_wait_word_removed():
    pair = read_mem(UTAH.with_suffix(".program.mem"))[340:348]
    assert pair == [0x20200003, 0, 0, 0, 0, 0, 0, 0x20230063]
    original = Emulator(pair + [0xC0000000], timing_check=True)
    original.registers[0] = 0x3400
    assert original.run(max_steps=9).reason == "stop_core"
    assert original.registers[3] == 0x3C00
    shortened = Emulator(pair[:6] + pair[7:] + [0xC0000000], timing_check=True)
    shortened.registers[0] = 0x3400
    with pytest.raises(TimingViolation, match=r"fadd16 reads r3 .* 1 more intervening"):
        shortened.run(max_steps=8)


READ = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
WRITE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)


class CState(ctypes.Structure):
    _fields_ = [
        ("regs", ctypes.c_uint32 * 64), ("cr", ctypes.c_uint8),
        ("vregs", (ctypes.c_uint32 * 4) * 8), ("pc", ctypes.c_uint32),
        ("pc_prev", ctypes.c_uint32), ("imm16_l", ctypes.c_uint16),
        ("pending_reg_bank", ctypes.c_uint8), ("stop_core", ctypes.c_bool),
        ("irq", ctypes.c_bool), ("trace_enabled", ctypes.c_bool),
        ("instruction_count", ctypes.c_uint64), ("cycle_count", ctypes.c_uint64),
        ("mem_read", READ), ("mem_write", WRITE), ("user_ctx", ctypes.c_void_p),
    ]


@pytest.fixture(scope="module")
def supplied_c(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("the optional supplied-C comparison requires a C compiler")
    library = tmp_path_factory.mktemp("mgcemu-c") / "interpreter.so"
    interpreter = ROOT / "interpreter"
    subprocess.run([compiler, "-std=c11", "-O2", "-fPIC", "-shared", f"-I{interpreter / 'include'}",
                    str(interpreter / "src/mgc_interpreter.c"), str(interpreter / "src/mgc_fp16_math.c"),
                    "-lm", "-o", str(library)], check=True)
    native = ctypes.CDLL(str(library))
    native.mgc_init.argtypes = [ctypes.POINTER(CState), READ, WRITE, ctypes.c_void_p]
    native.mgc_init.restype = None
    native.mgc_step.argtypes = [ctypes.POINTER(CState)]
    native.mgc_step.restype = ctypes.c_bool
    return native


def _word(opcode, selector, operands):
    return opcode << 27 | selector << 10 | operands


C_WORDS = (
    [0, 0x00100610, 0x080048C0, 0x08000023]
    + [_word(2, selector, 0x210003) for selector in range(4)]
    + [0x18210023, 0x20210043, 0x20210443, 0x28210043, 0x30210043]
    + [_word(7, selector, 0x210043) for selector in range(2)]
    + [_word(7, selector, 0x210003) for selector in range(2, 4)]
    + [_word(8, selector, 0x210003) for selector in range(4)]
    + [_word(9, selector, 0x2100C3) for selector in range(4)]
    + [_word(10, selector, 0x2100C3) for selector in range(2)]
    + [_word(11, 0, 0x210003)]
    + [_word(12, selector, 0xC3) for selector in range(4)]
    + [_word(13, selector, 0xC3) for selector in range(2)]
    + [_word(14, 0, 0xA3)]
    + [_word(15, selector, 0x210043) for selector in range(8)]
    + [_word(16, selector, 0x250043) for selector in range(3)]
    + [0x80210C43, 0x80211003, 0x80011443, 0x98000083, 0xA0210080, 0xA8000001,
       0xB0000040, 0xB8000080, 0xC0000000]
    + [_word(26, selector, 0x250043) for selector in range(6)]
)


@pytest.mark.parametrize("word", C_WORDS, ids=lambda word: f"{word:08x}")
def test_canonical_instruction_effect_against_supplied_c(supplied_c, word):
    baseline = [0, 0x4100, 0x3E00, 0xDEADBEEF, 0x100, 0xFFFFFFFF, 0x11223344] + [0] * 57
    vectors = [(0, 1, 2, 3), (0xFFFFFFFF, 5, 6, 7)] + [(2, 3, 4, 5)] * 6
    writes = []
    reads = []

    @READ
    def native_read(context, address):
        if address == 0:
            return word
        reads.append((address, 0x10293847))
        return 0x10293847

    @WRITE
    def native_write(context, address, value):
        writes.append((address, value))

    state = CState()
    supplied_c.mgc_init(ctypes.byref(state), native_read, native_write, None)
    state.regs[:] = baseline
    for register, lanes in zip(state.vregs, vectors):
        register[:] = lanes
    state.cr, state.imm16_l = 0xA5, 0xABCD
    supplied_c.mgc_step(ctypes.byref(state))

    memory = Memory()
    memory.map_words(0x100, [0x10293847], writable=True)
    emulator = Emulator([word], memory=memory)
    emulator.registers[:] = baseline
    emulator.vectors[:] = vectors
    emulator.flags, emulator.low_immediate = 0xA5, 0xABCD
    trace = emulator.step()
    assert emulator.registers == list(state.regs)
    assert emulator.vectors == [tuple(vector) for vector in state.vregs]
    assert (emulator.pc, emulator.flags, emulator.low_immediate, emulator.bank, emulator.irq, emulator.stopped) == (
        state.pc, state.cr, state.imm16_l, state.pending_reg_bank, state.irq, state.stop_core,
    )
    assert (trace.memory_reads, trace.memory_writes) == (tuple(reads), tuple(writes))
