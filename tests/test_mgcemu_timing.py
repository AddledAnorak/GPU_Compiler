import json

import pytest

from tools.mgcemu import Emulator, Memory, TimingViolation, assemble
from tools.mgcemu.__main__ import main


def timed_program(source, *, timing_check=True, allow=()):
    memory = Memory()
    memory.map_words(0x100, [0x3C00])
    emulator = Emulator(assemble(source), memory=memory, allow=allow, timing_check=timing_check)
    emulator.registers[1:3] = [0x100, 0x3800]
    return emulator


@pytest.mark.parametrize("producer, consumer, wait, slots, resource", [
    ("fadd16 #3, r2, r3", "copy r3, r4", "wait_fadd", 6, "r3"),
    ("fsub16 #3, r2, r3", "copy r3, r4", "wait_fadd", 6, "r3"),
    ("fmul16 #3, r2, r3", "copy r3, r4", "wait_fmul", 8, "r3"),
    ("fdiv16 #3, r2, r3", "copy r3, r4", "wait_fdiv", 8, "r3"),
    ("u16tof16_l r2, r3", "copy r3, r4", "wait_conv", 3, "r3"),
    ("f16toi32 #3, r3", "copy r3, r4", "wait_conv", 3, "r3"),
    ("load r1, r3", "copy r3, r4", "wait_load", 4, "r3"),
    ("fcomp_l #3, r2, cr5", "[cr5] copy #2, r4", "wait_fcomp", 3, "cr5"),
])
def test_documented_spacing_boundary(producer, consumer, wait, slots, resource):
    name = producer.split()[0]
    for gap in (0, slots - 1):
        emulator = timed_program(f"{producer}\n" + "nop\n" * gap + f"{consumer}\nstop_core")
        trace = []
        with pytest.raises(TimingViolation, match=rf"reads {resource} too early after {name}.*{wait}") as caught:
            emulator.run(max_steps=slots + 3, trace=trace.append)
        error = caught.value
        assert (error.resource, error.producer_pc, error.pc, error.missing_slots) == (
            resource, 0, gap + 1, slots - gap,
        )
        assert emulator.steps == error.pc == len(trace)
        assert emulator.registers[4] == 0 and not emulator.irq
    emulator = timed_program(f"{producer}\n{wait}\n{consumer}\nstop_core")
    assert emulator.run(max_steps=slots + 3).reason == "stop_core"


def test_wait_is_optional_in_functional_mode():
    emulator = timed_program("fadd16 #3, r2, r3\ncopy r3, r4\nstop_core", timing_check=False)
    assert emulator.run(max_steps=3).steps == 3
    assert emulator.registers[4] == 0x3E00


def test_independent_work_and_skipped_predicate_fill_load_spacing():
    emulator = timed_program("""
        load r1, r3
        [cr0] store r3, r3
        copy #3, r5
        copy r2, r6
        add r5, r6, r7
        copy r3, r4
        stop_core
    """)
    trace = []
    assert emulator.run(max_steps=7, trace=trace.append).reason == "stop_core"
    assert not trace[1].executed and not trace[1].memory_writes
    assert emulator.registers[4:8] == [0x3C00, 0x3C00, 0x3800, 0x7400]


def test_banked_sources_and_destinations_track_physical_registers():
    emulator = timed_program("""
        load r1, r35
        copy r3, r4
        copy #3, r5
        add #2, r35, r6
        stop_core
    """)
    emulator.registers[3] = 71
    with pytest.raises(TimingViolation, match=r"add reads r35 .*load at pc=1; 1 more") as caught:
        emulator.run(max_steps=10)
    assert caught.value.pc == 5
    assert emulator.registers[4:7] == [71, 0x3C00, 0]


def test_vector_operands_do_not_alias_scalar_registers():
    emulator = timed_program("load r1, r3\nvget_lane0_i32 v3, r4\nstop_core", allow=("sample-vector",))
    emulator.vectors[3] = (51, 52, 53, 54)
    assert emulator.run(max_steps=3).reason == "stop_core"
    assert emulator.registers[4] == 51


def test_branch_cannot_count_wait_words_it_skips():
    emulator = timed_program("""
        fadd16 #3, r2, r3
        jump consume
        wait_fadd
    consume: copy r3, r4
        stop_core
    """)
    with pytest.raises(TimingViolation) as caught:
        emulator.run(max_steps=10)
    assert (emulator.steps, caught.value.pc, caught.value.missing_slots) == (2, 8, 5)


def test_a_timing_fault_precedes_branch_choice_and_store_side_effects():
    emulator = timed_program("""
        fcomp_l #3, r2, cr0
        [cr0] jump end
    end: stop_core
    """)
    with pytest.raises(TimingViolation, match="jump reads cr0"):
        emulator.run(max_steps=3)
    assert emulator.flags == 0 and not emulator.stopped
    emulator = timed_program("load r1, r3\nstore #3, r3\nstop_core")
    emulator.memory.map(0x3C00, 4)
    with pytest.raises(TimingViolation, match="store reads r3"):
        emulator.run(max_steps=3)
    assert emulator.memory.read(0x3C00) == 0


def test_overwrites_replace_old_producers_and_skipped_writes_do_not():
    emulator = timed_program("""
        fmul16 #3, r2, r3
        copy #4, r3
        copy r3, r4
        fcomp_l #3, r2, cr0
        icmp_eq r2, r2, cr0
        [cr0] stop_core
    """)
    assert emulator.run(max_steps=6).reason == "stop_core"
    assert emulator.registers[4] == 0x3800
    skipped = timed_program("load r1, r3\n[cr0] copy #4, r3\ncopy r3, r4\nstop_core")
    with pytest.raises(TimingViolation, match=r"load at pc=0; 3 more"):
        skipped.run(max_steps=4)


def test_inspection_pause_keeps_pending_results_and_reset_clears_them():
    emulator = timed_program("load r1, r3\ncopy r3, r4\nstop_core")
    assert emulator.run(max_steps=1, stop_pc=1).reason == "inspection_pc"
    with pytest.raises(TimingViolation, match=r"load at pc=0; 4 more"):
        emulator.step()
    emulator.reset()
    emulator.pc = 1
    assert emulator.run(max_steps=2).reason == "stop_core"


def test_cli_timing_check_is_opt_in_and_reports_both_instruction_locations(tmp_path, capsys):
    source = tmp_path / "demo.asm"
    source.write_text("fmul16 #3, r2, r3\ncopy r3, r4\nstop_core\n")
    assert main([str(source), "--max-steps", "12"]) == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "stop_core"
    args = [str(source), "--max-steps", "12", "--timing-check"]
    assert main(args) == 1
    result = capsys.readouterr()
    assert not result.out
    assert "pc=1," in result.err and "fmul16 at pc=0; 8 more" in result.err
    source.write_text("fmul16 #3, r2, r3\nwait_fmul\ncopy r3, r4\nstop_core\n")
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["steps"] == 11
