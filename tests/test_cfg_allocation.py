"""Fresh direct-NIR CFG register reuse, parallel phi copies and live-pressure limits."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest

from tools.host_render import FragmentStatus, ShaderExecutor, pack_float
from tools.mgcdriver import ProgramFormat
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
SOURCE = ROOT / "tests/fixtures/cfg-allocation/mgc_cfg_allocation.c"


def checked(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def cfg_backend(tmp_path_factory):
    if not (BUILD / "build.ninja").is_file() or not (BUILD / "src/gallium/drivers/mgc/libmgc.a").is_file():
        pytest.skip("already-built Mesa required for fresh CFG allocation tests")
    ninja = shutil.which("ninja") or str(ROOT / ".venv/bin/ninja")
    directory = tmp_path_factory.mktemp("cfg-allocation-helper")
    target = "src/gallium/drivers/mgc/mgc_test_trig"
    original_object = target + ".p/mgc_test_trig.c.o"
    object_path = directory / "cfg_allocation.c.o"
    binary = directory / "cfg_allocation"

    def command(target_name):
        result = checked([ninja, "-C", str(BUILD), "-t", "commands", target_name])
        return shlex.split(result.stdout.splitlines()[-1])

    compile_command = [str(object_path) + token[len(original_object):]
                       if token.startswith(original_object) else str(SOURCE)
                       if token.endswith("/mgc_test_trig.c") else token
                       for token in command(original_object)]
    checked(compile_command, cwd=BUILD)
    link_command = [str(binary) if token == target else str(object_path)
                    if token == original_object else token for token in command(target)]
    checked(link_command, cwd=BUILD)
    return binary


def compile_case(binary, mode, count, *, success=True):
    result = subprocess.run([str(binary), mode, str(count)], capture_output=True, text=True, timeout=60)
    if not success:
        assert result.returncode == 1 and not result.stdout, result.stdout + result.stderr
        return result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    statistics = json.loads(next(line[6:] for line in result.stderr.splitlines() if line.startswith("stats=")))
    layout = ShaderLayout(json.loads(next(line[7:] for line in result.stderr.splitlines() if line.startswith("layout="))))
    assert layout.target == ProgramFormat.LEGACY and layout.fragment_status["version"] == 1
    assert "mgc_block_" in result.stdout and "mgc_edge_" in result.stdout
    assert 1 < statistics["blocks"] and statistics["registers"] <= 32 and statistics["predicates"] <= 8
    assert all(int(register) < 32 for register in re.findall(r"\br(\d+)\b", result.stdout))
    return result.stdout, layout, statistics


def invoke(assembly, layout, coordinates):
    executor = ShaderExecutor(assembly, layout)
    try:
        result = executor.invoke_fragment({entry["word_offset"]: pack_float(coordinates[entry["component"]])
                                           for entry in layout.entries("inputs")})
        assert result.status is FragmentStatus.COMMITTED
        assert executor.invocations == 1 and executor.steps > 1
        return {entry["component"]: result.outputs[entry["word_offset"]] for entry in layout.entries("outputs")}
    finally:
        executor.close()


@pytest.mark.parametrize("mode", ("sequential", "exclusive"))
@pytest.mark.parametrize("count", (32, 64, 256))
@pytest.mark.parametrize("seed", (-0.5, 0.5))
def test_more_than_24_nonlive_scalar_defs_and_exclusive_branches_reuse_registers(cfg_backend, mode, count, seed):
    assembly, layout, statistics = compile_case(cfg_backend, mode, count)
    assert statistics["definitions"] > 24
    expected = seed if mode == "sequential" and seed >= 0 else seed + count * (0.25 if mode == "sequential" or seed >= 0 else -0.25)
    assert invoke(assembly, layout, (seed, 0.25)) == {0: pack_float(expected)}
    for line in assembly.splitlines():
        if line.startswith("fadd "):
            operands = line[5:].split(", ")
            assert operands[-1] not in operands[:-1]


@pytest.mark.parametrize("count", (32, 64))
@pytest.mark.parametrize("seed", (-0.5, 0.5))
def test_nonlive_comparisons_bcsel_and_inot_recycle_predicates(cfg_backend, count, seed):
    assembly, layout, statistics = compile_case(cfg_backend, "predicate-chain", count)
    assert statistics["definitions"] > 24 and "fcomp_" in assembly and "copy.cr" in assembly
    value = seed
    selected = 0.25
    if seed < 0:
        for _ in range(count):
            selected = 0.25 if value < 0 else -0.25
            value += selected
    assert invoke(assembly, layout, (seed, 0.25)) == {0: pack_float(value), 1: pack_float(selected)}


@pytest.mark.parametrize("mode,width", (("swap-loop", 2), ("rotate-loop", 3)))
@pytest.mark.parametrize("iterations", (0, 1, 2, 3, 8))
def test_parallel_phi_cycles_preserve_distinct_loop_carried_values(cfg_backend, mode, width, iterations):
    assembly, layout, _ = compile_case(cfg_backend, mode, 1)
    initial = (0.25, 0.5, 0.75)
    expected = {component: pack_float(initial[(component + iterations) % width]) for component in range(width)}
    assert invoke(assembly, layout, (*initial, iterations)) == expected


@pytest.mark.parametrize("iterations", (0, 1, 2, 3, 8, 31))
@pytest.mark.parametrize("first,second", ((0.25, 0.5), (0.75, 0.25)))
def test_predicate_phi_loop_carried_inot_and_selection_preserve_sources(cfg_backend, iterations, first, second):
    assembly, layout, _ = compile_case(cfg_backend, "predicate-loop", 1)
    flag = first < second
    total = 0
    for _ in range(iterations):
        total += first if flag else second
        flag = not flag
    assert invoke(assembly, layout, (first, second, 0, iterations)) == {
        0: pack_float(total), 1: pack_float(first if flag else second)}


@pytest.mark.parametrize("count", (25, 32, 40))
def test_genuine_more_than_24_live_scalars_fail_without_partial_assembly(cfg_backend, count):
    diagnostic = compile_case(cfg_backend, "scalar-pressure", count, success=False)
    assert "CFG scalar register pressure exceeds 24 registers" in diagnostic and "no spills" in diagnostic


@pytest.mark.parametrize("count", (8, 12))
def test_genuine_more_than_7_live_predicates_fail_without_partial_assembly(cfg_backend, count):
    diagnostic = compile_case(cfg_backend, "predicate-pressure", count, success=False)
    assert "CFG predicate register pressure exceeds 7 registers" in diagnostic and "no spills" in diagnostic


def test_cfg_analysis_is_bounded_to_4096_values(cfg_backend):
    diagnostic = compile_case(cfg_backend, "sequential", 4097, success=False)
    assert "CFG exceeds the 4096-value analysis limit" in diagnostic
