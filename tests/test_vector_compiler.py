import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import struct
import sys

import pytest
import numpy as np

from tools.mgcdriver import Driver, Program, ProgramFormat, ProgramFormatError, SimulationBackend
from tools.mgchw import assemble_hardware, decode_word, disassemble_hardware
from tools.run_shader import ROOT, compile_glsl, execute_assembly
from tools.run_vector_shader import (VECTOR_TARGET, compile_vector_glsl, execute_vector_pair,
                                     phong_payload, reference_phong)


BUILD = ROOT / "mesa/build"


@pytest.mark.parametrize("assembly,word", [
    ("vbcast r28, v1", 0x803C1001), ("vget_lane0_i32 v0, r12", 0xD800080C),
    ("vfadd16 v5, v1, v5", 0x80051C25), ("vfsub16 v5, v1, v7", 0x80052027),
    ("vfmul16 v7, v7, v7", 0x800724E7), ("vfmax16 v5, v1, v5", 0x80052825),
    ("icmp_ge r9, r17, cr0", 0xD0290E20), ("[cr4] jump 71", 0xAD001C47),
])
def test_vector_target_uses_original_utah_golden_words(assembly, word):
    assert assemble_hardware(assembly) == [word]
    assert assemble_hardware("\n".join(disassemble_hardware([word]))) == [word]


@pytest.fixture(scope="module")
def vector_binary(tmp_path_factory):
    ninja = shutil.which("ninja") or str(ROOT / ".venv/bin/ninja")
    environment = dict(os.environ)
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    target = "src/gallium/drivers/mgc/mgc_test_trig"
    subprocess.run([ninja, "-C", str(BUILD), target], check=True, env=environment, capture_output=True, text=True)
    directory = tmp_path_factory.mktemp("vector-compiler")
    binary, replacement = directory / "vector-compiler", directory / "vector.o"
    original = target + ".p/mgc_test_trig.c.o"
    compile_text = subprocess.check_output([ninja, "-C", str(BUILD), "-t", "commands", original],
                                           text=True, env=environment).splitlines()[-1]
    command = [str(replacement) + token[len(original):] if token.startswith(original) else
               str(ROOT / "tests/fixtures/vector_compiler.c") if token.endswith("/mgc_test_trig.c") else token
               for token in shlex.split(compile_text)]
    command.append("-I" + str(ROOT / "mesa/src/gallium/drivers/mgc"))
    subprocess.run(command, cwd=BUILD, check=True, env=environment, capture_output=True, text=True)
    link_text = subprocess.check_output([ninja, "-C", str(BUILD), "-t", "commands", target],
                                        text=True, env=environment).splitlines()[-1]
    command = [str(binary) if token == target else str(replacement) if token == original else token
               for token in shlex.split(link_text)]
    subprocess.run(command, cwd=BUILD, check=True, env=environment, capture_output=True, text=True)
    return binary


def compiled_nir(binary, mode, *, success=True):
    completed = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=30)
    assert completed.returncode == (0 if success else 1), completed.stderr
    if not success:
        assert not completed.stdout
        assert "error exposed partial assembly" not in completed.stderr
        return completed.stderr
    return completed.stdout, json.loads(completed.stderr)


@pytest.mark.parametrize("mode,left,right,expected", [
    ("add", 0.25, 0.5, 0.75), ("subtract", -0.5, 0.25, -0.75),
    ("multiply", -0.5, 0.25, -0.125), ("minimum", -0.5, 0.25, -0.5),
    ("maximum", -0.5, 0.25, 0.25), ("negate", 0.25, 0, -0.25),
    ("absolute", -0.25, 0, 0.25), ("copy", -0.5, 0, -0.5),
])
def test_separate_vector_nir_lowering(vector_binary, mode, left, right, expected):
    from tools.mgcvector import VectorShaderExecutor

    assembly, layout = compiled_nir(vector_binary, mode)
    words = assemble_hardware(assembly)
    assert assemble_hardware("\n".join(disassemble_hardware(words))) == words
    assert layout["compiler_target"] == VECTOR_TARGET
    assert layout["program_format"] == ProgramFormat.HARDWARE.value
    inputs = {entry["word_offset"]: (left, right)[entry["word_offset"]] for entry in layout["inputs"]}
    with VectorShaderExecutor(assembly, layout, allow_approximate_fp16=True) as executor:
        result = executor.execute(inputs, {})
    assert result["outputs"][0] == expected
    assert result["status"] == "committed"
    assert not result["fpga_execution"]
    assert all(decode_word(word).mnemonic not in ("fadd16",) for word in words)


@pytest.mark.parametrize("value", [2 ** -24, 2 ** -20, 2 ** -14, 0.0625, 0.25, 0.5, 1, 2, 4, 16, 256, 65504])
@pytest.mark.parametrize("mode", ["rsqrt", "sqrt"])
def test_vector_normalization_across_fp16_range(vector_binary, value, mode):
    from tools.mgcvector import VectorShaderExecutor

    assembly, layout = compiled_nir(vector_binary, mode)
    with VectorShaderExecutor(assembly, layout, allow_approximate_fp16=True) as executor:
        result = executor.execute({0: float(value)}, {})
    expected = 1 / math.sqrt(value) if mode == "rsqrt" else math.sqrt(value)
    assert abs(result["outputs"][0] - expected) / expected < 0.003


def test_vector_rsqrt_algorithm_all_positive_finite_half_patterns():
    raw = np.arange(1, 0x7C00, dtype=np.uint16)
    values = raw.view(np.float16)
    estimate = np.full(values.shape, np.float16(math.ldexp(0.632455532, 12)), dtype=np.float16)
    for exponent in range(-12, 8):
        estimate[values >= np.float16(math.ldexp(1, 2 * exponent))] = np.float16(math.ldexp(0.632455532, -exponent))
    with np.errstate(all="raise"):
        for _ in range(6):
            product = np.float16(np.float16(np.float16(values * estimate) * estimate) * np.float16(0.5))
            estimate = np.float16(estimate * np.float16(np.float16(1.5) - product))
    expected = 1 / np.sqrt(values.astype(np.float64))
    relative = np.abs(estimate.astype(np.float64) - expected) / expected
    assert np.isfinite(estimate).all()
    assert float(relative.max()) < 0.001


def test_actual_compiled_vector_rsqrt_seed_boundaries(vector_binary):
    from tools.mgcvector import VectorShaderExecutor

    assembly, layout = compiled_nir(vector_binary, "rsqrt")
    patterns = {1, 2, 0x03FF, 0x0400, 0x7BFE, 0x7BFF}
    for exponent in range(-12, 8):
        threshold = int.from_bytes(struct.pack("<e", math.ldexp(1, 2 * exponent)), "little")
        patterns.update(pattern for pattern in (threshold - 1, threshold, threshold + 1) if 1 <= pattern < 0x7C00)
    with VectorShaderExecutor(assembly, layout, allow_approximate_fp16=True) as executor:
        for pattern in sorted(patterns):
            value = struct.unpack("<e", pattern.to_bytes(2, "little"))[0]
            actual = executor.execute({0: value}, {})["outputs"][0]
            expected = 1 / math.sqrt(value)
            assert abs(actual - expected) / expected < 0.001


@pytest.mark.parametrize("mode", ["divide", "integer", "fractional-power", "constant-inf", "output-reserved"])
def test_unsupported_vector_forms_fail_without_partial_assembly(vector_binary, mode):
    assert compiled_nir(vector_binary, mode, success=False)


@pytest.mark.parametrize("value", [0, -1, -0.0])
def test_invalid_rsqrt_never_returns_earlier_shader_output(vector_binary, value):
    from tools.mgcvector import VectorExecutionError, VectorShaderExecutor

    assembly, layout = compiled_nir(vector_binary, "late-invalid")
    with VectorShaderExecutor(assembly, layout, allow_approximate_fp16=True) as executor:
        with pytest.raises(VectorExecutionError):
            executor.execute({0: float(value)}, {})


def test_long_vector_chain_uses_bounded_scratch_not_scalar_allocation(vector_binary):
    from tools.mgcvector import VectorShaderExecutor

    assembly, layout = compiled_nir(vector_binary, "chain")
    assert sum(decode_word(word).mnemonic == "vfadd16" for word in assemble_hardware(assembly)) == 240
    assert 1 <= layout["vector_contract"]["scratch_words"] <= 16384
    with VectorShaderExecutor(assembly, layout, allow_approximate_fp16=True) as executor:
        actual = executor.execute({0: -0.5, 1: 0.03125}, {})["outputs"][0]
    expected = np.float16(-0.5)
    for _ in range(240):
        expected = np.float16(np.float16(expected * np.float16(0.99951171875)) + np.float16(0.03125))
    assert actual == float(expected)


@pytest.fixture(scope="module")
def vector_phong(tmp_path_factory, vector_binary):
    output = tmp_path_factory.mktemp("fresh-vector-phong")
    return compile_vector_glsl(ROOT / "shaders/07_lighting_phong", output)


@pytest.mark.parametrize("position,normal,light,view,color", [
    ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0.2, 0.4, 0.6)),
    ((0, 0, 0), (0, 0, 1), (0, 0, -2), (0, 0, 3), (0.2, 0.4, 0.6)),
    ((0.25, -0.125, 0), (0, 0, 1), (1, 0.5, 3), (0, 0, 3), (0.7, 0.4, 0.2)),
    ((-0.5, 0.25, 0.125), (0.2, -0.3, 0.5), (2, -1, 3), (1, 2, 4), (0.3, 0.6, 0.8)),
    ((0.5, 0.25, -0.25), (1, 0, 1), (-2, 1, 2), (1, -1, 3), (0.9, 0.2, 0.1)),
    ((0, 0, 0), (0, 0, 2), (0, 0, 1), (0, 0, 1), (0, 0, 0)),
])
def test_original_phong_glsl_to_vector_words_vs_fs_reference(vector_phong, position, normal, light, view, color):
    payload = phong_payload(vector_phong, position=position, normal=normal, light=light, view=view, color=color)
    result = execute_vector_pair(vector_phong, **payload, allow_approximate_fp16=True)
    expected = reference_phong(position, normal, light, view, color)
    actual = [result["fragment"]["outputs"][component] for component in range(4)]
    assert max(abs(left - right) for left, right in zip(actual, expected)) < 0.05
    assert result["vertex"]["host_steps"] > 0 and result["fragment"]["host_steps"] > 0
    assert not result["fpga_execution"] and not result["rasterization"]
    counts = vector_phong["manifest"]["stages"]["fs"]["instructions"]
    assert counts["vfmul16"] > 10 and counts["vfmax16"] >= 2 and counts["vfsub16"] > 0
    assert "fadd16" not in counts


def test_scalar_compiler_and_executors_remain_separate(vector_phong, tmp_path):
    legacy = compile_glsl(ROOT / "shaders/07_lighting_phong", BUILD, tmp_path / "legacy")
    assert legacy["fs"].startswith("; mgc legacy-simulator backend, ABI=1,")
    assert "fmul16" in legacy["fs"] and "vfmul16" not in legacy["fs"]
    with pytest.raises(ValueError, match="vector.*never.*scalar"):
        execute_assembly(vector_phong["assembly"]["fs"])
    with pytest.raises(ValueError, match="vector.*never.*scalar"):
        execute_assembly(vector_phong["assembly"]["fs"], target=ProgramFormat.HARDWARE)
    program = Program.from_words(assemble_hardware(vector_phong["assembly"]["fs"]), format=ProgramFormat.HARDWARE)
    with Driver(SimulationBackend(memory_size=5 << 20)) as driver:
        with pytest.raises(ProgramFormatError):
            driver.submit(program)


@pytest.mark.parametrize("name", ["08_texture", "09_runtime_loop", "10_branch_discard"])
def test_vector_glsl_rejects_unimplemented_shader_features(vector_phong, tmp_path, name):
    with pytest.raises(ValueError, match="vector GLSL compilation failed"):
        compile_vector_glsl(ROOT / "shaders" / name, tmp_path)
    assert not (tmp_path / "fs.mem").exists() and not (tmp_path / "vs.asm").exists()


def test_vector_compilation_refuses_to_overwrite_previous_artifacts(vector_phong, tmp_path):
    (tmp_path / "fs.asm").write_text("preserve this artifact")
    with pytest.raises(ValueError, match="fresh directory"):
        compile_vector_glsl(ROOT / "shaders/07_lighting_phong", tmp_path)
    assert (tmp_path / "fs.asm").read_text() == "preserve this artifact"


def test_vector_cli_clean_import_emits_machine_code_and_reference(tmp_path, vector_phong):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "-m", "tools.run_vector_shader", "--output", str(tmp_path),
                                "--allow-approximate-fp16"], cwd=ROOT, env=environment,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["manifest"]["machine_code_emitted"]
    assert not result["manifest"]["board_upload_ready"]
    assert (tmp_path / "fs.mem").is_file() and (tmp_path / "vs.bin").is_file()
    assert result["independent_reference"]["max_absolute_error"] < 0.05
