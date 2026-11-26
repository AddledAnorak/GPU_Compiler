import os
import subprocess
import sys

import pytest

from tools.run_shader import ABI_BASES, ABI_REGION_WORDS, ROOT, compile_glsl, execute_assembly, execute_shader_pair


@pytest.fixture(scope="module")
def fresh_glsl_artifacts(tmp_path_factory):
    if not (ROOT / "tools/mgc_harness").is_file() or not (ROOT / "mesa/build/build.ninja").is_file():
        pytest.skip("fresh Mesa/harness required; run bash tools/build_project.sh")
    output = tmp_path_factory.mktemp("fresh-glsl")
    return {name: compile_glsl(ROOT / "shaders" / name, ROOT / "mesa/build", output / name)
            for name in ("01_flat_color", "03_uniform", "09_runtime_loop", "12_runtime_arithmetic")}


def test_compiled_assembly_runs_through_driver_owned_buffers():
    result = execute_assembly("""
        movi_f16 1.0, r0
        movi_i32 0x100000, r1
        store r0, r1
        stop_core
    """)
    assert result["fence"] == "completed"
    assert result["outputs"]["fragment"]["fp16"][0] == 1.0
    assert not result["fpga_execution"]


def test_compiler_abi_word_address_becomes_managed_byte_address():
    result = execute_assembly("""
        movi_i32 0x10000, r0
        load r0, r1
        wait_load
        movi_i32 0x100000, r2
        store r1, r2
        stop_core
    """, {ABI_BASES["input"]: 0.5})
    assert result["outputs"]["fragment"]["fp16"][0] == 0.5


def test_unknown_host_abi_input_does_not_escape_allocations():
    with pytest.raises(ValueError, match="outside documented"):
        execute_assembly("stop_core", {0xDEADBEEF: 1})


@pytest.mark.parametrize("payload", [0, 5, -7, 0xFFFFFFFF, 0x80000000])
def test_integer_uniform_payload_is_not_converted_to_fp16(payload):
    result = execute_assembly("""
        movi_i32 0x20000, r0
        load r0, r1
        wait_load
        movi_i32 0x100000, r2
        store r1, r2
        stop_core
    """, raw_values={ABI_BASES["uniform"]: payload})
    assert result["outputs"]["fragment"]["raw_words"][0] == payload & 0xFFFFFFFF
    assert result["input_payload_words"] == {"fp16": 0, "raw_i32_u32": 1}


@pytest.mark.parametrize("payload", [True, 1.5, -(1 << 31) - 1, 1 << 32, "5"])
def test_raw_payload_rejects_non_integer_or_overflow(payload):
    with pytest.raises(ValueError, match="raw ABI values"):
        execute_assembly("stop_core", raw_values={ABI_BASES["uniform"]: payload})


def test_ambiguous_payload_type_is_rejected():
    with pytest.raises(ValueError, match="must not overlap"):
        execute_assembly("stop_core", {ABI_BASES["uniform"]: 1.0}, {ABI_BASES["uniform"]: 1})


def test_fresh_glsl_flat_color_invocation(fresh_glsl_artifacts):
    result = execute_assembly(fresh_glsl_artifacts["01_flat_color"]["fs"])
    assert result["outputs"]["fragment"]["fp16"][:4] == [1.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("count", [-3, 0, 1, 3, 5])
def test_fresh_glsl_runtime_loop_nonzero_integer_uniform(fresh_glsl_artifacts, count):
    color = [0.25, 0.5, 1.0]
    values = {ABI_BASES["input"] + index: value for index, value in enumerate(color)}
    result = execute_assembly(fresh_glsl_artifacts["09_runtime_loop"]["fs"], values,
                              {ABI_BASES["uniform"]: count})
    multiplier = max(count, 0) * (max(count, 0) + 1) // 2
    assert result["outputs"]["fragment"]["fp16"][:4] == [value * multiplier for value in color] + [1.0]


def test_fresh_glsl_uniform_vector_components_are_all_dynamic(fresh_glsl_artifacts):
    color = [0.25, 0.5, 0.75]
    tint = [0.5, 1.0, 2.0, 1.0]
    values = {ABI_BASES["input"] + 4 + index: value for index, value in enumerate(color)}
    values.update({ABI_BASES["uniform"] + 16 + index: value for index, value in enumerate(tint)})
    values.update({ABI_BASES["uniform"] + index * 5: 1.0 for index in range(4)})
    values[ABI_BASES["input"] + 3] = 1.0
    result = execute_shader_pair(fresh_glsl_artifacts["03_uniform"], varying_map={4: 0, 5: 1, 6: 2},
                                  vertex_values=values, fragment_values={ABI_BASES["uniform"] + 3: tint[3]})
    assert result["outputs"]["fragment"]["fp16"][:4] == [0.125, 0.5, 1.5, 1.0]


def test_fresh_glsl_arithmetic_preserves_both_scalar_uniforms(fresh_glsl_artifacts):
    values = {ABI_BASES["input"] + 4: 0.75, ABI_BASES["input"] + 5: 0.5,
              ABI_BASES["input"] + 6: 0.25, ABI_BASES["uniform"]: 0.25,
              ABI_BASES["uniform"] + 4: 1.5}
    result = execute_shader_pair(fresh_glsl_artifacts["12_runtime_arithmetic"], varying_map={4: 0, 5: 1, 6: 2},
                                  vertex_values=values)
    assert result["outputs"]["fragment"]["fp16"][:4] == [0.5, -0.5, 0.25, 1.0]


def test_all_pipeline_tool_modules_import_without_tools_path():
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "-c",
                                "import tools.validate_frontend, tools.reference_demo, tools.benchmark_reference"],
                               cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mapping", [{-1: 0}, {16: 0}, {True: 0}, {4: -1}, {4: ABI_REGION_WORDS},
                                      {4: True}, {4: 0, 5: 0}])
def test_shader_pair_rejects_invalid_or_ambiguous_varying_maps(mapping):
    with pytest.raises(ValueError):
        execute_shader_pair({"vs": "stop_core", "fs": "stop_core"}, varying_map=mapping)


def test_shader_pair_does_not_override_explicit_fragment_payload():
    with pytest.raises(ValueError, match="must not overlap"):
        execute_shader_pair({"vs": "stop_core", "fs": "stop_core"}, varying_map={4: 0},
                             fragment_values={ABI_BASES["input"]: 0.5})
