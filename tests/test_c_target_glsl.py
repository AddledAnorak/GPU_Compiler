import json
import math
import os
from pathlib import Path
import subprocess

import pytest

from tools.mgcc import assemble_c, disassemble_c
from tools.mgcc.runtime import CShaderExecutor, execute_c_pair
from tools.mgcdriver import ExecutionTimeoutError, ProgramFormat
from tools.run_c_shader import compile_c_glsl
from tools.run_shader import compile_glsl, execute_assembly
from tools.run_vector_shader import phong_payload, reference_phong

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))


@pytest.fixture(scope="module")
def c_glsl(tmp_path_factory):
    if not (BUILD / "build.ninja").is_file():
        pytest.skip("configured Mesa build is required for real GLSL frontend tests")
    folder = tmp_path_factory.mktemp("c-glsl")
    programs = {name: compile_c_glsl(ROOT / "shaders" / name, BUILD, folder / name)
                for name in ("01_flat_color", "03_uniform", "06_builtin_math", "07_lighting_phong", "12_runtime_arithmetic")}
    fixtures = {
        "branch": "uniform float value;void main(){if(value<0.0)gl_FragColor=vec4(0.25);else gl_FragColor=vec4(0.75);}",
        "loop": "uniform int count;varying vec3 color;void main(){vec3 total=vec3(0.0);for(int index=0;index<count;index++)total+=color*0.25;gl_FragColor=vec4(total,1.0);}",
        "sqrt": "uniform float value;void main(){gl_FragColor=vec4(sqrt(value),inversesqrt(value),value+0.5,1.0);}",
        "two-light": "varying vec3 normal;uniform vec3 light0;uniform vec3 light1;uniform vec3 tint;void main(){vec3 n=normalize(normal);float first=max(dot(n,normalize(light0)),0.0);float second=max(dot(n,normalize(light1)),0.0);gl_FragColor=vec4(tint*(0.1+first+second*0.5),1.0);}",
    }
    for name, fragment in fixtures.items():
        source = folder / (name + "-source")
        source.mkdir()
        varying = "varying vec3 color;" if name == "loop" else "varying vec3 normal;" if name == "two-light" else ""
        assignment = "color=attribute_value;" if name == "loop" else "normal=attribute_value;" if name == "two-light" else ""
        (source / "vert.glsl").write_text("#version 100\nattribute vec4 position;attribute vec3 attribute_value;" + varying + "void main(){gl_Position=position;" + assignment + "}")
        (source / "frag.glsl").write_text("#version 100\nprecision mediump float;" + fragment)
        programs[name] = compile_c_glsl(source, BUILD, folder / name)
    programs["folder"] = folder
    return programs


def vertex_inputs(compiled, attribute=(0.75, 0.5, 0.25)):
    position = (0.125, 0.25, 0.5, 1.0)
    return {entry["word_offset"]: (position if entry["location"] == 15 else attribute)[entry["component"]]
            for entry in compiled["layouts"]["vs"]["inputs"]}


def uniforms(compiled, stage, values):
    return {entry["word_offset"]: values[entry["word_offset"]] for entry in compiled["layouts"][stage]["uniforms"]}


def rgba(result):
    return [result["fragment"]["outputs"][offset] for offset in range(4)]


@pytest.mark.parametrize("name", ["01_flat_color", "03_uniform", "06_builtin_math", "07_lighting_phong", "12_runtime_arithmetic", "branch", "loop", "sqrt", "two-light"])
def test_fresh_c_target_tags_words_and_reflection(c_glsl, name):
    compiled = c_glsl[name]
    assert compiled["manifest"]["shader_abi_version"] == 5
    assert compiled["manifest"]["program_format"] == "c-interpreter-v1"
    for stage, source in compiled["assembly"].items():
        assert "wait_" not in source
        assert "goto " not in source and "goto.cr" not in source
        assert not any(name in source for name in ("vfadd16", "vfmul16", "vget_lane0_i32"))
        words = assemble_c(source)
        assert assemble_c("\n".join(disassemble_c(words))) == words
        assert "stop_core" in source
        assert compiled["layouts"][stage]["address_units"] == "bytes"


def test_actual_flat_color(c_glsl):
    compiled = c_glsl["01_flat_color"]
    result = execute_c_pair(compiled, vertex_inputs(compiled), {}, {})
    assert rgba(result) == [1, 0, 0, 1]
    assert result["fragment"]["completion"] == "stop_core-and-irq"
    assert result["fpga_execution"] is False


@pytest.mark.parametrize("value,expected", [(-1.0, 0.25), (0.0, 0.75), (1.0, 0.75)])
def test_real_glsl_runtime_predicated_branch(c_glsl, value, expected):
    compiled = c_glsl["branch"]
    result = execute_c_pair(compiled, vertex_inputs(compiled), uniforms(compiled, "vs", {0: value}), uniforms(compiled, "fs", {0: value}))
    assert rgba(result) == [expected] * 4


@pytest.mark.parametrize("count", [-1, 0, 1, 4, 7])
def test_real_glsl_integer_control_flow_and_parallel_phi(c_glsl, count):
    compiled = c_glsl["loop"]
    result = execute_c_pair(compiled, vertex_inputs(compiled), uniforms(compiled, "vs", {0: count}), uniforms(compiled, "fs", {0: count}))
    expected = [component * 0.25 * max(count, 0) for component in (0.75, 0.5, 0.25)] + [1.0]
    assert rgba(result) == expected
    assert "jump" in compiled["assembly"]["fs"]


@pytest.mark.parametrize("value,expected", [(0.25, [0.5, 2.0, 0.75, 1]), (1.0, [1, 1, 1.5, 1]), (4.0, [2, 0.5, 4.5, 1])])
def test_native_sqrt_and_rsqrt_from_nonconstant_glsl(c_glsl, value, expected):
    compiled = c_glsl["sqrt"]
    assert "fsqrt16" in compiled["assembly"]["fs"] and "frsqrt16" in compiled["assembly"]["fs"]
    result = execute_c_pair(compiled, vertex_inputs(compiled), uniforms(compiled, "vs", {0: value}), uniforms(compiled, "fs", {0: value}))
    assert rgba(result) == expected


@pytest.mark.parametrize("position,normal,light", [((0.25, -0.125, 0), (0, 0, 1), (1, 0.5, 3)),
    ((-0.25, 0.125, 0), (0, 0, -1), (0, 0, 3)),
    ((0, 0, 0), (0, 1, 0), (1, 3, 1)), ((0, 0, 0), (1, 1, 1), (-1, 2, 3))])
def test_actual_full_phong_nonconstant_vs_to_fs(c_glsl, position, normal, light):
    compiled = c_glsl["07_lighting_phong"]
    payload = phong_payload(compiled, position=position, normal=normal, light=light)
    result = execute_c_pair(compiled, **payload)
    expected = reference_phong(position, normal, light, (0, 0, 3), (0.7, 0.4, 0.2))
    assert max(abs(lhs - rhs) for lhs, rhs in zip(rgba(result), expected)) < 0.05
    assert "frsqrt16" in compiled["assembly"]["fs"]


@pytest.mark.parametrize("normal,light0,light1", [((0, 0, 1), (0, 0, 1), (1, 0, 1)),
    ((0, 1, 0), (0, -1, 0), (0, 1, 1)), ((1, 1, 1), (-1, 2, 3), (2, -3, 1))])
def test_two_runtime_lights_independent_numeric_formula(c_glsl, normal, light0, light1):
    compiled = c_glsl["two-light"]
    tint = (0.25, 0.5, 0.75)
    values = {base + component: value for base, vector in ((0, light0), (4, light1), (8, tint))
              for component, value in enumerate(vector)}
    result = execute_c_pair(compiled, vertex_inputs(compiled, normal), uniforms(compiled, "vs", values), uniforms(compiled, "fs", values))
    def normalized(vector):
        length = math.sqrt(sum(component * component for component in vector))
        return [component / length for component in vector]
    unit = normalized(normal)
    first = max(sum(lhs * rhs for lhs, rhs in zip(unit, normalized(light0))), 0.0)
    second = max(sum(lhs * rhs for lhs, rhs in zip(unit, normalized(light1))), 0.0)
    expected = [component * (0.1 + first + 0.5 * second) for component in tint] + [1.0]
    assert max(abs(lhs - rhs) for lhs, rhs in zip(rgba(result), expected)) < 0.02


def test_positive_native_bridge_and_guard_sanitizers(c_glsl, tmp_path):
    executable = tmp_path / "native-safety"
    command = [os.environ.get("CC", "cc"), "-std=c11", "-O0", "-g", "-Wall", "-Wextra", "-Werror",
               "-fsanitize=address,undefined,float-cast-overflow", "-fno-omit-frame-pointer",
               "-I", str(ROOT), "-I", str(ROOT / "interpreter/include"),
               str(ROOT / "tests/fixtures/c-target/native_safety.c"),
               str(ROOT / "interpreter/src/mgc_interpreter.c"), str(ROOT / "interpreter/src/mgc_fp16_math.c"),
               "-lm", "-o", str(executable)]
    compiled = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
    program_path = c_glsl["folder"] / "01_flat_color/fs.bin"
    completed = subprocess.run([str(executable), str(program_path)], env=environment,
                               text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "five guard checks and fresh flat GLSL: pass" in completed.stdout


@pytest.mark.parametrize("bias,divisor", [(0.25, 0.5), (-0.25, 1.5), (-0.75, 0.5)])
def test_optimized_stage_uniform_linkage_is_preserved(c_glsl, bias, divisor):
    compiled = c_glsl["12_runtime_arithmetic"]
    values = {0: bias, 4: divisor}
    result = execute_c_pair(compiled, vertex_inputs(compiled), uniforms(compiled, "vs", values), uniforms(compiled, "fs", values))
    expected = (0.75 + bias) / (divisor + 0.5)
    assert rgba(result) == [expected, -expected, 0.25, 1]


def test_c_output_never_routes_to_old_scalar_executor(c_glsl):
    with pytest.raises(ValueError, match="current C compiler output"):
        execute_assembly(c_glsl["01_flat_color"]["assembly"]["fs"])


def test_budget_failure_is_not_successful_shader_output(c_glsl):
    compiled = c_glsl["01_flat_color"]
    with CShaderExecutor(compiled["assembly"]["fs"], compiled["layouts"]["fs"], max_steps=1) as executor:
        with pytest.raises(ExecutionTimeoutError, match="instruction budget"):
            executor.execute({}, {})


def test_required_payload_offsets_are_not_guessed(c_glsl):
    compiled = c_glsl["sqrt"]
    with CShaderExecutor(compiled["assembly"]["fs"], compiled["layouts"]["fs"]) as executor:
        with pytest.raises(ValueError, match="exactly the reflected"):
            executor.execute({}, {})


@pytest.mark.parametrize("fragment,diagnostic", [
    ("uniform float value;void main(){if(value<0.0)discard;gl_FragColor=vec4(value);}", "discard/output-valid"),
    ("uniform int value;void main(){gl_FragColor=vec4(float(value));}", "signed integer/FP16"),
    ("uniform sampler2D image;void main(){gl_FragColor=texture2D(image,vec2(0.5));}", "texture"),
])
def test_unsupported_glsl_fails_at_compiler_without_assembly(tmp_path, fragment, diagnostic):
    source = tmp_path / "source"
    source.mkdir()
    (source / "vert.glsl").write_text("#version 100\nattribute vec4 position;void main(){gl_Position=position;}")
    (source / "frag.glsl").write_text("#version 100\nprecision mediump float;" + fragment)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="compilation failed"):
        compile_c_glsl(source, BUILD, output)
    assert diagnostic in (output / "compile.log").read_text()
    assert not (output / "fs.asm").exists()


def test_no_overwrite_and_clean_environment_cli(c_glsl, tmp_path):
    previous = c_glsl["folder"] / "01_flat_color"
    with pytest.raises(ValueError, match="not overwritten"):
        compile_c_glsl(ROOT / "shaders/01_flat_color", BUILD, previous)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    output = tmp_path / "cli"
    completed = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/run_c_shader.py"),
        str(ROOT / "shaders/07_lighting_phong"), "--build", str(BUILD), "--output", str(output), "--phong"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((output / "report.json").read_text())["passed"] is True
