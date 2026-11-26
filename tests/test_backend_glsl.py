"""Real GLSL -> Mesa -> compiler -> managed legacy driver/CPU checks."""

import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from tools.mgcasm import assemble, disassemble
from tools.run_shader import compile_glsl, execute_assembly

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = pathlib.Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
INPUT = 0x10000
UNIFORM = 0x20000
FRAMEBUFFER = 0x100000


@pytest.fixture(scope="module")
def glsl_programs(tmp_path_factory):
    if not (BUILD / "build.ninja").is_file() or not (ROOT / "tools/mgc_harness").is_file():
        pytest.skip("configured Mesa and the GLSL harness are required")
    libraries = list((BUILD / "src/gallium/targets/dri").glob("libgallium-*.so"))
    assert len(libraries) == 1
    target = str(libraries[0].relative_to(BUILD))
    environment = dict(os.environ)
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    ninja = shutil.which("ninja") or str(ROOT / ".venv/bin/ninja")
    result = subprocess.run([ninja, "-C", str(BUILD), target], text=True,
                            capture_output=True, env=environment, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    temporary = tmp_path_factory.mktemp("glsl_backend")
    programs = {}
    for name in ("03_uniform", "07_lighting_phong", "09_runtime_loop",
                 "10_branch_discard", "12_runtime_arithmetic"):
        programs[name] = compile_glsl(ROOT / "shaders" / name, BUILD, temporary / name)
    vertex = temporary / "vertex_source"
    vertex.mkdir()
    (vertex / "vert.glsl").write_text(
        "#version 100\nattribute vec4 a_position;\n"
        "void main() { gl_Position = a_position; }\n"
    )
    (vertex / "frag.glsl").write_text(
        "#version 100\nprecision mediump float;\n"
        "void main() { gl_FragColor = vec4(1.0); }\n"
    )
    programs["vertex"] = compile_glsl(vertex, BUILD, temporary / "vertex_dump")
    for stages in programs.values():
        for assembly in stages.values():
            words = assemble(assembly)
            assert assemble("\n".join(disassemble(words))) == words
    return programs


@pytest.mark.parametrize("count", [-1, 0, 1, 4, 7])
def test_real_glsl_integer_bound_vector_loop(glsl_programs, count):
    assembly = glsl_programs["09_runtime_loop"]["fs"]
    result = execute_assembly(assembly, {INPUT: 0.25, INPUT + 1: 0.5, INPUT + 2: 0.75},
                              raw_values={UNIFORM: count})
    repetitions = max(count, 0)
    multiplier = repetitions * (repetitions + 1) / 2
    expected = [0.25 * multiplier, 0.5 * multiplier, 0.75 * multiplier, 1]
    assert result["outputs"]["fragment"]["fp16"][:4] == expected
    assert result["fence"] == "completed"
    assert not result["fpga_execution"] and not result["rasterization"]
    assert "goto.cr" in assembly


@pytest.mark.parametrize("red,killed", [(0.25, False), (0.5, False), (0.75, True)])
def test_real_glsl_early_discard(glsl_programs, red, killed):
    values = {INPUT: red, INPUT + 1: 0.5, INPUT + 2: 0.75, UNIFORM: 0.5}
    values.update((FRAMEBUFFER + component, 0.875) for component in range(4))
    result = execute_assembly(glsl_programs["10_branch_discard"]["fs"], values)
    expected = [0.875] * 4 if killed else [red, 0.5, 0.75, 1]
    assert result["outputs"]["fragment"]["fp16"][:4] == expected


@pytest.mark.parametrize("direction", [1, -1])
def test_real_glsl_phong_with_actual_uniform_layout(glsl_programs, direction):
    values = {INPUT: 0, INPUT + 1: 0, INPUT + 2: 1,
              INPUT + 3: 0, INPUT + 4: 0, INPUT + 5: 0,
              UNIFORM: 0, UNIFORM + 1: 0, UNIFORM + 2: 2 * direction,
              UNIFORM + 4: 0, UNIFORM + 5: 0, UNIFORM + 6: 3,
              UNIFORM + 8: 0.2, UNIFORM + 9: 0.4, UNIFORM + 10: 0.6}
    result = execute_assembly(glsl_programs["07_lighting_phong"]["fs"], values)
    color = np.asarray([0.2, 0.4, 0.6], dtype=np.float16).astype(float)
    expected = color * (1.1 if direction > 0 else 0.1) + (0.5 if direction > 0 else 0)
    actual = result["outputs"]["fragment"]["fp16"][:4]
    for component in range(3):
        assert abs(actual[component] - expected[component]) < 0.025
    assert actual[3] == 1


def test_real_glsl_vertex_position_output(glsl_programs):
    position = [0.125, 0.25, 0.5, 1]
    result = execute_assembly(glsl_programs["vertex"]["vs"],
                              {INPUT + component: value for component, value in enumerate(position)})
    assert result["outputs"]["vertex"]["fp16"][:4] == position
    assert result["outputs"]["fragment"]["raw_words"] == [0] * 16


@pytest.mark.parametrize("tint", [(0.5, 1, 2, 1), (0, -1, 0.5, 0.5), (2, 0.5, -1, 2)])
def test_real_glsl_tint_survives_cross_stage_optimization(glsl_programs, tint):
    color = [0.25, 0.5, 0.75]
    position = [0.125, 0.25, 0.5, 1]
    values = {INPUT + component: value for component, value in enumerate(position)}
    values.update({INPUT + 4 + component: value for component, value in enumerate(color)})
    values.update({UNIFORM + component: 1 for component in (0, 5, 10, 15)})
    values.update({UNIFORM + 16 + component: value for component, value in enumerate(tint)})
    stages = glsl_programs["03_uniform"]
    vertex = execute_assembly(stages["vs"], values)
    varying = vertex["outputs"]["vertex"]["fp16"][4:7]
    expected = [color[component] * tint[component] for component in range(3)]
    assert vertex["outputs"]["vertex"]["fp16"][:4] == position
    assert varying == expected
    fragment_values = {INPUT + component: value for component, value in enumerate(varying)}
    fragment_values[UNIFORM + 3] = tint[3]
    fragment = execute_assembly(stages["fs"], fragment_values)
    assert fragment["outputs"]["fragment"]["fp16"][:4] == expected + [tint[3]]
    assert "movi_i32 131088," in stages["vs"]
    assert "fmul16" in stages["vs"]


@pytest.mark.parametrize("bias,divisor,expected", [(0.25, 1.5, 0.5),
                                                  (-0.25, 0.5, 0.5),
                                                  (1.25, 1.5, 1)])
def test_real_glsl_scalar_uniforms_survive_cross_stage_optimization(
        glsl_programs, bias, divisor, expected):
    values = {INPUT: 0.125, INPUT + 1: 0.25, INPUT + 2: 0.5, INPUT + 3: 1,
              INPUT + 4: 0.75, INPUT + 5: 0.5, INPUT + 6: 0.25,
              UNIFORM: bias, UNIFORM + 4: divisor}
    stages = glsl_programs["12_runtime_arithmetic"]
    vertex = execute_assembly(stages["vs"], values)
    varying = vertex["outputs"]["vertex"]["fp16"][4:7]
    assert varying == [0.75 + bias, 0.5 + divisor, 0.25]
    fragment = execute_assembly(stages["fs"],
                                {INPUT + component: value for component, value in enumerate(varying)})
    assert fragment["outputs"]["fragment"]["fp16"][:4] == [expected, -expected, 0.25, 1]
    assert "movi_i32 131072," in stages["vs"]
    assert "movi_i32 131076," in stages["vs"]
    assert "fadd16" in stages["vs"] and "fdiv16" in stages["fs"]
