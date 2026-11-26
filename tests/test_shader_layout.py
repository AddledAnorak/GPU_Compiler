"""Fresh compiler reflection, without rebuilding shared Gallium targets."""

import json
import os
import pathlib
import shlex
import shutil
import subprocess

import pytest

from tools.mgcdriver import ProgramFormat
from tools.run_shader import execute_assembly, mesa_environment

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = pathlib.Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
BASES = {"input": 0x10000, "uniform": 0x20000, "vertex": 0x30000, "fragment": 0x100000}

LAYOUT_CASES = r"""
   if (!strncmp(mode, "layout-", 7)) {
      if (!strcmp(mode, "layout-systemvalue")) {
         output(build, nir_i2f32(build, nir_load_vertex_id(build)), 0, 0);
      } else if (!strcmp(mode, "layout-demote") || !strcmp(mode, "layout-terminate")) {
         if (!strcmp(mode, "layout-demote"))
            nir_demote(build);
         else
            nir_terminate(build);
         output(build, nir_imm_float(build, 1), 0, 0);
      } else if (!strcmp(mode, "layout-terminate-if")) {
         nir_terminate_if(build, nir_fge_imm(build, input(build, 0, 0), 0.5f));
         output(build, nir_imm_float(build, 1), 0, 0);
      } else if (!strcmp(mode, "layout-dead")) {
         (void)nir_load_uniform(build, 1, 32, nir_imm_int(build, 55),
                                .dest_type = nir_type_float32);
         (void)input(build, 7, 2);
         output(build, nir_imm_float(build, 1), 0, 0);
      } else if (!strcmp(mode, "layout-uint") || !strcmp(mode, "layout-ambiguous") ||
                 !strcmp(mode, "layout-unknown")) {
         value = nir_load_ubo(build, 1, 32, nir_imm_int(build, 1), nir_imm_int(build, 12),
                               .align_mul = 4, .range = 4);
         if (!strcmp(mode, "layout-uint"))
            nir_store_output(build, value, nir_imm_int(build, 0), .write_mask = 1,
                              .src_type = nir_type_uint32,
                              .io_semantics = {.location = FRAG_RESULT_COLOR, .num_slots = 1});
         else if (!strcmp(mode, "layout-ambiguous"))
            output(build, nir_fadd(build, nir_i2f32(build, value), value), 0, 0);
         else {
            nir_store_output(build, value, nir_imm_int(build, 0), .write_mask = 1,
                              .io_semantics = {.location = FRAG_RESULT_COLOR, .num_slots = 1});
            nir_intrinsic_set_src_type(nir_instr_as_intrinsic(nir_builder_last_instr(build)),
                                        nir_type_invalid);
         }
      } else if (!strcmp(mode, "layout-interpolation")) {
         nir_def *barycentric = nir_load_barycentric_pixel(build, 32, .interp_mode = count);
         value = nir_load_interpolated_input(build, 1, 32, barycentric, nir_imm_int(build, 0),
                                             .base = 3, .component = 2,
                                             .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         output(build, value, 0, 0);
      } else {
         value = nir_load_input(build, 1, 32, nir_imm_int(build, 0),
                                 .base = 3, .component = 2, .dest_type = nir_type_float32,
                                 .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         nir_variable *varying = nir_variable_create(builder.shader, nir_var_shader_in,
                                                      glsl_vec4_type(), "layout_varying");
         varying->data.location = VARYING_SLOT_VAR0;
         varying->data.interpolation = INTERP_MODE_FLAT;
         if (!strcmp(mode, "layout-missing"))
            nir_intrinsic_set_io_semantics(nir_instr_as_intrinsic(value->parent_instr),
                                             (nir_io_semantics){0});
         if (!strcmp(mode, "layout-builtin"))
            nir_intrinsic_set_io_semantics(nir_instr_as_intrinsic(value->parent_instr),
                                             (nir_io_semantics){.location = VARYING_SLOT_POS, .num_slots = 1});
         if (!strcmp(mode, "layout-missing-interpolation"))
            varying->data.interpolation = INTERP_MODE_EXPLICIT;
         if (!strcmp(mode, "layout-conflict") || !strcmp(mode, "layout-alias")) {
            nir_def *other = nir_load_input(build, 1, 32, nir_imm_int(build, 0),
                                             .base = !strcmp(mode, "layout-alias") ? 4 : 3,
                                             .component = 2, .dest_type = nir_type_float32,
                                             .io_semantics = {
                                                .location = !strcmp(mode, "layout-alias") ?
                                                   VARYING_SLOT_VAR0 : VARYING_SLOT_VAR0 + 1,
                                                .num_slots = 1});
            value = nir_fadd(build, value, other);
         }
         output(build, value, 2, 1);
      }
   } else
"""


def run_checked(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def layout_backend(tmp_path_factory):
    if not (BUILD / "build.ninja").is_file():
        pytest.skip("configured Mesa build required")
    ninja = shutil.which("ninja") or str(ROOT / ".venv/bin/ninja")
    directory = tmp_path_factory.mktemp("private_shader_layout")

    def command(target):
        result = run_checked([ninja, "-C", str(BUILD), "-t", "commands", target])
        return shlex.split(result.stdout.splitlines()[-1])

    compiler_target = "src/gallium/drivers/mgc/libmgc.a.p/mgc_compiler.c.o"
    compiler_object = directory / "mgc_compiler.c.o"
    compile_command = [str(compiler_object) + token[len(compiler_target):]
                       if token.startswith(compiler_target) else token
                       for token in command(compiler_target)]
    run_checked(compile_command, cwd=BUILD)
    vector_target = "src/gallium/drivers/mgc/libmgc.a.p/mgc_vector_compiler.c.o"
    vector_object = directory / "mgc_vector_compiler.c.o"
    compile_vector = [str(vector_object) + token[len(vector_target):]
                      if token.startswith(vector_target) else token
                      for token in command(vector_target)]
    run_checked(compile_vector, cwd=BUILD)
    target = "src/gallium/drivers/mgc/mgc_test_trig"
    original_object = target + ".p/mgc_test_trig.c.o"
    source = (ROOT / "mesa/src/gallium/drivers/mgc/mgc_test_backend.c").read_text()
    source = source.replace(".src_type = nir_type_float32);", """.src_type = nir_type_float32,
                    .io_semantics = {.location = builder->shader->info.stage == MESA_SHADER_VERTEX ?
                       (slot ? VARYING_SLOT_VAR0 + slot - 1 : VARYING_SLOT_POS) : FRAG_RESULT_COLOR,
                       .num_slots = 1});""", 1)
    source = source.replace(".dest_type = nir_type_float32);", """.dest_type = nir_type_float32,
                         .io_semantics = {.location = builder->shader->info.stage == MESA_SHADER_VERTEX ?
                            VERT_ATTRIB_GENERIC0 + slot : VARYING_SLOT_VAR0 + slot,
                            .num_slots = 1});""", 1)
    flat_branch = '   } else if (!strcmp(mode, "flat")) {'
    assert source.count(flat_branch) == 1
    source = source.replace(flat_branch, '   } else ' + LAYOUT_CASES.lstrip() +
                            '   if (!strcmp(mode, "flat")) {', 1)
    source = source.replace("   if (shader->error) {", """
   mgc_maybe_dump_asm(shader, getenv("MGC_LAYOUT_STAGE") ? getenv("MGC_LAYOUT_STAGE") : "shader");
   if (shader->layout_json)
      fprintf(stderr, "layout=%s", shader->layout_json);
   if (shader->error) {""", 1)
    source_head, options_marker, source_tail = source.rpartition("   struct mgc_compile_options options =")
    assert options_marker
    source = source_head + """
   nir_foreach_block(block, builder.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_input && intr->intrinsic != nir_intrinsic_store_output)
            continue;
         nir_io_semantics semantics = nir_intrinsic_io_semantics(intr);
         if (!semantics.num_slots)
            continue;
         nir_variable_mode domain = intr->intrinsic == nir_intrinsic_load_input ?
                                      nir_var_shader_in : nir_var_shader_out;
         bool found = false;
         nir_foreach_variable_with_modes(var, builder.shader, domain)
            found |= var->data.location == semantics.location;
         if (!found) {
            nir_variable *var = nir_variable_create(builder.shader, domain, glsl_vec4_type(), "layout_io");
            var->data.location = semantics.location;
            var->data.interpolation = INTERP_MODE_SMOOTH;
         }
      }
   }
   struct mgc_compile_options options =""" + source_tail
    source_path = directory / "layout_backend.c"
    source_path.write_text(source)
    object_path = directory / "layout_backend.c.o"
    compile_command = [str(object_path) + token[len(original_object):]
                       if token.startswith(original_object) else
                       str(source_path) if token.endswith("/mgc_test_trig.c") else token
                       for token in command(original_object)]
    run_checked(compile_command, cwd=BUILD)
    binary = directory / "layout_backend"
    link_command = [str(binary) if token == target else
                    str(object_path) if token == original_object else token
                    for token in command(target)]
    link_command.insert(link_command.index(str(object_path)) + 1, str(compiler_object))
    link_command.insert(link_command.index(str(compiler_object)) + 1, str(vector_object))
    run_checked(link_command, cwd=BUILD)
    return {"binary": binary, "compiler": compiler_object, "directory": directory,
            "command": command}


def compile_layout(backend, directory, mode, count=4, option="unscheduled", success=True):
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, MGC_ASM_DUMP=str(directory))
    result = subprocess.run([str(backend["binary"]), mode, str(count), option],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if success else 1), result.stdout + result.stderr
    lines = [line.removeprefix("layout=") for line in result.stderr.splitlines()
             if line.startswith("layout=")]
    layout = json.loads(lines[0]) if lines else None
    if success:
        assert (directory / "shader.asm").read_text() == result.stdout
        assert json.loads((directory / "shader.layout.json").read_text()) == layout
    else:
        assert not result.stdout
        assert not list(directory.glob("*.asm"))
    return result.stdout, layout


def validate_contract(layout, hardware=False):
    assert set(layout) == {"version", "program_format", "shader_abi_version", "stage",
                           "address_units", "bases", "inputs", "outputs", "uniforms",
                           "uses_discard", "supported", "errors"}
    assert layout["version"] == 1
    assert layout["program_format"] == ("fp16-hardware-v1" if hardware else "legacy-mgcsim-v1")
    assert layout["shader_abi_version"] == (2 if hardware else 1)
    assert layout["stage"] in {"vertex", "fragment"}
    assert layout["address_units"] == ("bytes" if hardware else "words")
    assert layout["bases"] == ({domain: base * 4 for domain, base in BASES.items()} if hardware else BASES)
    assert type(layout["uses_discard"]) is bool
    assert type(layout["supported"]) is bool
    assert layout["supported"] == (not layout["errors"])
    for domain in ("inputs", "outputs", "uniforms"):
        entries = layout[domain]
        offsets = [entry["word_offset"] for entry in entries]
        assert offsets == sorted(set(offsets))
        assert len(entries) <= 65536
        for entry in entries:
            assert set(entry) == ({"word_offset", "type", "bit_size"} if domain == "uniforms" else
                                  {"word_offset", "type", "bit_size", "location", "component", "interpolation"})
            assert type(entry["word_offset"]) is int and 0 <= entry["word_offset"] < 65536
            assert entry["type"] in {"float", "int", "uint", "unknown"}
            assert type(entry["bit_size"]) is int
            if domain != "uniforms":
                assert type(entry["location"]) is int
                assert type(entry["component"]) is int
                assert entry["interpolation"] in {"smooth", "flat", "noperspective", "unknown"}


@pytest.mark.parametrize("mode", ["flat", "copy", "uniform", "ubo", "vertex", "integer-runtime-loop"])
def test_existing_backend_layouts(layout_backend, tmp_path, mode):
    _, layout = compile_layout(layout_backend, tmp_path, mode)
    validate_contract(layout)
    assert layout["supported"], layout["errors"]


def test_driver_offset_and_semantic_are_distinct(layout_backend, tmp_path):
    _, layout = compile_layout(layout_backend, tmp_path, "layout-typed")
    assert layout["supported"]
    assert layout["inputs"] == [{"word_offset": 14, "location": 32, "component": 2,
                                  "type": "float", "bit_size": 32, "interpolation": "flat"}]
    assert layout["outputs"][0]["word_offset"] == 9


@pytest.mark.parametrize("mode,expected_type", [("integer-runtime-loop", "int"), ("layout-uint", "uint")])
def test_raw_uniform_consumer_types(layout_backend, tmp_path, mode, expected_type):
    assembly, layout = compile_layout(layout_backend, tmp_path, mode)
    assert layout["supported"]
    expected_offset = 0 if expected_type == "int" else 4099
    assert layout["uniforms"] == [{"word_offset": expected_offset, "type": expected_type, "bit_size": 32}]
    for count in (0, 1, 4):
        result = execute_assembly(assembly, {BASES["input"]: 0.125},
                                  raw_values={BASES["uniform"] + expected_offset: count})
        if expected_type == "int":
            assert result["outputs"]["fragment"]["fp16"][0] == count * 0.125
        else:
            assert result["outputs"]["fragment"]["raw_words"][0] == count


@pytest.mark.parametrize("mode", ["layout-missing", "layout-missing-interpolation", "layout-conflict",
                                  "layout-alias", "layout-ambiguous", "layout-unknown"])
def test_unavailable_reflection_does_not_fail_compilation(layout_backend, tmp_path, mode):
    assembly, layout = compile_layout(layout_backend, tmp_path, mode)
    assert assembly
    validate_contract(layout)
    assert not layout["supported"] and layout["errors"]
    if mode in {"layout-ambiguous", "layout-unknown"}:
        assert layout["uniforms"][0]["type"] == "unknown"


@pytest.mark.parametrize("mode,expected", [(1, "smooth"), (2, "flat"), (3, "noperspective")])
def test_lowered_barycentric_qualifier_survives(layout_backend, tmp_path, mode, expected):
    _, layout = compile_layout(layout_backend, tmp_path, "layout-interpolation", mode)
    assert layout["supported"]
    assert layout["inputs"][0]["interpolation"] == expected


def test_dead_accesses_are_not_reflected(layout_backend, tmp_path):
    _, layout = compile_layout(layout_backend, tmp_path, "layout-dead")
    assert layout["supported"]
    assert layout["inputs"] == layout["uniforms"] == []
    assert not layout["uses_discard"]


@pytest.mark.parametrize("mode", ["discard", "layout-demote", "layout-terminate", "layout-terminate-if"])
def test_remaining_discard_is_explicit(layout_backend, tmp_path, mode):
    _, layout = compile_layout(layout_backend, tmp_path, mode)
    assert layout["supported"]
    assert layout["uses_discard"]


@pytest.mark.parametrize("mode,success,diagnostic", [("layout-builtin", True, "unsupported builtin"),
                                                  ("layout-systemvalue", False, "unsupported intrinsic"),
                                                  ("texture", False, "texture and sampler")])
def test_unsupported_forms_have_explicit_diagnostics(layout_backend, tmp_path, mode, success, diagnostic):
    _, layout = compile_layout(layout_backend, tmp_path, mode, success=success)
    assert not layout["supported"]
    assert any(diagnostic in error for error in layout["errors"])


def test_invalid_stage_does_not_invent_stage_name(layout_backend, tmp_path):
    _, layout = compile_layout(layout_backend, tmp_path, "compute", success=False)
    assert layout is None


def test_json_is_deterministic(layout_backend, tmp_path):
    compile_layout(layout_backend, tmp_path / "first", "integer-runtime-loop")
    compile_layout(layout_backend, tmp_path / "second", "integer-runtime-loop")
    assert (tmp_path / "first/shader.layout.json").read_bytes() == (tmp_path / "second/shader.layout.json").read_bytes()


@pytest.mark.parametrize("mode,count,domain", [("input-boundary", 16383, "inputs"),
                                             ("uniform-boundary", 65535, "uniforms"),
                                             ("ubo-block-boundary", 15, "uniforms"),
                                             ("ubo-word-boundary", 4095, "uniforms")])
def test_scalar_region_bounds_and_hardware_units(layout_backend, tmp_path, mode, count, domain):
    _, legacy = compile_layout(layout_backend, tmp_path / "legacy", mode, count)
    _, hardware = compile_layout(layout_backend, tmp_path / "hardware", mode, count, "hardware")
    validate_contract(legacy)
    validate_contract(hardware, hardware=True)
    assert legacy[domain] == hardware[domain]
    assert hardware[domain][0]["word_offset"] == 65535
    _, failed = compile_layout(layout_backend, tmp_path / "invalid", mode, count + 1, success=False)
    assert failed and not failed["supported"]


@pytest.mark.parametrize("mode", ["runtime-loop", "integer-runtime-loop", "io", "integer-output", "texture"])
def test_hardware_subset_is_not_expanded(layout_backend, tmp_path, mode):
    _, layout = compile_layout(layout_backend, tmp_path, mode, option="hardware", success=False)
    assert not layout["supported"]


def test_dump_rejects_path_traversal_and_keeps_stderr_assembly(layout_backend, tmp_path):
    environment = dict(os.environ, MGC_ASM_DUMP=str(tmp_path), MGC_LAYOUT_STAGE="../escape")
    result = run_checked([str(layout_backend["binary"]), "flat"], env=environment)
    assert not list(tmp_path.iterdir())
    assert "invalid shader dump stage name" in result.stderr
    environment["MGC_ASM_DUMP"] = "-"
    result = run_checked([str(layout_backend["binary"]), "flat"], env=environment)
    assert result.stdout in result.stderr


@pytest.fixture(scope="module")
def reflected_glsl(layout_backend, tmp_path_factory):
    if not (ROOT / "tools/mgc_harness").is_file():
        pytest.skip("existing GLSL harness required")
    libraries = list((BUILD / "src/gallium/targets/dri").glob("libgallium-*.so"))
    assert len(libraries) == 1
    directory = layout_backend["directory"]
    archive = directory / "private-libmgc.a"
    original_archive = BUILD / "src/gallium/drivers/mgc/libmgc.a"
    members = run_checked(["ar", "t", str(original_archive)]).stdout.splitlines()
    compiler_name = layout_backend["compiler"].name
    objects = [str(layout_backend["compiler"]) if pathlib.Path(member).name == compiler_name else
               str(pathlib.Path(member).resolve()) for member in members]
    run_checked(["ar", "rcs", str(archive), *objects])
    target = str(libraries[0].relative_to(BUILD))
    library = directory / libraries[0].name
    link_command = [str(library) if token == target else str(archive)
                    if token == "src/gallium/drivers/mgc/libmgc.a" else token
                    for token in layout_backend["command"](target)]
    run_checked(link_command, cwd=BUILD)
    (directory / "swrast_dri.so").symlink_to(library)
    dumps = tmp_path_factory.mktemp("reflected_glsl")
    programs = {}
    for name in ("01_flat_color", "02_varying", "03_uniform", "09_runtime_loop",
                 "10_branch_discard", "12_runtime_arithmetic"):
        dump = dumps / name
        dump.mkdir()
        environment = mesa_environment(BUILD, dump)
        environment["LD_LIBRARY_PATH"] = str(directory) + os.pathsep + environment["LD_LIBRARY_PATH"]
        environment["LIBGL_DRIVERS_PATH"] = str(directory)
        run_checked([str(ROOT / "tools/mgc_harness"), str(ROOT / "shaders" / name / "vert.glsl"),
                     str(ROOT / "shaders" / name / "frag.glsl")], env=environment, timeout=60)
        programs[name] = {stage: {"asm": (dump / f"{stage}.asm").read_text(),
                                  "layout": json.loads((dump / f"{stage}.layout.json").read_text())}
                          for stage in ("vs", "fs")}
    return programs


def test_fresh_glsl_layouts_are_supported(reflected_glsl):
    for stages in reflected_glsl.values():
        for artifact in stages.values():
            validate_contract(artifact["layout"])
            assert artifact["layout"]["supported"], artifact["layout"]["errors"]


def test_fresh_glsl_position_color_and_semantic_link(reflected_glsl):
    stages = reflected_glsl["02_varying"]
    vertex_layout = stages["vs"]["layout"]
    fragment_layout = stages["fs"]["layout"]
    outputs = {(entry["location"], entry["component"]): entry["word_offset"]
               for entry in vertex_layout["outputs"]}
    assert {entry["location"] for entry in fragment_layout["outputs"]} == {2}
    assert [outputs[(0, component)] for component in range(4)] == [0, 1, 2, 3]
    colors = [0.25, 0.5, 0.75]
    values = {BASES["input"] + component: value for component, value in enumerate([0.125, 0.25, 0.5, 1])}
    values.update({BASES["input"] + 4 + component: value for component, value in enumerate(colors)})
    vertex = execute_assembly(stages["vs"]["asm"], values)
    fragment_values = {}
    for entry in fragment_layout["inputs"]:
        key = entry["location"], entry["component"]
        assert entry["interpolation"] == "smooth"
        fragment_values[BASES["input"] + entry["word_offset"]] = vertex["outputs"]["vertex"]["fp16"][outputs[key]]
    fragment = execute_assembly(stages["fs"]["asm"], fragment_values)
    assert fragment["outputs"]["fragment"]["fp16"][:4] == colors + [1]


def test_fresh_glsl_uniforms_and_raw_loop_counter(reflected_glsl):
    assert reflected_glsl["03_uniform"]["fs"]["layout"]["uniforms"] == [
        {"word_offset": 3, "type": "float", "bit_size": 32}]
    fragment = reflected_glsl["09_runtime_loop"]["fs"]
    assert fragment["layout"]["uniforms"] == [{"word_offset": 0, "type": "int", "bit_size": 32}]
    for count in (0, 1, 4):
        result = execute_assembly(fragment["asm"], {BASES["input"]: 0.25,
                                                   BASES["input"] + 1: 0.5,
                                                   BASES["input"] + 2: 0.75},
                                  raw_values={BASES["uniform"]: count})
        multiplier = count * (count + 1) / 2
        assert result["outputs"]["fragment"]["fp16"][:4] == [0.25 * multiplier, 0.5 * multiplier,
                                                               0.75 * multiplier, 1]


def test_fresh_glsl_discard_flag(reflected_glsl):
    assert reflected_glsl["10_branch_discard"]["fs"]["layout"]["uses_discard"]
    assert not reflected_glsl["10_branch_discard"]["vs"]["layout"]["uses_discard"]


@pytest.mark.parametrize("name", ["01_flat_color", "02_varying"])
def test_fresh_glsl_hardware_layout(reflected_glsl, layout_backend, tmp_path, name):
    environment = mesa_environment(BUILD, tmp_path, ProgramFormat.HARDWARE)
    directory = layout_backend["directory"]
    environment["LD_LIBRARY_PATH"] = str(directory) + os.pathsep + environment["LD_LIBRARY_PATH"]
    environment["LIBGL_DRIVERS_PATH"] = str(directory)
    run_checked([str(ROOT / "tools/mgc_harness"), str(ROOT / "shaders" / name / "vert.glsl"),
                 str(ROOT / "shaders" / name / "frag.glsl")], env=environment, timeout=60)
    for stage in ("vs", "fs"):
        layout = json.loads((tmp_path / f"{stage}.layout.json").read_text())
        validate_contract(layout, hardware=True)
        assert layout["supported"]
        for domain in ("inputs", "outputs", "uniforms"):
            assert layout[domain] == reflected_glsl[name][stage]["layout"][domain]
