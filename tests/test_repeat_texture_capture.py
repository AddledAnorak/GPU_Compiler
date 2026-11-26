"""Bounded C repeat capture guards; no device sampling or rendered-output claims."""

import pytest

from test_texture_capture import (
    BINDING_REJECTIONS,
    DISCARD_CONTRACT_REJECTIONS,
    HOST_TEXTURE,
    RESOURCE_REJECTIONS,
    SAMPLER,
    SAMPLER_REJECTIONS,
    SOURCE,
    TRANSACTIONAL_BINDINGS,
    VALID_TEXTURES,
    assert_texture_capture,
    build_texture_guard,
    read_capture,
    run_guard,
    texture_guard,
)


REPEAT_VERSIONS = (3, 4)
EXTRA_VALID_CASES = ("valid_wrap_r", "valid_map_released")
REPEAT_SAMPLER_REJECTIONS = tuple(
    case for case in SAMPLER_REJECTIONS if case not in ("min_linear", "mag_linear")
) + (
    "min_wrong", "mag_wrong", "both_wrong", "mip_unknown",
    "wrap_both_clamp", "wrap_s_mirror", "wrap_t_mirror", "wrap_both_mirror",
)
LAYOUT_REJECTIONS = (
    "abi_zero", "abi_unknown", "abi_vs_unknown", "abi_hardware", "uses_texture_false",
    "abi_matching_wrong_filter", "abi_matching_wrong_wrap", "legacy_unknown_opt_in",
    "layout_version_zero", "layout_version_one", "layout_version_three",
    "layout_host_version_zero", "layout_host_version_unknown", "layout_host_version_float",
    "layout_filter_wrong", "layout_filter_case", "layout_filter_unknown",
    "layout_wrap_clamp", "layout_wrap_case", "layout_wrap_unknown",
    "layout_texture_slot", "layout_sampler_slot", "layout_block", "layout_max_width",
    "layout_max_height", "layout_duplicate", "layout_unknown", "layout_missing_field",
    "layout_missing_texture", "layout_stage", "layout_shader_abi",
)
DRAW_REJECTIONS = (
    "draw_lines", "draw_instanced", "draw_start_instance", "draw_restart",
    "draw_range", "draw_user_indices",
)
COMPILE_REJECTIONS = (
    "compile_default", "compile_zero", "compile_unknown", "compile_negative",
    "compile_leading_zero", "compile_float", "compile_space",
    "compile_hardware_default", "compile_hardware_zero",
) + tuple(f"compile_hardware_{version}" for version in (1, 2, 3, 4))


HOOKS = r'''
static void
replace_repeat_layout(struct mgc_shader *shader, const char *original,
                      const char *replacement)
{
   const char *value = strstr(shader->layout_json, original);
   assert(value);
   shader->layout_json = ralloc_asprintf(shader, "%.*s%s%s",
      (int)(value - shader->layout_json), shader->layout_json,
      replacement, value + strlen(original));
   assert(shader->layout_json);
}

static bool
change_repeat_contract(const char *name, struct mgc_context *ctx)
{
   struct mgc_shader *shader = ctx->shaders[1]->shader;
   if (is_case(name, "abi_zero")) shader->abi.host_texture_version = 0;
   else if (!strncmp(name, "abi_version_", 12))
      shader->abi.host_texture_version = strtoul(name + 12, NULL, 10);
   else if (is_case(name, "abi_unknown") || is_case(name, "legacy_unknown_opt_in"))
      shader->abi.host_texture_version = 11;
   else if (is_case(name, "abi_vs_unknown")) ctx->shaders[0]->shader->abi.host_texture_version = 11;
   else if (is_case(name, "abi_hardware")) shader->abi.target = MGC_TARGET_FP16_HARDWARE_V1;
   else if (is_case(name, "uses_texture_false")) shader->uses_host_texture = false;
   else if (is_case(name, "abi_matching_wrong_filter")) {
      shader->abi.host_texture_version = @OTHER_VERSION@;
      replace_repeat_layout(shader, "\"host_texture\":{\"version\":@VERSION@",
                            "\"host_texture\":{\"version\":@OTHER_VERSION@");
   } else if (is_case(name, "abi_matching_wrong_wrap")) {
      shader->abi.host_texture_version = @CLAMP_VERSION@;
      replace_repeat_layout(shader, "\"host_texture\":{\"version\":@VERSION@",
                            "\"host_texture\":{\"version\":@CLAMP_VERSION@");
   } else {
      static const struct {
         const char *name;
         const char *original;
         const char *replacement;
      } mutations[] = {
         {"layout_version_zero", "{\"version\":2,", "{\"version\":0,"},
         {"layout_version_one", "{\"version\":2,", "{\"version\":1,"},
         {"layout_version_three", "{\"version\":2,", "{\"version\":3,"},
         {"layout_host_version_zero", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":0"},
         {"layout_host_version_1", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":1"},
         {"layout_host_version_2", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":2"},
         {"layout_host_version_3", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":3"},
         {"layout_host_version_4", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":4"},
         {"layout_host_version_unknown", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":11"},
         {"layout_host_version_float", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":@VERSION@.0"},
         {"layout_filter_wrong", "\"filter\":\"@FILTER@\"", "\"filter\":\"@OTHER_FILTER@\""},
         {"layout_filter_case", "\"filter\":\"@FILTER@\"", "\"filter\":\"@CASE_FILTER@\""},
         {"layout_filter_unknown", "\"filter\":\"@FILTER@\"", "\"filter\":\"cubic\""},
         {"layout_wrap_clamp", "\"wrap\":\"repeat\"", "\"wrap\":\"clamp-to-edge\""},
         {"layout_wrap_case", "\"wrap\":\"repeat\"", "\"wrap\":\"Repeat\""},
         {"layout_wrap_unknown", "\"wrap\":\"repeat\"", "\"wrap\":\"mirror-repeat\""},
         {"layout_texture_slot", "\"texture\":0", "\"texture\":1"},
         {"layout_sampler_slot", "\"sampler\":0", "\"sampler\":1"},
         {"layout_block", "\"storage_block\":15", "\"storage_block\":14"},
         {"layout_max_width", "\"max_width\":16", "\"max_width\":17"},
         {"layout_max_height", "\"max_height\":16", "\"max_height\":17"},
         {"layout_duplicate", "\"filter\":\"@FILTER@\"", "\"filter\":\"@FILTER@\",\"filter\":\"@FILTER@\""},
         {"layout_unknown", "\"filter\":\"@FILTER@\"", "\"filter\":\"@FILTER@\",\"extra\":0"},
         {"layout_missing_field", ",\"filter\":\"@FILTER@\"", ""},
         {"layout_missing_texture", ",\"host_texture\":{\"version\":@VERSION@,\"texture\":0,\"sampler\":0,\"storage_block\":15,\"max_width\":16,\"max_height\":16,\"filter\":\"@FILTER@\",\"wrap\":\"repeat\"}", ""},
         {"layout_stage", "\"stage\":\"fragment\"", "\"stage\":\"vertex\""},
         {"layout_shader_abi", "\"shader_abi_version\":1", "\"shader_abi_version\":2"},
      };
      for (unsigned index = 0; index < ARRAY_SIZE(mutations); index++) {
         if (is_case(name, mutations[index].name)) {
            replace_repeat_layout(shader, mutations[index].original, mutations[index].replacement);
            return true;
         }
      }
      return false;
   }
   return true;
}

static bool
change_repeat_sampler(const char *name, struct pipe_sampler_state *sampler)
{
   if (is_case(name, "min_wrong")) sampler->min_img_filter = @OTHER_FILTER_ENUM@;
   else if (is_case(name, "mag_wrong")) sampler->mag_img_filter = @OTHER_FILTER_ENUM@;
   else if (is_case(name, "both_wrong")) {
      sampler->min_img_filter = @OTHER_FILTER_ENUM@;
      sampler->mag_img_filter = @OTHER_FILTER_ENUM@;
   } else if (is_case(name, "mip_unknown")) sampler->min_mip_filter = 3;
   else if (is_case(name, "wrap_both_clamp")) {
      sampler->wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler->wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   } else if (is_case(name, "wrap_s_mirror")) sampler->wrap_s = PIPE_TEX_WRAP_MIRROR_REPEAT;
   else if (is_case(name, "wrap_t_mirror")) sampler->wrap_t = PIPE_TEX_WRAP_MIRROR_REPEAT;
   else if (is_case(name, "wrap_both_mirror")) {
      sampler->wrap_s = PIPE_TEX_WRAP_MIRROR_REPEAT;
      sampler->wrap_t = PIPE_TEX_WRAP_MIRROR_REPEAT;
   } else if (is_case(name, "valid_wrap_r")) sampler->wrap_r = PIPE_TEX_WRAP_MIRROR_REPEAT;
   else return false;
   return true;
}

static bool
change_repeat_storage(const char *name, struct pipe_context *pipe,
                      struct pipe_resource *resource)
{
   struct mgc_resource *storage = mgc_resource(resource);
   if (is_case(name, "host_execution_owned")) {
      simple_mtx_lock(&mgc_screen(pipe->screen)->lock);
      storage->active_host_executions = 1;
      simple_mtx_unlock(&mgc_screen(pipe->screen)->lock);
   } else if (is_case(name, "valid_map_released")) {
      struct pipe_box box = {.width = 16, .height = 16, .depth = 1};
      struct pipe_transfer *transfer = NULL;
      assert(pipe->texture_map(pipe, resource, 0, PIPE_MAP_READ, &box, &transfer));
      assert(transfer && storage->active_maps == 1);
      pipe->texture_unmap(pipe, transfer);
      assert(!storage->active_maps && !mgc_context(pipe)->transfers);
   } else return false;
   return true;
}

static void
change_repeat_draw(const char *name, struct pipe_draw_info *info,
                   struct pipe_draw_start_count_bias *draw)
{
   if (is_case(name, "draw_lines")) info->mode = MESA_PRIM_LINES;
   else if (is_case(name, "draw_instanced")) info->instance_count = 2;
   else if (is_case(name, "draw_start_instance")) info->start_instance = 1;
   else if (is_case(name, "draw_restart")) info->primitive_restart = true;
   else if (is_case(name, "draw_range")) draw->start = UINT_MAX;
   else if (is_case(name, "draw_user_indices")) {
      info->index_size = 2;
      info->has_user_indices = true;
   }
}

static int
compile_repeat_guard(const char *name)
{
   assert(MGC_RUNTIME_ABI_VERSION == 1 && MGC_ABI_VERSION == 1);
   assert(MGC_HOST_TEXTURE_VERSION == 1 && MGC_HOST_TEXTURE_LINEAR_VERSION == 2);
   assert(MGC_HOST_TEXTURE_REPEAT_VERSION == 3 && MGC_HOST_TEXTURE_LINEAR_REPEAT_VERSION == 4);
   bool accepted = !strncmp(name, "compile_accept_", 15);
   const char *version = NULL;
   if (accepted) version = name + 15;
   else if (is_case(name, "compile_zero")) version = "0";
   else if (is_case(name, "compile_unknown")) version = "11";
   else if (is_case(name, "compile_negative")) version = "-1";
   else if (is_case(name, "compile_leading_zero")) version = "03";
   else if (is_case(name, "compile_float")) version = "3.0";
   else if (is_case(name, "compile_space")) version = " 3";
   else if (!strncmp(name, "compile_hardware_", 17)) {
      assert(!setenv("MGC_COMPILE_TARGET", "fp16-hardware-v1", 1));
      if (!is_case(name, "compile_hardware_default"))
         version = is_case(name, "compile_hardware_zero") ? "0" : name + 17;
   } else assert(is_case(name, "compile_default"));
   if (version) assert(!setenv("MGC_HOST_TEXTURE", version, 1));
   else assert(!unsetenv("MGC_HOST_TEXTURE"));
   glsl_type_singleton_init_or_ref();
   struct pipe_screen *screen = mgc_create_screen(NULL);
   assert(screen);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   assert(pipe);
   nir_builder builder = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      mgc_test_get_compiler_options(), "repeat_capture_compile_guard");
   nir_variable *output = io_variable(builder.shader, nir_var_shader_out,
      glsl_vec4_type(), "color", FRAG_RESULT_COLOR, 0);
   nir_tex_instr *texture = nir_tex_instr_create(builder.shader, 1);
   texture->op = nir_texop_tex;
   texture->sampler_dim = GLSL_SAMPLER_DIM_2D;
   texture->dest_type = nir_type_float32;
   texture->coord_components = 2;
   texture->src[0].src_type = nir_tex_src_coord;
   texture->src[0].src = nir_src_for_ssa(nir_imm_vec2(&builder, 0.25, 0.75));
   nir_def_init(&texture->instr, &texture->def, 4, 32);
   nir_builder_instr_insert(&builder, &texture->instr);
   nir_store_var(&builder, output, &texture->def, 15);
   struct pipe_shader_state shader = {
      .type = PIPE_SHADER_IR_NIR, .ir.nir = builder.shader,
      .report_compile_error = true,
   };
   struct mgc_shader_state *state = pipe->create_fs_state(pipe, &shader);
   if (accepted) {
      assert(state && !shader.error_message);
      assert(state->shader->uses_host_texture && state->shader->abi_version == 1);
      assert(state->shader->abi.host_texture_version == strtoul(version, NULL, 10));
      pipe->delete_fs_state(pipe, state);
   } else assert(!state && shader.error_message);
   free(shader.error_message);
   struct mgc_runtime_status status;
   mgc_context_get_runtime_status(pipe, &status);
   assert(status.error == (accepted ? 0 : PIPE_ERROR_BAD_INPUT));
   pipe->destroy(pipe);
   assert(mgc_screen(screen)->allocated_bytes == 0);
   screen->destroy(screen);
   glsl_type_singleton_decref();
   printf("{\"accepted\":0,\"rejected\":0,\"error\":%d,\"callbacks\":0,"
          "\"compile_accepted\":%u,\"compile_rejected\":%u}\n",
          status.error, accepted, !accepted);
   return 0;
}

static int repeat_capture_guard_main(int argc, char **argv);

int
main(int argc, char **argv)
{
   assert(argc == 2);
   if (!strncmp(argv[1], "compile_", 8)) return compile_repeat_guard(argv[1]);
   return repeat_capture_guard_main(argc, argv);
}
'''


@pytest.fixture(scope="module", params=REPEAT_VERSIONS, ids=("nearest-repeat-v3", "linear-repeat-v4"))
def repeat_texture_guard(request, tmp_path_factory):
    version = request.param
    texture_filter = "nearest" if version == 3 else "linear"
    other_filter = "linear" if version == 3 else "nearest"
    hooks = HOOKS
    for original, replacement in {
        "@VERSION@": str(version),
        "@OTHER_VERSION@": str(7 - version),
        "@CLAMP_VERSION@": str(version - 2),
        "@FILTER@": texture_filter,
        "@OTHER_FILTER@": other_filter,
        "@CASE_FILTER@": texture_filter.capitalize(),
        "@OTHER_FILTER_ENUM@": "PIPE_TEX_FILTER_" + other_filter.upper(),
    }.items():
        hooks = hooks.replace(original, replacement)
    replacements = (
        ('if (opt_in) assert(!setenv("MGC_HOST_TEXTURE", "1", 1));',
         f'if (opt_in) assert(!setenv("MGC_HOST_TEXTURE", "{version}", 1));'),
        (".min_img_filter = PIPE_TEX_FILTER_NEAREST,",
         f".min_img_filter = PIPE_TEX_FILTER_{texture_filter.upper()},"),
        (".mag_img_filter = PIPE_TEX_FILTER_NEAREST,",
         f".mag_img_filter = PIPE_TEX_FILTER_{texture_filter.upper()},"),
        (".wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,", ".wrap_s = PIPE_TEX_WRAP_REPEAT,"),
        (".wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,", ".wrap_t = PIPE_TEX_WRAP_REPEAT,"),
        ('state->wrap_s = PIPE_TEX_WRAP_REPEAT;', 'state->wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE;'),
        ('state->wrap_t = PIPE_TEX_WRAP_REPEAT;', 'state->wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;'),
        ('known |= change_discard_contract(name, ctx->shaders[1]->shader);',
         'known |= change_discard_contract(name, ctx->shaders[1]->shader);\n'
         '   known |= change_repeat_contract(name, ctx);'),
        ('known |= change_sampler(name, &sampler);',
         'known |= change_sampler(name, &sampler);\n'
         '      known |= change_repeat_sampler(name, &sampler);'),
        ('((struct pipe_sampler_state *)state)->min_img_filter = PIPE_TEX_FILTER_LINEAR;',
         f'((struct pipe_sampler_state *)state)->min_img_filter = PIPE_TEX_FILTER_{other_filter.upper()};'),
        ('known |= change_storage(name, resource);',
         'known |= change_storage(name, resource);\n'
         '      known |= change_repeat_storage(name, pipe, resource);'),
        ('assert(known);', 'assert(known || !strncmp(name, "draw_", 5));'),
        ('pipe->draw_vbo(pipe, &info, 7, NULL, &draw, 1);',
         'change_repeat_draw(name, &info, &draw);\n'
         '   pipe->draw_vbo(pipe, &info, 7, NULL, &draw, 1);'),
        ('struct mgc_resource *storage = mgc_resource(resource);\n      resource->width0',
         'struct mgc_resource *storage = mgc_resource(resource);\n'
         '      if (is_case(name, "host_execution_owned")) {\n'
         '         assert(storage->active_host_executions == 1);\n'
         '         simple_mtx_lock(&mgc_screen(screen)->lock);\n'
         '         storage->active_host_executions = 0;\n'
         '         simple_mtx_unlock(&mgc_screen(screen)->lock);\n'
         '      }\n'
         '      assert(!storage->active_maps && !storage->active_host_executions);\n'
         '      resource->width0'),
        ('int\nmain(int argc, char **argv)',
         hooks + '\nstatic int\nrepeat_capture_guard_main(int argc, char **argv)'),
    )
    contents = SOURCE.read_text()
    for original, replacement in replacements:
        assert contents.count(original) == 1, original
        contents = contents.replace(original, replacement, 1)
    source = tmp_path_factory.mktemp(f"repeat-texture-capture-v{version}-source") / "guard.c"
    source.write_text(contents)
    return build_texture_guard(tmp_path_factory, source), version, texture_filter


@pytest.mark.parametrize("case", VALID_TEXTURES + TRANSACTIONAL_BINDINGS + EXTRA_VALID_CASES)
def test_direct_c_repeat_capture_and_owned_snapshots(repeat_texture_guard, tmp_path, case):
    helper, version, texture_filter = repeat_texture_guard
    status = run_guard(helper, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["callbacks"] == 0 and status["sequence"] == 1
    assert (status["operation_error"] != 0) == (case in TRANSACTIONAL_BINDINGS)
    assert_texture_capture(
        read_capture(tmp_path), case,
        host_texture=dict(HOST_TEXTURE, version=version, filter=texture_filter, wrap="repeat"),
        sampler=dict(SAMPLER, min_filter=texture_filter, mag_filter=texture_filter,
                     wrap_s="repeat", wrap_t="repeat"),
    )


@pytest.mark.parametrize("case", REPEAT_SAMPLER_REJECTIONS + RESOURCE_REJECTIONS +
                         BINDING_REJECTIONS + DISCARD_CONTRACT_REJECTIONS +
                         LAYOUT_REJECTIONS + DRAW_REJECTIONS + (
                             "host_execution_owned", "custom_texture",
                             "custom_texture_option_zero", "callback_disabled",
                         ))
def test_direct_c_repeat_rejects_before_capture_or_callback(repeat_texture_guard, tmp_path, case):
    helper, _, _ = repeat_texture_guard
    status = run_guard(helper, tmp_path, case)
    assert status["accepted"] == 0 and status["rejected"] == 1
    assert status["error"] != 0 and status["operation_error"] == 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ("abi_version", "layout_host_version"))
@pytest.mark.parametrize("wrong_version", (1, 2, 3, 4))
def test_direct_c_repeat_requires_exact_compiled_version(repeat_texture_guard, tmp_path,
                                                       field, wrong_version):
    helper, version, _ = repeat_texture_guard
    if wrong_version == version:
        assert run_guard(helper, tmp_path, "valid")["accepted"] == 1
        assert read_capture(tmp_path)["shaders"]["fs"]["layout"]["host_texture"]["version"] == version
        return
    status = run_guard(helper, tmp_path, f"{field}_{wrong_version}")
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ("legacy_default", "legacy_opt_in", "legacy_custom",
                                 "legacy_custom_opt_in"))
def test_repeat_option_preserves_untextured_v1_and_callback_abi(repeat_texture_guard, tmp_path, case):
    helper, _, _ = repeat_texture_guard
    status = run_guard(helper, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    custom = "custom" in case
    assert status["callbacks"] == int(custom)
    if custom:
        assert list(tmp_path.iterdir()) == []
    else:
        artifact = read_capture(tmp_path)
        assert artifact["schema"] == "mgc-host-draw-v1"
        assert artifact["shader_abi_version"] == 1 and "textures" not in artifact
        for shader in artifact["shaders"].values():
            assert shader["layout"]["version"] == 1 and "host_texture" not in shader["layout"]


def test_repeat_capture_keeps_existing_payload_keys(texture_guard, repeat_texture_guard, tmp_path):
    helper, _, _ = repeat_texture_guard
    captures = []
    for binary, name in ((texture_guard, "clamp"), (helper, "repeat")):
        directory = tmp_path / name
        directory.mkdir()
        assert run_guard(binary, directory, "valid")["accepted"] == 1
        captures.append(read_capture(directory))
    clamp, repeat = captures
    assert clamp.keys() == repeat.keys()
    for stage in ("vs", "fs"):
        assert clamp["shaders"][stage].keys() == repeat["shaders"][stage].keys()
        assert clamp["shaders"][stage]["layout"].keys() == repeat["shaders"][stage]["layout"].keys()
    assert repeat["shaders"]["fs"]["layout"]["host_texture"].keys() == HOST_TEXTURE.keys()
    assert clamp["textures"][0].keys() == repeat["textures"][0].keys()
    assert clamp["textures"][0]["sampler"].keys() == repeat["textures"][0]["sampler"].keys()
    assert dict(repeat["textures"][0], sampler=SAMPLER) == clamp["textures"][0]


@pytest.mark.parametrize("case", COMPILE_REJECTIONS)
def test_direct_c_default_unknown_and_hardware_reject_repeat(repeat_texture_guard, tmp_path, case):
    helper, _, _ = repeat_texture_guard
    status = run_guard(helper, tmp_path, case)
    assert status["compile_rejected"] == 1 and status["compile_accepted"] == 0
    assert status["error"] != 0 and status["accepted"] == 0 and status["rejected"] == 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", (1, 2, 3, 4))
def test_direct_c_legacy_option_accepts_exactly_supported_versions(repeat_texture_guard, tmp_path, version):
    helper, _, _ = repeat_texture_guard
    status = run_guard(helper, tmp_path, f"compile_accept_{version}")
    assert status["compile_accepted"] == 1 and status["compile_rejected"] == 0
    assert status["error"] == 0 and status["accepted"] == 0 and status["rejected"] == 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []
