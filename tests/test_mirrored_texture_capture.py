"""Direct C mirrored-repeat capture guards; no sampling or rendered-output claims."""

import pytest

from test_repeat_texture_capture import (
    COMPILE_REJECTIONS,
    DRAW_REJECTIONS,
    EXTRA_VALID_CASES,
    HOOKS as REPEAT_HOOKS,
    LAYOUT_REJECTIONS,
)
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


MIRRORED_VERSIONS = (5, 6)
WRAP_REJECTIONS = tuple(
    f"sampler_wrap_{axis}_{wrap}"
    for axis in ("s", "t", "both")
    for wrap in ("repeat", "clamp", "clamp_edge", "clamp_border",
                 "mirror_clamp", "mirror_clamp_edge", "mirror_clamp_border")
)
MIRRORED_SAMPLER_REJECTIONS = tuple(
    case for case in SAMPLER_REJECTIONS if case not in ("min_linear", "mag_linear")
) + ("min_wrong", "mag_wrong", "both_wrong", "mip_unknown", "wrap_both_clamp",
     "wrap_s_repeat", "wrap_t_repeat", "wrap_both_repeat") + WRAP_REJECTIONS
EXTRA_LAYOUT_REJECTIONS = (
    "abi_sampler_filter_mismatch", "abi_sampler_clamp_mismatch", "abi_sampler_repeat_mismatch",
    "layout_wrap_repeat", "layout_wrap_mirror_clamp", "layout_wrap_pair",
    "layout_wrap_duplicate", "layout_version_duplicate", "layout_filter_null",
    "layout_version_string", "layout_host_null", "layout_missing_version",
    "layout_missing_texture_slot", "layout_missing_sampler_slot", "layout_missing_block",
    "layout_missing_width", "layout_missing_height", "layout_missing_wrap",
    "layout_vs_texture", "layout_trailing", "layout_truncated", "layout_budget",
    "assembly_budget", "buffer_budget",
)
MIRRORED_COMPILE_REJECTIONS = COMPILE_REJECTIONS + (
    "compile_hardware_5", "compile_hardware_6", "compile_hardware_7",
    "compile_empty", "compile_plus", "compile_trailing_space", "compile_silent_unknown",
)


EXTRA_HOOKS = r'''
static bool
change_mirrored_contract(const char *name, struct mgc_context *ctx)
{
   struct mgc_shader *shader = ctx->shaders[1]->shader;
   if (is_case(name, "abi_sampler_filter_mismatch")) {
      shader->abi.host_texture_version = @OTHER_VERSION@;
      replace_repeat_layout(shader, "\"host_texture\":{\"version\":@VERSION@",
                            "\"host_texture\":{\"version\":@OTHER_VERSION@");
      replace_repeat_layout(shader, "\"filter\":\"@FILTER@\"", "\"filter\":\"@OTHER_FILTER@\"");
   } else if (is_case(name, "abi_sampler_clamp_mismatch") ||
              is_case(name, "abi_sampler_repeat_mismatch")) {
      bool repeat = is_case(name, "abi_sampler_repeat_mismatch");
      shader->abi.host_texture_version = repeat ? @REPEAT_VERSION@ : @CLAMP_VERSION@;
      replace_repeat_layout(shader, "\"host_texture\":{\"version\":@VERSION@",
                            repeat ? "\"host_texture\":{\"version\":@REPEAT_VERSION@" :
                                     "\"host_texture\":{\"version\":@CLAMP_VERSION@");
      replace_repeat_layout(shader, "\"wrap\":\"mirrored-repeat\"",
                            repeat ? "\"wrap\":\"repeat\"" : "\"wrap\":\"clamp-to-edge\"");
   } else if (is_case(name, "layout_vs_texture")) {
      struct mgc_shader *vertex = ctx->shaders[0]->shader;
      vertex->layout_json = ralloc_strdup(vertex, shader->layout_json);
      assert(vertex->layout_json);
      replace_repeat_layout(vertex, "\"stage\":\"fragment\"", "\"stage\":\"vertex\"");
      vertex->uses_host_texture = true;
   } else if (is_case(name, "layout_trailing")) {
      shader->layout_json = ralloc_asprintf(shader, "%s{}", shader->layout_json);
      assert(shader->layout_json);
   } else if (is_case(name, "layout_truncated")) {
      size_t length = strlen(shader->layout_json);
      while (length && strchr(" \t\r\n", shader->layout_json[length - 1]))
         length--;
      assert(length && shader->layout_json[length - 1] == '}');
      shader->layout_json[length - 1] = '\0';
   } else if (is_case(name, "assembly_budget") || is_case(name, "layout_budget")) {
      char *oversized = ralloc_array(shader, char, MGC_HOST_CAPTURE_MAX_TEXT_BYTES + 2);
      assert(oversized);
      memset(oversized, ' ', MGC_HOST_CAPTURE_MAX_TEXT_BYTES + 1);
      oversized[MGC_HOST_CAPTURE_MAX_TEXT_BYTES + 1] = '\0';
      if (is_case(name, "assembly_budget")) shader->asm_text = oversized;
      else shader->layout_json = oversized;
   } else if (is_case(name, "buffer_budget")) {
      ctx->vertex_buffers[0].buffer.resource->width0 = MGC_HOST_CAPTURE_MAX_BUFFER_BYTES + 1;
   } else {
      static const struct {
         const char *name;
         const char *original;
         const char *replacement;
      } mutations[] = {
         {"layout_host_version_5", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":5"},
         {"layout_host_version_6", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":6"},
         {"layout_wrap_repeat", "\"wrap\":\"mirrored-repeat\"", "\"wrap\":\"repeat\""},
         {"layout_wrap_mirror_clamp", "\"wrap\":\"mirrored-repeat\"", "\"wrap\":\"mirror-clamp-to-edge\""},
         {"layout_wrap_pair", "\"wrap\":\"mirrored-repeat\"", "\"wrap\":[\"mirrored-repeat\",\"repeat\"]"},
         {"layout_wrap_duplicate", "\"wrap\":\"mirrored-repeat\"", "\"wrap\":\"mirrored-repeat\",\"wrap\":\"mirrored-repeat\""},
         {"layout_version_duplicate", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":@VERSION@,\"version\":@VERSION@"},
         {"layout_filter_null", "\"filter\":\"@FILTER@\"", "\"filter\":null"},
         {"layout_version_string", "\"host_texture\":{\"version\":@VERSION@", "\"host_texture\":{\"version\":\"@VERSION@\""},
         {"layout_host_null", "\"host_texture\":{\"version\":@VERSION@,\"texture\":0,\"sampler\":0,\"storage_block\":15,\"max_width\":16,\"max_height\":16,\"filter\":\"@FILTER@\",\"wrap\":\"mirrored-repeat\"}", "\"host_texture\":null"},
         {"layout_missing_version", "\"host_texture\":{\"version\":@VERSION@,", "\"host_texture\":{"},
         {"layout_missing_texture_slot", ",\"texture\":0", ""},
         {"layout_missing_sampler_slot", ",\"sampler\":0", ""},
         {"layout_missing_block", ",\"storage_block\":15", ""},
         {"layout_missing_width", ",\"max_width\":16", ""},
         {"layout_missing_height", ",\"max_height\":16", ""},
         {"layout_missing_wrap", ",\"wrap\":\"mirrored-repeat\"", ""},
      };
      for (unsigned index = 0; index < ARRAY_SIZE(mutations); index++) {
         if (!is_case(name, mutations[index].name))
            continue;
         replace_repeat_layout(shader, mutations[index].original, mutations[index].replacement);
         return true;
      }
      return false;
   }
   return true;
}

static bool
change_mirrored_sampler(const char *name, struct pipe_sampler_state *sampler)
{
   static const struct {
      const char *name;
      unsigned wrap;
   } wraps[] = {
      {"repeat", PIPE_TEX_WRAP_REPEAT}, {"clamp", PIPE_TEX_WRAP_CLAMP},
      {"clamp_edge", PIPE_TEX_WRAP_CLAMP_TO_EDGE},
      {"clamp_border", PIPE_TEX_WRAP_CLAMP_TO_BORDER},
      {"mirror_clamp", PIPE_TEX_WRAP_MIRROR_CLAMP},
      {"mirror_clamp_edge", PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE},
      {"mirror_clamp_border", PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER},
   };
   const char *axes[] = {"s", "t", "both"};
   for (unsigned index = 0; index < ARRAY_SIZE(wraps); index++) {
      for (unsigned axis = 0; axis < ARRAY_SIZE(axes); axis++) {
         char expected[80];
         snprintf(expected, sizeof(expected), "sampler_wrap_%s_%s", axes[axis], wraps[index].name);
         if (!is_case(name, expected))
            continue;
         if (axis != 1) sampler->wrap_s = wraps[index].wrap;
         if (axis != 0) sampler->wrap_t = wraps[index].wrap;
         return true;
      }
   }
   return false;
}

static bool
exercise_mirrored_references(const char *name, struct pipe_context *pipe,
                             struct pipe_resource *resource)
{
   if (!is_case(name, "valid_rebind_ownership"))
      return false;
   struct mgc_context *ctx = mgc_context(pipe);
   struct pipe_sampler_view *view = NULL;
   pipe_sampler_view_reference(&view, ctx->sampler_views[1][0]);
   assert(view->reference.count == 2 && resource->reference.count == 2);
   pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 0, &view);
   assert(view->reference.count == 2);
   pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 0, NULL);
   assert(!ctx->sampler_views[1][0] && view->reference.count == 1);
   assert(resource->reference.count == 2);
   pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 0, &view);
   assert(ctx->sampler_views[1][0] == view && view->reference.count == 2);
   pipe_sampler_view_reference(&view, NULL);
   assert(ctx->sampler_views[1][0]->reference.count == 1);
   return true;
}

static void
check_mirrored_public_status(struct pipe_context *pipe,
                             const struct mgc_runtime_status *draw_status)
{
   mgc_context_get_runtime_status(pipe, NULL);
   assert((draw_status->error != 0) == (draw_status->message[0] != '\0'));
   struct pipe_fence_handle *fence = NULL;
   pipe->flush(pipe, &fence, 0);
   assert(fence && pipe->screen->fence_finish(pipe->screen, pipe, fence, 0));
   uint64_t sequence = 0;
   int error = PIPE_OK;
   assert(mgc_fence_get_status(pipe->screen, fence, &sequence, &error));
   assert(sequence > draw_status->sequence && error == draw_status->error);
   assert(mgc_fence_get_status(pipe->screen, fence, NULL, NULL));
   assert(!mgc_fence_get_status(pipe->screen, NULL, &sequence, &error));
   mgc_context_clear_error(pipe);
   struct mgc_runtime_status cleared;
   mgc_context_get_runtime_status(pipe, &cleared);
   assert(cleared.error == PIPE_OK && !cleared.message[0]);
   assert(cleared.accepted_draws == draw_status->accepted_draws);
   assert(cleared.rejected_draws == draw_status->rejected_draws);
   assert(cleared.sequence == sequence);
   assert(mgc_fence_get_status(pipe->screen, fence, &sequence, &error));
   assert(error == draw_status->error);
   pipe->screen->fence_reference(pipe->screen, &fence, NULL);
   assert(!fence);
}
'''


def mirrored_guard_source(version):
    texture_filter = "nearest" if version == 5 else "linear"
    other_filter = "linear" if version == 5 else "nearest"
    hooks = REPEAT_HOOKS
    for original, replacement in (
        (r'\"wrap\":\"repeat\"', r'\"wrap\":\"mirrored-repeat\"'),
        (r'\"wrap\":\"Repeat\"', r'\"wrap\":\"Mirrored-repeat\"'),
        ("PIPE_TEX_WRAP_MIRROR_REPEAT", "PIPE_TEX_WRAP_REPEAT"),
        ("wrap_s_mirror", "wrap_s_repeat"),
        ("wrap_t_mirror", "wrap_t_repeat"),
        ("wrap_both_mirror", "wrap_both_repeat"),
        ('version = "03";', 'version = "0@VERSION@";'),
        ('version = "3.0";', 'version = "@VERSION@.0";'),
        ('version = " 3";', 'version = " @VERSION@";'),
        ('else if (is_case(name, "compile_unknown")) version = "11";',
         'else if (is_case(name, "compile_unknown") || is_case(name, "compile_silent_unknown")) version = "11";\n'
         '   else if (is_case(name, "compile_empty")) version = "";\n'
         '   else if (is_case(name, "compile_plus")) version = "+@VERSION@";\n'
         '   else if (is_case(name, "compile_trailing_space")) version = "@VERSION@ ";'),
        ("bool accepted = !strncmp(name, \"compile_accept_\", 15);",
         "assert(MGC_HOST_TEXTURE_MIRRORED_REPEAT_VERSION == 5 &&\n"
         "          MGC_HOST_TEXTURE_LINEAR_MIRRORED_REPEAT_VERSION == 6);\n"
         "   bool accepted = !strncmp(name, \"compile_accept_\", 15);"),
        ("} else assert(!state && shader.error_message);",
         "} else {\n"
         "      assert(!state);\n"
         "      assert((shader.error_message != NULL) == shader.report_compile_error);\n"
         "   }"),
        (".report_compile_error = true,",
         '.report_compile_error = !is_case(name, "compile_silent_unknown"),'),
        ("   pipe->destroy(pipe);",
         "   check_mirrored_public_status(pipe, &status);\n   pipe->destroy(pipe);"),
    ):
        assert original in hooks, original
        hooks = hooks.replace(original, replacement)
    position = hooks.index("static int\ncompile_repeat_guard")
    hooks = hooks[:position] + EXTRA_HOOKS + "\n" + hooks[position:]
    for original, replacement in {
        "@VERSION@": str(version),
        "@OTHER_VERSION@": str(11 - version),
        "@CLAMP_VERSION@": str(version - 4),
        "@REPEAT_VERSION@": str(version - 2),
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
        (".wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,", ".wrap_s = PIPE_TEX_WRAP_MIRROR_REPEAT,"),
        (".wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,", ".wrap_t = PIPE_TEX_WRAP_MIRROR_REPEAT,"),
        ("state->wrap_s = PIPE_TEX_WRAP_REPEAT;", "state->wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE;"),
        ("state->wrap_t = PIPE_TEX_WRAP_REPEAT;", "state->wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;"),
        ("known |= change_discard_contract(name, ctx->shaders[1]->shader);",
         "known |= change_discard_contract(name, ctx->shaders[1]->shader);\n"
         "   known |= change_repeat_contract(name, ctx);\n"
         "   known |= change_mirrored_contract(name, ctx);"),
        ("known |= change_sampler(name, &sampler);",
         "known |= change_sampler(name, &sampler);\n"
         "      known |= change_repeat_sampler(name, &sampler);\n"
         "      known |= change_mirrored_sampler(name, &sampler);"),
        ("((struct pipe_sampler_state *)state)->min_img_filter = PIPE_TEX_FILTER_LINEAR;",
         f"((struct pipe_sampler_state *)state)->min_img_filter = PIPE_TEX_FILTER_{other_filter.upper()};"),
        ("known |= change_storage(name, resource);",
         "known |= change_storage(name, resource);\n"
         "      known |= change_repeat_storage(name, pipe, resource);\n"
         "      known |= exercise_mirrored_references(name, pipe, resource);"),
        ("assert(known);", 'assert(known || !strncmp(name, "draw_", 5));'),
        ("pipe->draw_vbo(pipe, &info, 7, NULL, &draw, 1);",
         "change_repeat_draw(name, &info, &draw);\n"
         "   pipe->draw_vbo(pipe, &info, 7, NULL, &draw, 1);"),
        ("struct mgc_resource *storage = mgc_resource(resource);\n      resource->width0",
         "struct mgc_resource *storage = mgc_resource(resource);\n"
         '      if (is_case(name, "host_execution_owned")) {\n'
         "         assert(storage->active_host_executions == 1);\n"
         "         simple_mtx_lock(&mgc_screen(screen)->lock);\n"
         "         storage->active_host_executions = 0;\n"
         "         simple_mtx_unlock(&mgc_screen(screen)->lock);\n"
         "      }\n"
         "      assert(!storage->active_maps && !storage->active_host_executions);\n"
         "      resource->width0"),
        ("assert(compiled->abi_version == 1);",
         "assert(compiled->abi_version == 1);\n"
         "   assert(mgc_context(pipe)->shaders[fragment]->reference.count == 1);\n"
         '   const char *version = getenv("MGC_HOST_TEXTURE");\n'
         "   assert(compiled->abi.host_texture_version == (version ? strtoul(version, NULL, 10) : 0));"),
        ("pipe_sampler_view_reference(&view, NULL);",
         "assert(view->reference.count == 2 && resource->reference.count == 2);\n"
         "      pipe_sampler_view_reference(&view, NULL);\n"
         "      assert(ctx->sampler_views[1][0]->reference.count == 1);"),
        ("assert(ctx->sampler_views[1][0] == saved_view);",
         "assert(ctx->sampler_views[1][0] == saved_view);\n"
         "   assert(saved_view->reference.count == 1 && resource->reference.count == 2);"),
        ("   pipe->destroy(pipe);",
         "   check_mirrored_public_status(pipe, &status);\n   pipe->destroy(pipe);"),
        ("int\nmain(int argc, char **argv)",
         hooks + "\nstatic int\nrepeat_capture_guard_main(int argc, char **argv)"),
    )
    contents = SOURCE.read_text()
    for original, replacement in replacements:
        assert contents.count(original) == 1, original
        contents = contents.replace(original, replacement, 1)
    assert "@" not in contents
    return contents, texture_filter


@pytest.fixture(scope="module", params=MIRRORED_VERSIONS,
                ids=("nearest-mirrored-v5", "linear-mirrored-v6"))
def mirrored_texture_guard(request, tmp_path_factory):
    version = request.param
    contents, texture_filter = mirrored_guard_source(version)
    source = tmp_path_factory.mktemp(f"mirrored-texture-capture-v{version}-source") / "guard.c"
    source.write_text(contents)
    return build_texture_guard(tmp_path_factory, source), version, texture_filter


@pytest.mark.parametrize("case", VALID_TEXTURES + EXTRA_VALID_CASES +
                         ("valid_rebind_ownership",) + TRANSACTIONAL_BINDINGS)
def test_direct_c_mirrored_capture_and_binding_ownership(mirrored_texture_guard, tmp_path, case):
    helper, version, texture_filter = mirrored_texture_guard
    status = run_guard(helper, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["callbacks"] == 0 and status["sequence"] == 1
    assert (status["operation_error"] != 0) == (case in TRANSACTIONAL_BINDINGS)
    assert_texture_capture(
        read_capture(tmp_path), case,
        host_texture=dict(HOST_TEXTURE, version=version, filter=texture_filter, wrap="mirrored-repeat"),
        sampler=dict(SAMPLER, min_filter=texture_filter, mag_filter=texture_filter,
                     wrap_s="mirrored-repeat", wrap_t="mirrored-repeat"),
    )


@pytest.mark.parametrize("case", MIRRORED_SAMPLER_REJECTIONS + RESOURCE_REJECTIONS +
                         BINDING_REJECTIONS + DISCARD_CONTRACT_REJECTIONS +
                         LAYOUT_REJECTIONS + EXTRA_LAYOUT_REJECTIONS + DRAW_REJECTIONS + (
                             "host_execution_owned", "custom_texture",
                             "custom_texture_option_zero", "callback_disabled",
                         ))
def test_direct_c_mirrored_rejects_before_capture_or_callback(mirrored_texture_guard, tmp_path, case):
    helper, _, _ = mirrored_texture_guard
    status = run_guard(helper, tmp_path, case)
    assert status["accepted"] == 0 and status["rejected"] == 1
    assert status["error"] != 0 and status["operation_error"] == 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ("abi_version", "layout_host_version"))
@pytest.mark.parametrize("contract_version", (1, 2, 3, 4, 5, 6))
def test_direct_c_mirrored_requires_exact_compiled_version(mirrored_texture_guard, tmp_path,
                                                         field, contract_version):
    helper, version, _ = mirrored_texture_guard
    status = run_guard(helper, tmp_path, f"{field}_{contract_version}")
    if contract_version == version:
        assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
        assert read_capture(tmp_path)["shaders"]["fs"]["layout"]["host_texture"]["version"] == version
    else:
        assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
        assert list(tmp_path.iterdir()) == []
    assert status["callbacks"] == 0 and status["operation_error"] == 0


@pytest.mark.parametrize("case", ("legacy_default", "legacy_opt_in", "legacy_custom",
                                 "legacy_custom_opt_in"))
def test_mirrored_option_preserves_untextured_v1_and_public_callback_abi(mirrored_texture_guard,
                                                                      tmp_path, case):
    helper, _, _ = mirrored_texture_guard
    status = run_guard(helper, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["sequence"] == 1 and status["operation_error"] == 0
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


def test_mirrored_capture_keeps_existing_payload_keys(texture_guard, mirrored_texture_guard, tmp_path):
    helper, _, _ = mirrored_texture_guard
    captures = []
    for binary, name in ((texture_guard, "clamp"), (helper, "mirrored")):
        directory = tmp_path / name
        directory.mkdir()
        assert run_guard(binary, directory, "valid")["accepted"] == 1
        captures.append(read_capture(directory))
    clamp, mirrored = captures
    assert clamp.keys() == mirrored.keys()
    for stage in ("vs", "fs"):
        assert clamp["shaders"][stage].keys() == mirrored["shaders"][stage].keys()
        assert clamp["shaders"][stage]["layout"].keys() == mirrored["shaders"][stage]["layout"].keys()
    assert mirrored["shaders"]["fs"]["layout"]["host_texture"].keys() == HOST_TEXTURE.keys()
    assert clamp["textures"][0].keys() == mirrored["textures"][0].keys()
    assert clamp["textures"][0]["sampler"].keys() == mirrored["textures"][0]["sampler"].keys()
    assert dict(mirrored["textures"][0], sampler=SAMPLER) == clamp["textures"][0]


@pytest.mark.parametrize("case", MIRRORED_COMPILE_REJECTIONS)
def test_direct_c_default_unknown_and_hardware_reject_mirrored(mirrored_texture_guard, tmp_path, case):
    helper, _, _ = mirrored_texture_guard
    status = run_guard(helper, tmp_path, case)
    assert status["compile_rejected"] == 1 and status["compile_accepted"] == 0
    assert status["error"] != 0 and status["accepted"] == 0 and status["rejected"] == 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", (1, 2, 3, 4, 5, 6))
def test_direct_c_legacy_option_accepts_exactly_supported_versions(mirrored_texture_guard, tmp_path,
                                                                 version):
    helper, _, _ = mirrored_texture_guard
    status = run_guard(helper, tmp_path, f"compile_accept_{version}")
    assert status["compile_accepted"] == 1 and status["compile_rejected"] == 0
    assert status["error"] == 0 and status["accepted"] == 0 and status["rejected"] == 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []
