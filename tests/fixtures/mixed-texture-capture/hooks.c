static void
replace_layout(struct mgc_shader *shader, const char *original, const char *replacement)
{
   const char *value = strstr(shader->layout_json, original);
   assert(value);
   shader->layout_json = ralloc_asprintf(shader, "%.*s%s%s",
      (int)(value - shader->layout_json), shader->layout_json,
      replacement, value + strlen(original));
   assert(shader->layout_json);
}

static bool
change_mixed_contract(const char *name, struct mgc_context *ctx)
{
   struct mgc_shader *shader = ctx->shaders[1]->shader;
   if (is_case(name, "abi_zero")) shader->abi.host_texture_version = 0;
   else if (is_case(name, "abi_unknown")) shader->abi.host_texture_version = 11;
   else if (is_case(name, "abi_vs_unknown")) ctx->shaders[0]->shader->abi.host_texture_version = 11;
   else if (is_case(name, "abi_hardware")) shader->abi.target = MGC_TARGET_FP16_HARDWARE_V1;
   else if (is_case(name, "abi_wrap_s_invalid")) shader->abi.host_texture_wrap_s = 3;
   else if (is_case(name, "abi_wrap_t_invalid")) shader->abi.host_texture_wrap_t = 3;
   else if (is_case(name, "abi_wrap_s_negative")) shader->abi.host_texture_wrap_s = -1;
   else if (is_case(name, "abi_wrap_t_negative")) shader->abi.host_texture_wrap_t = -1;
   else if (is_case(name, "abi_vs_wrap_invalid")) ctx->shaders[0]->shader->abi.host_texture_wrap_s = 3;
   else if (is_case(name, "abi_wrap_s_mismatch"))
      shader->abi.host_texture_wrap_s = (shader->abi.host_texture_wrap_s + 1) % 3;
   else if (is_case(name, "abi_wrap_t_mismatch"))
      shader->abi.host_texture_wrap_t = (shader->abi.host_texture_wrap_t + 1) % 3;
   else if (is_case(name, "uses_texture_false")) shader->uses_host_texture = false;
   else if (is_case(name, "abi_filter_mismatch")) {
      unsigned other = shader->abi.host_texture_version == 7 ? 8 : 7;
      char original[64], replacement[64];
      snprintf(original, sizeof(original), "\"host_texture\":{\"version\":%u", shader->abi.host_texture_version);
      snprintf(replacement, sizeof(replacement), "\"host_texture\":{\"version\":%u", other);
      replace_layout(shader, original, replacement);
      shader->abi.host_texture_version = other;
      replace_layout(shader, other == 8 ? "\"filter\":\"nearest\"" : "\"filter\":\"linear\"",
                     other == 8 ? "\"filter\":\"linear\"" : "\"filter\":\"nearest\"");
   } else if (!strncmp(name, "layout_", 7)) {
      const char *suffix = name + 7;
      if (!strncmp(suffix, "wrap_", 5)) {
         unsigned axis = suffix[5] == 't';
         const char *change = suffix + 7;
         char original[80], replacement[160];
         snprintf(original, sizeof(original), "\"wrap_%c\":\"%s\"", axis ? 't' : 's',
                  mgc_host_texture_wrap_name(mgc_host_texture_wrap_for_axis(&shader->abi, axis)));
         if (!strcmp(change, "missing")) {
            snprintf(original, sizeof(original), ",\"wrap_%c\":\"%s\"", axis ? 't' : 's',
                     mgc_host_texture_wrap_name(mgc_host_texture_wrap_for_axis(&shader->abi, axis)));
            replacement[0] = '\0';
         } else if (!strcmp(change, "duplicate"))
            snprintf(replacement, sizeof(replacement), "%s,%s", original, original);
         else if (!strcmp(change, "legacy"))
            snprintf(replacement, sizeof(replacement), "\"wrap\":\"%s\"",
                     mgc_host_texture_wrap_name(mgc_host_texture_wrap_for_axis(&shader->abi, axis)));
         else if (!strcmp(change, "mismatch"))
            snprintf(replacement, sizeof(replacement), "\"wrap_%c\":\"%s\"", axis ? 't' : 's',
                     mgc_host_texture_wrap_name((mgc_host_texture_wrap_for_axis(&shader->abi, axis) + 1) % 3));
         else
            snprintf(replacement, sizeof(replacement), "\"wrap_%c\":%s", axis ? 't' : 's', change);
         replace_layout(shader, original, replacement);
      } else {
         static const struct {
            const char *name;
            const char *original;
            const char *replacement;
         } changes[] = {
            {"version", "\"version\":2", "\"version\":1"},
            {"filter", "\"filter\":", "\"wrong_filter\":"},
            {"unknown", "\"texture\":0", "\"texture\":0,\"unknown\":0"},
            {"axis_unknown", "\"texture\":0", "\"texture\":0,\"wrap_r\":\"repeat\""},
            {"duplicate", "\"texture\":0", "\"texture\":0,\"texture\":0"},
            {"missing_texture", "\"texture\":0,", ""},
            {"missing_sampler", "\"sampler\":0,", ""},
            {"missing_block", "\"storage_block\":15,", ""},
            {"missing_width", "\"max_width\":16,", ""},
            {"missing_height", "\"max_height\":16,", ""},
            {"texture_slot", "\"texture\":0", "\"texture\":1"},
            {"sampler_slot", "\"sampler\":0", "\"sampler\":1"},
            {"block", "\"storage_block\":15", "\"storage_block\":14"},
            {"width", "\"max_width\":16", "\"max_width\":17"},
            {"height", "\"max_height\":16", "\"max_height\":17"},
            {"legacy_wrap", "\"texture\":0", "\"texture\":0,\"wrap\":\"repeat\""},
            {"host_null", "\"host_texture\":{", "\"host_texture\":null,\"ignored\":{"},
            {"stage", "\"stage\":\"fragment\"", "\"stage\":\"vertex\""},
            {"shader_abi", "\"shader_abi_version\":1", "\"shader_abi_version\":2"},
         };
         for (unsigned index = 0; index < ARRAY_SIZE(changes); index++) {
            if (!strcmp(suffix, changes[index].name)) {
               replace_layout(shader, changes[index].original, changes[index].replacement);
               return true;
            }
         }
         if (!strcmp(suffix, "host_version") || !strcmp(suffix, "host_version_missing")) {
            char original[64];
            snprintf(original, sizeof(original), "\"host_texture\":{\"version\":%u,",
                     shader->abi.host_texture_version);
            replace_layout(shader, original, !strcmp(suffix, "host_version") ?
                           "\"host_texture\":{\"version\":9," : "\"host_texture\":{");
         } else if (!strcmp(suffix, "trailing")) shader->layout_json = ralloc_asprintf(shader, "%sx", shader->layout_json);
         else if (!strcmp(suffix, "truncated")) shader->layout_json[strlen(shader->layout_json) - 2] = '\0';
         else return false;
      }
   } else return false;
   return true;
}

static bool
change_mixed_sampler(const char *name, struct pipe_sampler_state *sampler)
{
   unsigned wrong = sampler->min_img_filter == PIPE_TEX_FILTER_LINEAR ?
      PIPE_TEX_FILTER_NEAREST : PIPE_TEX_FILTER_LINEAR;
   if (is_case(name, "min_wrong")) sampler->min_img_filter = wrong;
   else if (is_case(name, "mag_wrong")) sampler->mag_img_filter = wrong;
   else if (is_case(name, "both_wrong")) sampler->min_img_filter = sampler->mag_img_filter = wrong;
   else if (is_case(name, "mip_unknown")) sampler->min_mip_filter = 3;
   else if (!strncmp(name, "sampler_wrap_", 13)) {
      unsigned wrap = strtoul(name + 15, NULL, 10);
      if (name[13] == 's') sampler->wrap_s = wrap;
      else if (name[13] == 't') sampler->wrap_t = wrap;
      else return false;
   } else return false;
   return true;
}

static bool
change_mixed_ownership(const char *name, struct mgc_context *ctx)
{
   struct pipe_sampler_view *view = ctx->sampler_views[1][0];
   if (is_case(name, "view_context")) view->context = NULL;
   else if (is_case(name, "view_texture_null")) view->texture = NULL;
   else if (is_case(name, "resource_screen")) view->texture->screen = NULL;
   else if (is_case(name, "resource_next")) view->texture->next = view->texture;
   else if (is_case(name, "storage_busy")) mgc_resource(view->texture)->active_host_executions = 1;
   else return false;
   return true;
}

static int
compile_rejection_guard(void)
{
   glsl_type_singleton_init_or_ref();
   struct pipe_screen *screen = mgc_create_screen(NULL);
   assert(screen);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   assert(pipe);
   nir_builder builder = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      mgc_test_get_compiler_options(), "mixed_texture_compile_guard");
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
      .type = PIPE_SHADER_IR_NIR, .ir.nir = builder.shader, .report_compile_error = true,
   };
   assert(!pipe->create_fs_state(pipe, &shader));
   assert(shader.error_message);
   free(shader.error_message);
   struct mgc_runtime_status status;
   mgc_context_get_runtime_status(pipe, &status);
   assert(status.error == PIPE_ERROR_BAD_INPUT);
   pipe->destroy(pipe);
   assert(mgc_screen(screen)->allocated_bytes == 0);
   screen->destroy(screen);
   glsl_type_singleton_decref();
   printf("{\"accepted\":0,\"rejected\":0,\"error\":%d,\"operation_error\":0,"
          "\"callbacks\":0,\"compile_rejected\":1}\n", status.error);
   return 0;
}
