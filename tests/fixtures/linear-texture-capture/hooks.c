static bool
change_linear_sampler(const char *name, struct pipe_sampler_state *sampler)
{
   if (is_case(name, "min_nearest")) sampler->min_img_filter = PIPE_TEX_FILTER_NEAREST;
   else if (is_case(name, "mag_nearest")) sampler->mag_img_filter = PIPE_TEX_FILTER_NEAREST;
   else if (is_case(name, "both_nearest")) {
      sampler->min_img_filter = PIPE_TEX_FILTER_NEAREST;
      sampler->mag_img_filter = PIPE_TEX_FILTER_NEAREST;
   } else if (is_case(name, "min_unknown")) sampler->min_img_filter = 2;
   else if (is_case(name, "mag_unknown")) sampler->mag_img_filter = 2;
   else return false;
   return true;
}

static bool
change_linear_contract(const char *name, struct mgc_context *ctx)
{
   struct mgc_shader *shader = ctx->shaders[1]->shader;
   if (is_case(name, "abi_zero")) shader->abi.host_texture_version = 0;
   else if (is_case(name, "abi_nearest")) shader->abi.host_texture_version = MGC_HOST_TEXTURE_VERSION;
   else if (is_case(name, "abi_unknown") || is_case(name, "legacy_unknown_opt_in"))
      shader->abi.host_texture_version = 11;
   else if (is_case(name, "abi_vs_unknown")) ctx->shaders[0]->shader->abi.host_texture_version = 11;
   else if (is_case(name, "uses_texture_false")) shader->uses_host_texture = false;
   else if (is_case(name, "nearest_abi_linear")) shader->abi.host_texture_version = MGC_HOST_TEXTURE_LINEAR_VERSION;
   else if (is_case(name, "nearest_hardware") || is_case(name, "linear_hardware"))
      shader->abi.target = MGC_TARGET_FP16_HARDWARE_V1;
   else if (is_case(name, "nearest_linear_sampler")) return true;
   else {
      static const struct {
         const char *name;
         const char *original;
         const char *replacement;
      } mutations[] = {
         {"layout_version_one", "{\"version\":2,", "{\"version\":1,"},
         {"layout_version_three", "{\"version\":2,", "{\"version\":3,"},
         {"layout_host_version_one", "\"host_texture\":{\"version\":2", "\"host_texture\":{\"version\":1"},
         {"layout_host_version_three", "\"host_texture\":{\"version\":2", "\"host_texture\":{\"version\":3"},
         {"layout_host_version_float", "\"host_texture\":{\"version\":2", "\"host_texture\":{\"version\":2.0"},
         {"layout_filter_nearest", "\"filter\":\"linear\"", "\"filter\":\"nearest\""},
         {"layout_filter_case", "\"filter\":\"linear\"", "\"filter\":\"Linear\""},
         {"layout_filter_unknown", "\"filter\":\"linear\"", "\"filter\":\"cubic\""},
         {"layout_wrap_repeat", "\"wrap\":\"clamp-to-edge\"", "\"wrap\":\"repeat\""},
         {"layout_texture_slot", "\"texture\":0", "\"texture\":1"},
         {"layout_sampler_slot", "\"sampler\":0", "\"sampler\":1"},
         {"layout_block", "\"storage_block\":15", "\"storage_block\":14"},
         {"layout_max_width", "\"max_width\":16", "\"max_width\":17"},
         {"layout_max_height", "\"max_height\":16", "\"max_height\":17"},
         {"layout_duplicate", "\"filter\":\"linear\"", "\"filter\":\"linear\",\"filter\":\"linear\""},
         {"layout_unknown", "\"filter\":\"linear\"", "\"filter\":\"linear\",\"extra\":0"},
         {"layout_missing_field", ",\"filter\":\"linear\"", ""},
         {"layout_missing_texture", ",\"host_texture\":{\"version\":2,\"texture\":0,\"sampler\":0,\"storage_block\":15,\"max_width\":16,\"max_height\":16,\"filter\":\"linear\",\"wrap\":\"clamp-to-edge\"}", ""},
         {"layout_stage", "\"stage\":\"fragment\"", "\"stage\":\"vertex\""},
         {"layout_shader_abi", "\"shader_abi_version\":1", "\"shader_abi_version\":2"},
      };
      for (unsigned index = 0; index < ARRAY_SIZE(mutations); index++) {
         if (!is_case(name, mutations[index].name))
            continue;
         const char *value = strstr(shader->layout_json, mutations[index].original);
         assert(value);
         shader->layout_json = ralloc_asprintf(shader, "%.*s%s%s",
            (int)(value - shader->layout_json), shader->layout_json,
            mutations[index].replacement, value + strlen(mutations[index].original));
         assert(shader->layout_json);
         return true;
      }
      return false;
   }
   return true;
}

static int
compile_rejection_guard(const char *name)
{
   assert(MGC_RUNTIME_ABI_VERSION == 1);
   assert(MGC_HOST_TEXTURE_LINEAR_VERSION == 2);
   if (is_case(name, "compile_default")) assert(!unsetenv("MGC_HOST_TEXTURE"));
   else if (is_case(name, "compile_zero")) assert(!setenv("MGC_HOST_TEXTURE", "0", 1));
   else if (is_case(name, "compile_unknown")) assert(!setenv("MGC_HOST_TEXTURE", "11", 1));
   else {
      assert(is_case(name, "compile_hardware_nearest") || is_case(name, "compile_hardware_linear"));
      assert(!setenv("MGC_COMPILE_TARGET", "fp16-hardware-v1", 1));
      assert(!setenv("MGC_HOST_TEXTURE", is_case(name, "compile_hardware_nearest") ? "1" : "2", 1));
   }
   glsl_type_singleton_init_or_ref();
   struct pipe_screen *screen = mgc_create_screen(NULL);
   assert(screen);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   assert(pipe);
   nir_builder builder = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      mgc_test_get_compiler_options(), "linear_capture_compile_rejection");
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
   printf("{\"accepted\":0,\"rejected\":0,\"error\":%d,\"callbacks\":0,\"compile_rejected\":1}\n",
          status.error);
   return 0;
}

static int capture_guard_main(int argc, char **argv);

int
main(int argc, char **argv)
{
   assert(argc == 2);
   if (!strncmp(argv[1], "compile_", 8))
      return compile_rejection_guard(argv[1]);
   return capture_guard_main(argc, argv);
}
