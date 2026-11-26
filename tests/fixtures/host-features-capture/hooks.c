static bool
case_is(const char *name, const char *expected)
{
   return !strcmp(name, expected);
}

static nir_variable *
features_io(nir_shader *shader, nir_variable_mode mode, const struct glsl_type *type,
              const char *name, unsigned location, unsigned driver_location)
{
   nir_variable *variable = nir_variable_create(shader, mode, type, name);
   variable->data.location = location;
   variable->data.driver_location = driver_location;
   variable->data.interpolation = INTERP_MODE_SMOOTH;
   return variable;
}

static void
features_shader(struct pipe_context *pipe, bool fragment, const char *name, bool textured)
{
   nir_builder builder = nir_builder_init_simple_shader(fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX,
      mgc_test_get_compiler_options(), "host_features_capture_guard");
   nir_variable *output = features_io(builder.shader, nir_var_shader_out, glsl_vec4_type(),
      "output", fragment ? FRAG_RESULT_COLOR : VARYING_SLOT_POS, 0);
   nir_def *result;
   if (!fragment) {
      nir_variable *input = features_io(builder.shader, nir_var_shader_in, glsl_vec4_type(),
         "position", VERT_ATTRIB_GENERIC0, 0);
      result = nir_load_var(&builder, input);
      if (case_is(name, "builtins")) {
         nir_def *identifiers = nir_iadd(&builder, nir_load_instance_id(&builder), nir_load_vertex_id(&builder));
         identifiers = nir_iadd(&builder, identifiers, nir_load_base_instance(&builder));
         identifiers = nir_iadd(&builder, identifiers, nir_load_draw_id(&builder));
         result = nir_fadd(&builder, result, nir_fmul_imm(&builder, nir_i2f32(&builder, identifiers), 0.001));
      }
      if (textured) {
         nir_variable *varying = features_io(builder.shader, nir_var_shader_out, glsl_vec2_type(),
            "uv", VARYING_SLOT_VAR0, 1);
         nir_store_var(&builder, varying, nir_imm_vec2(&builder, 0.25, 0.75), 3);
      }
   } else if (textured) {
      nir_variable *input = features_io(builder.shader, nir_var_shader_in, glsl_vec2_type(),
         "uv", VARYING_SLOT_VAR0, 0);
      nir_def *coordinates = nir_load_var(&builder, input);
      unsigned samples = case_is(name, "four_bindings") || case_is(name, "aliases") ? 4 :
         case_is(name, "sparse") || case_is(name, "same_texture") ? 2 : 1;
      result = nir_imm_vec4(&builder, 0, 0, 0, 0);
      for (unsigned sample = 0; sample < samples; sample++) {
         bool implicit = case_is(name, "implicit");
         bool gradients = case_is(name, "gradients");
         nir_tex_instr *texture = nir_tex_instr_create(builder.shader, implicit ? 1 : gradients ? 3 : 2);
         texture->op = implicit ? nir_texop_tex : gradients ? nir_texop_txd : nir_texop_txl;
         texture->sampler_dim = GLSL_SAMPLER_DIM_2D;
         texture->dest_type = nir_type_float32;
         texture->coord_components = 2;
         texture->texture_index = case_is(name, "sparse") ? (sample ? 1 : 3) : sample;
         texture->sampler_index = case_is(name, "sparse") ? (sample ? 3 : 1) : sample;
         if (case_is(name, "same_texture")) texture->texture_index = 0;
         texture->src[0].src_type = nir_tex_src_coord;
         texture->src[0].src = nir_src_for_ssa(coordinates);
         if (!implicit) {
            texture->src[1].src_type = gradients ? nir_tex_src_ddx : nir_tex_src_lod;
            texture->src[1].src = nir_src_for_ssa(gradients ? nir_imm_vec2(&builder, 0.125, 0) : nir_imm_float(&builder, 1.25));
         }
         if (gradients) {
            texture->src[2].src_type = nir_tex_src_ddy;
            texture->src[2].src = nir_src_for_ssa(nir_imm_vec2(&builder, 0, 0.125));
         }
         nir_def_init(&texture->instr, &texture->def, 4, 32);
         nir_builder_instr_insert(&builder, &texture->instr);
         result = nir_fadd(&builder, result, &texture->def);
      }
   } else {
      result = nir_imm_vec4(&builder, 1, 0.5, 0.25, 1);
      if (case_is(name, "builtins")) {
         nir_def *position = nir_load_frag_coord(&builder);
         result = nir_bcsel(&builder, nir_load_front_face(&builder, 1), position, result);
      } else if (!strncmp(name, "coord_mask_", 11)) {
         unsigned mask = strtoul(name + 11, NULL, 10);
         assert(mask && mask < 16);
         nir_def *position = nir_load_frag_coord(&builder);
         nir_def *components[4];
         for (unsigned component = 0; component < 4; component++)
            components[component] = mask & (1u << component) ?
               nir_channel(&builder, position, component) : nir_imm_float(&builder, 1);
         result = nir_vec(&builder, components, 4);
      } else if (case_is(name, "derivatives")) {
         nir_variable *input = features_io(builder.shader, nir_var_shader_in, glsl_float_type(),
            "value", VARYING_SLOT_VAR0, 0);
         nir_def *operand = nir_load_var(&builder, input);
         result = nir_fadd(&builder, result, nir_ddx_fine(&builder, operand));
      }
   }
   nir_store_var(&builder, output, result, 15);
   struct pipe_shader_state state = {.type = PIPE_SHADER_IR_NIR, .ir.nir = builder.shader, .report_compile_error = true};
   void *compiled = fragment ? pipe->create_fs_state(pipe, &state) : pipe->create_vs_state(pipe, &state);
   if (!compiled)
      fprintf(stderr, "shader compile: %s\n", state.error_message ? state.error_message : "unknown");
   free(state.error_message);
   assert(compiled);
   if (fragment) {
      pipe->bind_fs_state(pipe, compiled);
      pipe->delete_fs_state(pipe, compiled);
   } else {
      pipe->bind_vs_state(pipe, compiled);
      pipe->delete_vs_state(pipe, compiled);
   }
}

static struct pipe_resource *
features_resource(struct pipe_context *pipe, enum pipe_format format,
                    unsigned width, unsigned height, unsigned bind, unsigned last_level)
{
   struct pipe_resource template = {.target = PIPE_TEXTURE_2D, .format = format,
      .width0 = width, .height0 = height, .depth0 = 1, .array_size = 1,
      .bind = bind, .last_level = last_level, .usage = PIPE_USAGE_DEFAULT};
   struct pipe_resource *resource = pipe->screen->resource_create(pipe->screen, &template);
   assert(resource);
   struct mgc_resource *storage = mgc_resource(resource);
   for (size_t index = 0; index < storage->size; index++)
      ((uint8_t *)storage->data)[index] = (index * 17 + 3) & 255;
   return resource;
}

static void
features_pipeline(struct pipe_context *pipe, struct pipe_resource *color,
                    struct pipe_resource *depth, const char *name)
{
   struct pipe_framebuffer_state framebuffer = {.width = color->width0, .height = color->height0, .nr_cbufs = 1};
   framebuffer.cbufs[0] = (struct pipe_surface){.texture = color, .format = color->format};
   if (depth)
      framebuffer.zsbuf = (struct pipe_surface){.texture = depth, .format = depth->format};
   pipe->set_framebuffer_state(pipe, &framebuffer);
   struct pipe_blend_state blend = {0};
   blend.rt[0] = (struct pipe_rt_blend_state){.colormask = 15, .rgb_src_factor = PIPE_BLENDFACTOR_ONE,
      .alpha_src_factor = PIPE_BLENDFACTOR_ONE, .rgb_dst_factor = PIPE_BLENDFACTOR_ZERO,
      .alpha_dst_factor = PIPE_BLENDFACTOR_ZERO};
   if (!strncmp(name, "factor_", 7)) {
      const unsigned factors[] = {PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_SRC_COLOR,
         PIPE_BLENDFACTOR_INV_SRC_COLOR, PIPE_BLENDFACTOR_DST_COLOR, PIPE_BLENDFACTOR_INV_DST_COLOR,
         PIPE_BLENDFACTOR_SRC_ALPHA, PIPE_BLENDFACTOR_INV_SRC_ALPHA, PIPE_BLENDFACTOR_DST_ALPHA,
         PIPE_BLENDFACTOR_INV_DST_ALPHA, PIPE_BLENDFACTOR_CONST_COLOR, PIPE_BLENDFACTOR_INV_CONST_COLOR,
         PIPE_BLENDFACTOR_CONST_ALPHA, PIPE_BLENDFACTOR_INV_CONST_ALPHA, PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE};
      unsigned factor = strtoul(name + 7, NULL, 10);
      assert(factor < ARRAY_SIZE(factors));
      blend.rt[0].blend_enable = 1;
      blend.rt[0].rgb_src_factor = blend.rt[0].alpha_src_factor = factors[factor];
      blend.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_CONST_COLOR;
      blend.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_INV_DST_ALPHA;
   }
   if (!strncmp(name, "equation_", 9)) {
      unsigned equation = strtoul(name + 9, NULL, 10);
      blend.rt[0].blend_enable = 1;
      blend.rt[0].rgb_func = equation;
      blend.rt[0].alpha_func = (equation + 1) % 5;
   }
   if (case_is(name, "mask")) blend.rt[0].colormask = PIPE_MASK_R | PIPE_MASK_B;
   if (case_is(name, "bad_factor")) { blend.rt[0].blend_enable = 1; blend.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_SRC1_COLOR; }
   if (case_is(name, "bad_equation")) { blend.rt[0].blend_enable = 1; blend.rt[0].rgb_func = 7; }
   if (case_is(name, "dither")) blend.dither = 1;
   void *state = pipe->create_blend_state(pipe, &blend);
   assert(state); pipe->bind_blend_state(pipe, state); pipe->delete_blend_state(pipe, state);
   struct pipe_blend_color constant = {.color = {0.25, 0.5, 0.75, 1}};
   if (case_is(name, "nan_constant")) constant.color[0] = NAN;
   pipe->set_blend_color(pipe, &constant);
   struct pipe_rasterizer_state rasterizer = {.front_ccw = false, .depth_clip_near = true,
      .depth_clip_far = true, .half_pixel_center = true};
   if (case_is(name, "scissor")) rasterizer.scissor = true;
   if (case_is(name, "front_cw")) rasterizer.front_ccw = true;
   state = pipe->create_rasterizer_state(pipe, &rasterizer);
   assert(state); pipe->bind_rasterizer_state(pipe, state); pipe->delete_rasterizer_state(pipe, state);
   struct pipe_depth_stencil_alpha_state depth_state = {.depth_enabled = depth != NULL,
      .depth_writemask = true, .depth_func = PIPE_FUNC_LESS};
   if (!strncmp(name, "depth_", 6)) depth_state.depth_func = strtoul(name + 6, NULL, 10);
   if (case_is(name, "depth_no_write")) depth_state.depth_writemask = false;
   if (case_is(name, "stencil")) {
      depth_state.stencil[0] = (struct pipe_stencil_state){.enabled = true, .func = PIPE_FUNC_GEQUAL,
         .valuemask = 0xa5, .writemask = 0x5a, .fail_op = PIPE_STENCIL_OP_INCR_WRAP,
         .zfail_op = PIPE_STENCIL_OP_DECR, .zpass_op = PIPE_STENCIL_OP_REPLACE};
      depth_state.stencil[1] = (struct pipe_stencil_state){.enabled = true, .func = PIPE_FUNC_NOTEQUAL,
         .valuemask = 0x3c, .writemask = 0xc3, .fail_op = PIPE_STENCIL_OP_ZERO,
         .zfail_op = PIPE_STENCIL_OP_INVERT, .zpass_op = PIPE_STENCIL_OP_DECR_WRAP};
   }
   state = pipe->create_depth_stencil_alpha_state(pipe, &depth_state);
   assert(state); pipe->bind_depth_stencil_alpha_state(pipe, state); pipe->delete_depth_stencil_alpha_state(pipe, state);
   pipe->set_stencil_ref(pipe, (struct pipe_stencil_ref){.ref_value = {17, 231}});
   struct pipe_viewport_state viewport = {.scale = {color->width0 * 0.5f, color->height0 * 0.5f, 0.5f},
      .translate = {color->width0 * 0.5f, color->height0 * 0.5f, 0.5f},
      .swizzle_x = PIPE_VIEWPORT_SWIZZLE_POSITIVE_X, .swizzle_y = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Y,
      .swizzle_z = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Z, .swizzle_w = PIPE_VIEWPORT_SWIZZLE_POSITIVE_W};
   if (case_is(name, "bad_viewport")) viewport.translate[0]++;
   pipe->set_viewport_states(pipe, 0, 1, &viewport);
}

static void
features_replace(struct mgc_shader *shader, const char *original, const char *replacement)
{
   char *found = strstr(shader->layout_json, original);
   assert(found);
   char *updated = ralloc_asprintf(shader, "%.*s%s%s", (int)(found - shader->layout_json),
      shader->layout_json, replacement, found + strlen(original));
   assert(updated);
   shader->layout_json = updated;
}
