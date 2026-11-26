static bool
configure_instancing(const char *name, struct pipe_draw_info *info,
                     struct pipe_draw_start_count_bias *draws, unsigned *num_draws)
{
   capture_rows = 4;
   capture_stride = 16;
   capture_divisor = 1;
   capture_elements = 1;
   if (is_case(name, "single") || is_case(name, "custom_single")) {
      info->instance_count = 1;
      capture_divisor = 0;
   } else if (is_case(name, "single_divisor")) info->instance_count = 1;
   else if (is_case(name, "base")) { info->start_instance = 2; capture_rows = 6; }
   else if (is_case(name, "floor") || is_case(name, "short_floor")) {
      info->start_instance = 2; info->instance_count = 5; capture_divisor = 3;
      capture_rows = is_case(name, "floor") ? 4 : 3;
   } else if (is_case(name, "short_base")) {
      info->start_instance = 3; capture_divisor = 2;
   } else if (is_case(name, "short_instance")) capture_rows = 3;
   else if (is_case(name, "short_vertex")) { capture_divisor = 0; capture_rows = 2; }
   else if (is_case(name, "per_vertex") || is_case(name, "mixed_divisors")) {
      capture_divisor = 0; capture_rows = 3;
      if (is_case(name, "mixed_divisors")) capture_elements = 2;
   } else if (is_case(name, "max_instances")) { info->instance_count = 16; capture_rows = 16; }
   else if (is_case(name, "large_instances")) info->instance_count = 17;
   else if (is_case(name, "zero_instances")) info->instance_count = 0;
   else if (is_case(name, "zero_vertices")) draws[0].count = 0;
   else if (is_case(name, "max_budget") || is_case(name, "large_budget")) {
      info->instance_count = 16; capture_divisor = 0;
      capture_rows = draws[0].count = is_case(name, "max_budget") ? 256 : 257;
   } else if (is_case(name, "max_base")) {
      info->start_instance = 1048576; capture_stride = 0; capture_rows = 1;
   } else if (is_case(name, "large_base")) { info->start_instance = 1048577; capture_stride = 0; }
   else if (is_case(name, "base_overflow")) { info->start_instance = UINT_MAX; capture_stride = 0; }
   else if (is_case(name, "max_divisor")) capture_divisor = 1048576;
   else if (is_case(name, "large_divisor")) capture_divisor = 1048577;
   else if (is_case(name, "divisor_unknown")) capture_divisor = UINT_MAX;
   else if (is_case(name, "offset")) capture_buffer_offset = 16;
   else if (is_case(name, "restart")) info->primitive_restart = true;
   else if (is_case(name, "lines")) info->mode = MESA_PRIM_LINES;
   else if (is_case(name, "batch_invalid")) { *num_draws = 2; draws[1].count = 5; capture_divisor = 0; }
   else if (!strncmp(name, "indexed", 7)) {
      if (strstr(name, "per_vertex") || strstr(name, "bias")) capture_divisor = 0;
      if (strstr(name, "base")) info->start_instance = 2;
      if (strstr(name, "positive_bias")) draws[0].index_bias = 1;
      if (strstr(name, "negative_bias")) draws[0].index_bias = -1;
      if (strstr(name, "start")) draws[0].start = 1;
   } else if (!strncmp(name, "default_", 8) || strstr(name, "option_one") ||
              is_case(name, "shader_vs_id_flag")) {
      info->instance_count = 1;
      capture_divisor = 0;
   } else if (!strncmp(name, "metadata_", 9) || !strncmp(name, "shader_", 7) ||
              !strncmp(name, "header_", 7) || !strncmp(name, "default_", 8) ||
              is_case(name, "valid") || is_case(name, "custom") || is_case(name, "callback_disabled") ||
              is_case(name, "malformed_env") || is_case(name, "dual_slot") ||
              is_case(name, "vertex_screen") || is_case(name, "vertex_target") ||
              is_case(name, "vertex_storage") || is_case(name, "vertex_null") ||
              is_case(name, "vertex_mapped") || is_case(name, "vertex_offset") ||
              is_case(name, "hardware_vertex") || is_case(name, "hardware_fragment") ||
              is_case(name, "unknown_version") || is_case(name, "env_after_creation") ||
              is_case(name, "unsupported_reflection")) {
   } else return false;
   return true;
}

static struct pipe_resource *
bind_indices(const char *name, struct pipe_context *pipe, struct pipe_draw_info *info)
{
   if (strncmp(name, "indexed", 7)) return NULL;
   unsigned size = strstr(name, "_1") ? 1 : strstr(name, "_4") ? 4 : 2;
   unsigned count = strstr(name, "start") ? 4 : strstr(name, "short") ? 2 : 3;
   struct pipe_resource *resource = pipe_buffer_create(pipe->screen,
      PIPE_BIND_INDEX_BUFFER, PIPE_USAGE_DEFAULT, count * size);
   assert(resource);
   for (unsigned index = 0; index < count; index++) {
      uint32_t value = strstr(name, "start") ? (index ? index - 1 : 100) : index;
      if (strstr(name, "divisor")) value += 100;
      memcpy((uint8_t *)mgc_resource(resource)->data + index * size, &value, size);
   }
   info->index_size = size;
   info->index.resource = resource;
   if (strstr(name, "ownership")) {
      struct pipe_resource *owned = NULL;
      pipe_resource_reference(&owned, resource);
      info->index.resource = owned;
      info->take_index_buffer_ownership = true;
   }
   return resource;
}

static void
mutate_instancing(const char *name, struct mgc_context *ctx, struct pipe_draw_info *info)
{
   struct pipe_resource *resource = ctx->vertex_buffers[0].buffer.resource;
   if (is_case(name, "dual_slot")) ctx->vertex_elements.elements[0].dual_slot = true;
   else if (is_case(name, "vertex_screen")) resource->screen = NULL;
   else if (is_case(name, "vertex_target")) resource->target = PIPE_TEXTURE_2D;
   else if (is_case(name, "vertex_storage")) mgc_resource(resource)->size = 16;
   else if (is_case(name, "vertex_null")) mgc_resource(resource)->data = NULL;
   else if (is_case(name, "vertex_mapped")) mgc_resource(resource)->active_maps = 1;
   else if (is_case(name, "vertex_offset")) ctx->vertex_buffers[0].buffer_offset = resource->width0 + 1;
   else if (is_case(name, "hardware_vertex")) ctx->shaders[0]->shader->abi.target = MGC_TARGET_FP16_HARDWARE_V1;
   else if (is_case(name, "hardware_fragment")) ctx->shaders[1]->shader->abi.target = MGC_TARGET_FP16_HARDWARE_V1;
   else if (is_case(name, "unknown_version")) ctx->host_instancing_version = 2;
   else if (is_case(name, "env_after_creation")) assert(!setenv("MGC_HOST_INSTANCING", "0", 1));
   else if (is_case(name, "unsupported_reflection")) {
      struct mgc_shader *shader = ctx->shaders[0]->shader;
      const char *value = strstr(shader->layout_json, "\"supported\":true");
      assert(value);
      shader->layout_json = ralloc_asprintf(shader, "%.*s\"supported\":false%s",
         (int)(value - shader->layout_json), shader->layout_json, value + strlen("\"supported\":true"));
   }
}

static void
replace_instancing_text(struct mgc_shader *shader, bool assembly,
                        const char *original, const char *replacement)
{
   char **text = assembly ? &shader->asm_text : &shader->layout_json;
   const char *value = strstr(*text, original);
   assert(value);
   *text = ralloc_asprintf(shader, "%.*s%s%s", (int)(value - *text), *text,
                          replacement, value + strlen(original));
   assert(*text);
}

static void
remove_instancing_field(struct mgc_shader *shader, const char *field)
{
   const char *value = strstr(shader->layout_json, field);
   assert(value);
   char original[256];
   if (value[strlen(field)] == ',') snprintf(original, sizeof(original), "%s,", field);
   else {
      assert(value > shader->layout_json && value[-1] == ',');
      snprintf(original, sizeof(original), ",%s", field);
   }
   replace_instancing_text(shader, false, original, "");
}

static void
change_instancing_shader_contract(const char *name, struct mgc_context *ctx)
{
   unsigned stage = strstr(name, "_fs_") ? 1 : 0;
   struct mgc_shader *shader = ctx->shaders[stage]->shader;
   if (!strncmp(name, "shader_", 7)) {
      const char *change = name + 10;
      if (!strcmp(change, "option_zero")) shader->abi.host_instancing_version = 0;
      else if (!strcmp(change, "option_one")) shader->abi.host_instancing_version = 1;
      else if (!strcmp(change, "option_two")) shader->abi.host_instancing_version = 2;
      else if (!strcmp(change, "abi1")) shader->abi_version = MGC_ABI_VERSION;
      else if (!strcmp(change, "abi2")) shader->abi_version = MGC_HARDWARE_ABI_VERSION;
      else if (!strcmp(change, "id_flag")) shader->uses_host_instance_id = !shader->uses_host_instance_id;
      else assert(false);
   } else if (!strncmp(name, "header_", 7)) {
      replace_instancing_text(shader, true, "ABI=3,", "ABI=1,");
   } else if (!strncmp(name, "default_", 8)) {
      if (is_case(name, "default_metadata_object"))
         replace_instancing_text(shader, false, "\"inputs\":[",
            "\"host_instancing\":{\"version\":1,\"max_instances\":16,\"instance_id_word\":65535,"
            "\"instance_id\":\"local-zero-based\",\"uses_instance_id\":false},\"inputs\":[");
      else if (is_case(name, "default_layout3"))
         replace_instancing_text(shader, false, "\"version\":1", "\"version\":3");
      else if (is_case(name, "default_header3"))
         replace_instancing_text(shader, true, "ABI=1,", "ABI=3,");
      else assert(false);
   } else if (!strncmp(name, "metadata_", 9)) {
      const char *change = name + 12;
      if (!strncmp(change, "object_", 7)) {
         const char *start = strstr(shader->layout_json, "\"host_instancing\":{");
         assert(start);
         const char *end = strchr(start, '}');
         assert(end);
         char object[512], replacement[1024];
         snprintf(object, sizeof(object), "%.*s", (int)(end + 1 - start), start);
         if (!strcmp(change, "object_missing")) remove_instancing_field(shader, object);
         else if (!strcmp(change, "object_null")) replace_instancing_text(shader, false, object, "\"host_instancing\":null");
         else if (!strcmp(change, "object_duplicate")) {
            snprintf(replacement, sizeof(replacement), "%s,%s", object, object);
            replace_instancing_text(shader, false, object, replacement);
         } else assert(false);
      } else if (!strncmp(change, "layout", 6)) {
         char replacement[32];
         snprintf(replacement, sizeof(replacement), "\"version\":%s", change + 6);
         replace_instancing_text(shader, false, "\"version\":3", replacement);
      } else {
         const char *fields[] = {
            "version", "max_instances", "instance_id_word", "instance_id", "uses_instance_id",
         };
         const char *values[] = {
            "1", "16", "65535", "\"local-zero-based\"", shader->uses_host_instance_id ? "true" : "false",
         };
         const char *mode = NULL;
         const char *field = change;
         if (!strncmp(field, "missing_", 8)) { mode = "missing"; field += 8; }
         else if (!strncmp(field, "duplicate_", 10)) { mode = "duplicate"; field += 10; }
         for (unsigned index = 0; index < ARRAY_SIZE(fields); index++) {
            size_t length = strlen(fields[index]);
            if (strncmp(field, fields[index], length) || (field[length] && field[length] != '_'))
               continue;
            char original[128], replacement[256];
            snprintf(original, sizeof(original), "\"%s\":%s", fields[index], values[index]);
            if (mode && !strcmp(mode, "missing")) remove_instancing_field(shader, original);
            else if (mode) {
               snprintf(replacement, sizeof(replacement), "%s,%s", original, original);
               replace_instancing_text(shader, false, original, replacement);
            } else {
               snprintf(replacement, sizeof(replacement), "\"%s\":%s", fields[index], field + length + 1);
               replace_instancing_text(shader, false, original, replacement);
            }
            return;
         }
         if (!strcmp(change, "unknown"))
            replace_instancing_text(shader, false, "\"max_instances\":16", "\"max_instances\":16,\"unknown\":0");
         else assert(false);
      }
   }
}
