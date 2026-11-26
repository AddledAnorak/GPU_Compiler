#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "util/ralloc.h"
#include "util/u_inlines.h"
#include "mgc_compiler.h"
#include "mgc_context.h"
#include "mgc_public.h"
#include "mgc_resource.h"
#include "mgc_screen.h"

static struct mgc_compile_options capture_options;
static unsigned capture_rows, capture_stride, capture_divisor, capture_elements;
static unsigned capture_buffer_offset;
static bool capture_instance_id;

static unsigned
sampler_wrap(unsigned axis)
{
   const unsigned wraps[] = {PIPE_TEX_WRAP_CLAMP_TO_EDGE, PIPE_TEX_WRAP_REPEAT, PIPE_TEX_WRAP_MIRROR_REPEAT};
   return wraps[mgc_host_texture_wrap_for_axis(&capture_options, axis)];
}

static bool
is_case(const char *name, const char *expected)
{
   return !strcmp(name, expected);
}

static nir_variable *
io_variable(nir_shader *shader, nir_variable_mode mode,
            const struct glsl_type *type, const char *name,
            unsigned location, unsigned driver_location)
{
   nir_variable *variable = nir_variable_create(shader, mode, type, name);
   variable->data.location = location;
   variable->data.driver_location = driver_location;
   variable->data.interpolation = INTERP_MODE_SMOOTH;
   return variable;
}

static bool
bind_shader(struct pipe_context *pipe, bool fragment, bool textured, bool discard)
{
   nir_builder builder = nir_builder_init_simple_shader(
      fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX,
      mgc_test_get_compiler_options(), "texture_capture_guard");
   nir_variable *output = io_variable(builder.shader, nir_var_shader_out,
      glsl_vec4_type(), fragment ? "color" : "position",
      fragment ? FRAG_RESULT_COLOR : VARYING_SLOT_POS, 0);
   if (fragment) {
      nir_def *color;
      if (textured) {
         nir_variable *input = io_variable(builder.shader, nir_var_shader_in,
            glsl_vec2_type(), "uv", VARYING_SLOT_VAR0, 0);
         nir_def *coordinates = nir_load_var(&builder, input);
         nir_tex_instr *texture = nir_tex_instr_create(builder.shader, 1);
         texture->op = nir_texop_tex;
         texture->sampler_dim = GLSL_SAMPLER_DIM_2D;
         texture->dest_type = nir_type_float32;
         texture->coord_components = 2;
         texture->src[0].src_type = nir_tex_src_coord;
         texture->src[0].src = nir_src_for_ssa(coordinates);
         nir_def_init(&texture->instr, &texture->def, 4, 32);
         nir_builder_instr_insert(&builder, &texture->instr);
         color = &texture->def;
      } else {
         color = nir_imm_vec4(&builder, 1, 0.5, 0.25, 1);
      }
      if (discard) {
         nir_terminate_if(&builder, nir_flt_imm(&builder, nir_channel(&builder, color, 0), 0.5));
         builder.shader->info.fs.uses_discard = true;
      }
      nir_store_var(&builder, output, color, 15);
   } else {
      nir_variable *input = io_variable(builder.shader, nir_var_shader_in,
         glsl_vec4_type(), "attribute", VERT_ATTRIB_GENERIC0, 0);
      nir_def *position = nir_load_var(&builder, input);
      if (capture_instance_id) {
         nir_def *local_id = nir_i2f32(&builder, nir_load_instance_id(&builder));
         nir_def *zero = nir_imm_float(&builder, 0);
         position = nir_fadd(&builder, position, nir_vec4(&builder, local_id, zero, zero, zero));
      }
      nir_store_var(&builder, output, position, 15);
      if (textured) {
         nir_variable *varying = io_variable(builder.shader, nir_var_shader_out,
            glsl_vec2_type(), "uv", VARYING_SLOT_VAR0, 1);
         nir_store_var(&builder, varying, nir_imm_vec2(&builder, 0.25, 0.75), 3);
      }
   }
   struct pipe_shader_state shader = {
      .type = PIPE_SHADER_IR_NIR, .ir.nir = builder.shader,
      .report_compile_error = true,
   };
   void *state = fragment ? pipe->create_fs_state(pipe, &shader) :
      pipe->create_vs_state(pipe, &shader);
   if (!state)
      fprintf(stderr, "shader compilation: %s\n",
              shader.error_message ? shader.error_message : "unknown error");
   free(shader.error_message);
   if (!state)
      return false;
   if (fragment) {
      pipe->bind_fs_state(pipe, state);
      pipe->delete_fs_state(pipe, state);
   } else {
      pipe->bind_vs_state(pipe, state);
      pipe->delete_vs_state(pipe, state);
   }
   const struct mgc_shader *compiled = mgc_context(pipe)->shaders[fragment]->shader;
   assert(compiled->uses_host_texture == (fragment && textured));
   unsigned instancing = mgc_context(pipe)->host_instancing_version == MGC_HOST_INSTANCING_VERSION ?
      MGC_HOST_INSTANCING_VERSION : 0;
   assert(compiled->abi.host_instancing_version == instancing);
   assert(compiled->abi_version == (instancing ? MGC_HOST_INSTANCING_ABI_VERSION :
          compiled->abi.target == MGC_TARGET_LEGACY ? MGC_ABI_VERSION : MGC_HARDWARE_ABI_VERSION));
   assert(compiled->uses_host_instance_id == (!fragment && capture_instance_id));
   return true;
}

static void
bind_vertices(struct pipe_context *pipe)
{
   struct pipe_vertex_buffer buffers[2] = {0};
   struct pipe_vertex_element elements[2] = {0};
   for (unsigned slot = 0; slot < capture_elements; slot++) {
      struct pipe_resource *resource = pipe_buffer_create(pipe->screen,
         PIPE_BIND_VERTEX_BUFFER, PIPE_USAGE_DEFAULT, capture_rows * 16 + capture_buffer_offset);
      assert(resource);
      for (unsigned row = 0; row < capture_rows; row++) {
         float vertex[] = {(float)row, (float)slot, 0, 1};
         memcpy((uint8_t *)mgc_resource(resource)->data + capture_buffer_offset + row * 16,
                vertex, sizeof(vertex));
      }
      buffers[slot].buffer.resource = resource;
      buffers[slot].buffer_offset = capture_buffer_offset;
      elements[slot] = (struct pipe_vertex_element){
         .src_format = PIPE_FORMAT_R32G32B32A32_FLOAT, .src_stride = capture_stride,
         .vertex_buffer_index = slot, .instance_divisor = slot ? 2 : capture_divisor,
      };
   }
   pipe->set_vertex_buffers(pipe, capture_elements, buffers);
   for (unsigned slot = 0; slot < capture_elements; slot++)
      assert(buffers[slot].buffer.resource->reference.count == 1);
   void *state = pipe->create_vertex_elements_state(pipe, capture_elements, elements);
   assert(state);
   pipe->bind_vertex_elements_state(pipe, state);
   pipe->delete_vertex_elements_state(pipe, state);
}

static bool
custom_callback(void *data, const struct mgc_draw_artifact *artifact)
{
   unsigned *calls = data;
   assert(artifact->runtime_abi_version == MGC_RUNTIME_ABI_VERSION);
   assert(artifact->runtime_abi_version == 1);
   assert(artifact->shader_abi_version == 1);
   assert(artifact->layout[0] && artifact->layout[1]);
   (*calls)++;
   return true;
}

static struct pipe_sampler_state
sampler_template(void)
{
   return (struct pipe_sampler_state){
      .wrap_s = sampler_wrap(0),
      .wrap_t = sampler_wrap(1),
      .wrap_r = PIPE_TEX_WRAP_REPEAT,
      .min_img_filter = mgc_host_texture_is_linear(capture_options.host_texture_version) ?
         PIPE_TEX_FILTER_LINEAR : PIPE_TEX_FILTER_NEAREST,
      .mag_img_filter = mgc_host_texture_is_linear(capture_options.host_texture_version) ?
         PIPE_TEX_FILTER_LINEAR : PIPE_TEX_FILTER_NEAREST,
      .min_mip_filter = PIPE_TEX_MIPFILTER_NONE,
      .compare_mode = PIPE_TEX_COMPARE_NONE,
      .max_lod = 1000,
   };
}

static bool
change_sampler(const char *name, struct pipe_sampler_state *state)
{
   if (is_case(name, "min_linear")) state->min_img_filter = PIPE_TEX_FILTER_LINEAR;
   else if (is_case(name, "mag_linear")) state->mag_img_filter = PIPE_TEX_FILTER_LINEAR;
   else if (is_case(name, "mip_nearest")) state->min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
   else if (is_case(name, "mip_linear")) state->min_mip_filter = PIPE_TEX_MIPFILTER_LINEAR;
   else if (is_case(name, "wrap_s")) state->wrap_s = PIPE_TEX_WRAP_REPEAT;
   else if (is_case(name, "wrap_t")) state->wrap_t = PIPE_TEX_WRAP_REPEAT;
   else if (is_case(name, "unnormalized")) state->unnormalized_coords = 1;
   else if (is_case(name, "compare")) state->compare_mode = PIPE_TEX_COMPARE_R_TO_TEXTURE;
   else if (is_case(name, "anisotropy")) state->max_anisotropy = 2;
   else if (is_case(name, "lod_bias")) state->lod_bias = 0.5;
   else if (is_case(name, "lod_bias_nan")) state->lod_bias = NAN;
   else if (is_case(name, "min_lod")) state->min_lod = 0.5;
   else if (is_case(name, "min_lod_nan")) state->min_lod = NAN;
   else if (is_case(name, "max_lod_negative")) state->max_lod = -1;
   else if (is_case(name, "max_lod_nan")) state->max_lod = NAN;
   else if (is_case(name, "max_lod_infinite")) state->max_lod = INFINITY;
   else if (is_case(name, "reduction")) state->reduction_mode = PIPE_TEX_REDUCTION_MIN;
   else if (is_case(name, "seamless_cube")) state->seamless_cube_map = 1;
   else if (is_case(name, "integer_border")) state->border_color_is_integer = 1;
   else if (is_case(name, "valid_anisotropy_one")) state->max_anisotropy = 1;
   else if (is_case(name, "valid_max_lod_zero")) state->max_lod = 0;
   else return false;
   return true;
}

static struct pipe_sampler_view *
create_view(struct pipe_context *pipe, struct pipe_resource *resource)
{
   struct pipe_sampler_view view = {
      .format = resource->format, .target = resource->target,
      .swizzle_r = PIPE_SWIZZLE_X, .swizzle_g = PIPE_SWIZZLE_Y,
      .swizzle_b = PIPE_SWIZZLE_Z, .swizzle_a = PIPE_SWIZZLE_W,
   };
   struct pipe_sampler_view *result = pipe->create_sampler_view(pipe, resource, &view);
   assert(result);
   return result;
}

static int
exercise_binding(const char *name, struct pipe_context *pipe,
                 struct pipe_resource *resource)
{
   struct mgc_context *ctx = mgc_context(pipe);
   struct pipe_sampler_state saved = ctx->sampler_states[1][0];
   struct pipe_sampler_view *saved_view = ctx->sampler_views[1][0];
   void *samplers[] = {&saved, &saved};
   struct pipe_sampler_view *views[] = {saved_view, saved_view};
   if (is_case(name, "bind_state_stage"))
      pipe->bind_sampler_states(pipe, PIPE_SHADER_COMPUTE, 0, 1, samplers);
   else if (is_case(name, "bind_state_start"))
      pipe->bind_sampler_states(pipe, PIPE_SHADER_FRAGMENT, UINT_MAX, 1, samplers);
   else if (is_case(name, "bind_state_count"))
      pipe->bind_sampler_states(pipe, PIPE_SHADER_FRAGMENT, 7, 2, samplers);
   else if (is_case(name, "create_state_null"))
      assert(!pipe->create_sampler_state(pipe, NULL));
   else if (is_case(name, "bind_view_stage"))
      pipe->set_sampler_views(pipe, PIPE_SHADER_COMPUTE, 0, 1, 0, views);
   else if (is_case(name, "bind_view_start"))
      pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, UINT_MAX, 1, 0, views);
   else if (is_case(name, "bind_view_count"))
      pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 7, 2, 0, views);
   else if (is_case(name, "bind_view_trailing"))
      pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, UINT_MAX, views);
   else if (is_case(name, "bind_view_transaction")) {
      struct pipe_context *other = pipe->screen->context_create(pipe->screen, NULL, 0);
      assert(other);
      struct pipe_sampler_view *foreign = create_view(other, resource);
      struct pipe_sampler_view *batch[] = {NULL, foreign};
      pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 2, 0, batch);
      pipe_sampler_view_reference(&foreign, NULL);
      other->destroy(other);
   } else {
      return 0;
   }
   assert(ctx->status.error == PIPE_ERROR_BAD_INPUT);
   assert(ctx->sampler_state_valid[1][0]);
   assert(!memcmp(&saved, &ctx->sampler_states[1][0], sizeof(saved)));
   assert(ctx->sampler_views[1][0] == saved_view);
   for (unsigned stage = 0; stage < MGC_RUNTIME_STAGES; stage++) {
      for (unsigned slot = 0; slot < 8; slot++) {
         if (stage != 1 || slot != 0) {
            assert(!ctx->sampler_views[stage][slot]);
            assert(!ctx->sampler_state_valid[stage][slot]);
         }
      }
   }
   int error = ctx->status.error;
   mgc_context_clear_error(pipe);
   return error;
}

static bool
change_view(const char *name, struct pipe_sampler_view *view)
{
   if (is_case(name, "view_format")) view->format = PIPE_FORMAT_B8G8R8A8_UNORM;
   else if (is_case(name, "view_level")) view->u.tex.first_level = view->u.tex.last_level = 1;
   else if (is_case(name, "view_last_level")) view->u.tex.last_level = 1;
   else if (is_case(name, "view_layer")) view->u.tex.first_layer = view->u.tex.last_layer = 1;
   else if (is_case(name, "view_last_layer")) view->u.tex.last_layer = 1;
   else if (is_case(name, "view_target")) view->target = PIPE_TEXTURE_RECT;
   else if (is_case(name, "view_from_buffer")) view->is_tex2d_from_buf = true;
   else if (is_case(name, "view_swizzle")) view->swizzle_r = PIPE_SWIZZLE_Z;
   else return false;
   return true;
}

static bool
change_discard_contract(const char *name, struct mgc_shader *shader)
{
   const char *original = NULL;
   const char *replacement = NULL;
   if (is_case(name, "discard_unversioned")) {
      original = "\"uses_discard\":false";
      replacement = "\"uses_discard\":true";
   } else if (is_case(name, "discard_status_option_zero")) {
      shader->abi.fragment_status_version = 0;
      return true;
   } else if (is_case(name, "discard_status_option_two")) {
      shader->abi.fragment_status_version = 2;
      return true;
   } else if (is_case(name, "discard_status_missing")) {
      original = ",\"fragment_status\":{\"version\":1,\"word_offset\":65535,\"pending\":0,\"discarded\":1,\"committed\":2}";
      replacement = "";
   } else if (is_case(name, "discard_status_version")) {
      original = "\"fragment_status\":{\"version\":1";
      replacement = "\"fragment_status\":{\"version\":2";
   } else if (is_case(name, "discard_status_offset")) {
      original = "\"word_offset\":65535";
      replacement = "\"word_offset\":65534";
   } else if (is_case(name, "discard_status_pending")) {
      original = "\"pending\":0";
      replacement = "\"pending\":1";
   } else if (is_case(name, "discard_status_discarded")) {
      original = "\"discarded\":1";
      replacement = "\"discarded\":2";
   } else if (is_case(name, "discard_status_committed")) {
      original = "\"committed\":2";
      replacement = "\"committed\":1";
   } else if (is_case(name, "discard_status_duplicate")) {
      original = "\"pending\":0";
      replacement = "\"pending\":0,\"pending\":0";
   } else if (is_case(name, "discard_status_unknown")) {
      original = "\"pending\":0";
      replacement = "\"pending\":0,\"extra\":0";
   } else if (is_case(name, "discard_status_missing_field")) {
      original = ",\"committed\":2";
      replacement = "";
   } else if (is_case(name, "discard_status_noninteger")) {
      original = "\"pending\":0";
      replacement = "\"pending\":0.0";
   } else {
      return false;
   }
   const char *value = strstr(shader->layout_json, original);
   assert(value);
   shader->layout_json = ralloc_asprintf(shader, "%.*s%s%s",
      (int)(value - shader->layout_json), shader->layout_json,
      replacement, value + strlen(original));
   assert(shader->layout_json);
   return true;
}

static bool
change_storage(const char *name, struct pipe_resource *resource)
{
   struct mgc_resource *storage = mgc_resource(resource);
   struct mgc_resource_level *level = &storage->levels[0];
   if (is_case(name, "storage_stride")) level->stride = resource->width0 * 4 - 1;
   else if (is_case(name, "storage_layer_stride")) level->layer_stride = 0;
   else if (is_case(name, "storage_offset")) level->offset = storage->size + 1;
   else if (is_case(name, "storage_size")) storage->size = resource->width0 * 4 - 1;
   else if (is_case(name, "storage_large_stride")) level->stride = UINT_MAX;
   else if (is_case(name, "storage_width")) level->width++;
   else if (is_case(name, "storage_height")) level->height++;
   else if (is_case(name, "storage_layers")) level->layers = 2;
   else if (is_case(name, "storage_null")) storage->data = NULL;
   else if (is_case(name, "width_zero")) resource->width0 = 0;
   else if (is_case(name, "height_zero")) resource->height0 = 0;
   else if (is_case(name, "samples")) resource->nr_samples = 2;
   else if (is_case(name, "storage_samples")) resource->nr_storage_samples = 2;
   else if (is_case(name, "depth")) resource->depth0 = 2;
   else if (is_case(name, "valid_padded")) {
      resource->width0 = level->width = 3;
      resource->height0 = level->height = 2;
      level->offset = 4;
   } else return false;
   return true;
}

#include "hooks.c"

int
main(int argc, char **argv)
{
   assert(argc == 2);
   const char *name = argv[1];
   capture_instance_id = !strncmp(name, "instance_id", 11);
   if (capture_instance_id)
      name = name[11] == '_' ? name + 12 : "valid";
   capture_options = mgc_default_compile_options();
   const char *version = getenv("MGC_HOST_TEXTURE");
   capture_options.host_texture_version = version ? strtoul(version, NULL, 10) : 0;
   const char *axes[] = {getenv("MGC_HOST_TEXTURE_WRAP_S"), getenv("MGC_HOST_TEXTURE_WRAP_T")};
   for (unsigned axis = 0; axis < 2; axis++) {
      for (unsigned wrap = 0; axes[axis] && wrap < 3; wrap++) {
         if (!strcmp(axes[axis], mgc_host_texture_wrap_name(wrap))) {
            if (axis) capture_options.host_texture_wrap_t = wrap;
            else capture_options.host_texture_wrap_s = wrap;
         }
      }
   }
   bool legacy = !capture_options.host_texture_version;
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 4};
   struct pipe_draw_start_count_bias draws[2] = {{0, 3, 0}, {0, 3, 0}};
   unsigned num_draws = 1;
   assert(configure_instancing(name, &info, draws, &num_draws));
   bool fragment_status = is_case(name, "valid_status") || is_case(name, "valid_discard") ||
      !strncmp(name, "discard_status_", 15);
   if (fragment_status) assert(!setenv("MGC_FRAGMENT_STATUS", "1", 1));
   else assert(!unsetenv("MGC_FRAGMENT_STATUS"));
   glsl_type_singleton_init_or_ref();
   struct pipe_screen *screen = mgc_create_screen(NULL);
   assert(screen);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   assert(pipe);
   struct mgc_context *ctx = mgc_context(pipe);
   int init_error = ctx->status.error;
   mgc_context_clear_error(pipe);
   if (!bind_shader(pipe, false, !legacy, false) ||
       !bind_shader(pipe, true, !legacy, fragment_status && !is_case(name, "valid_status"))) {
      int error = ctx->status.error;
      assert(error == PIPE_ERROR_BAD_INPUT);
      pipe->destroy(pipe);
      assert(mgc_screen(screen)->allocated_bytes == 0);
      screen->destroy(screen);
      glsl_type_singleton_decref();
      printf("{\"accepted\":0,\"rejected\":0,\"sequence\":0,\"error\":%d,"
             "\"operation_error\":0,\"callbacks\":0,\"init_error\":%d,\"compile_rejected\":1}\n",
             error, init_error);
      return 0;
   }
   bind_vertices(pipe);
   unsigned callbacks = 0;
   if (strstr(name, "custom"))
      mgc_context_set_artifact_callback(pipe, custom_callback, &callbacks);
   if (is_case(name, "callback_disabled"))
      mgc_context_set_artifact_callback(pipe, NULL, NULL);
   if (is_case(name, "custom_texture_option_zero"))
      ctx->shaders[1]->shader->abi.host_texture_version = 0;
   struct pipe_resource *resource = NULL;
   struct pipe_transfer *transfer = NULL;
   struct pipe_resource saved_resource = {0};
   struct mgc_resource_level saved_level = {0};
   size_t saved_size = 0;
   void *saved_data = NULL;
   int operation_error = 0;
   bool known = true;
   change_instancing_shader_contract(name, ctx);
   known |= change_discard_contract(name, ctx->shaders[1]->shader);
   if (!legacy) {
      struct pipe_resource template = {
         .target = PIPE_TEXTURE_2D, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 16, .height0 = 16, .depth0 = 1, .array_size = 1,
         .bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET,
         .usage = PIPE_USAGE_DEFAULT,
      };
      if (is_case(name, "valid_small")) template.width0 = template.height0 = 1;
      if (is_case(name, "valid_npot")) { template.width0 = 3; template.height0 = 5; }
      if (is_case(name, "width_large")) { template.width0 = 17; known = true; }
      if (is_case(name, "height_large")) { template.height0 = 17; known = true; }
      if (is_case(name, "resource_format")) { template.format = PIPE_FORMAT_B8G8R8A8_UNORM; known = true; }
      if (is_case(name, "mip_chain") || is_case(name, "view_level") || is_case(name, "view_last_level")) {
         template.last_level = 1;
         known = true;
      }
      if (is_case(name, "array")) {
         template.target = PIPE_TEXTURE_2D_ARRAY;
         template.array_size = 2;
         known = true;
      }
      if (is_case(name, "rectangle")) { template.target = PIPE_TEXTURE_RECT; known = true; }
      resource = screen->resource_create(screen, &template);
      assert(resource);
      struct mgc_resource *storage = mgc_resource(resource);
      for (size_t index = 0; index < storage->size; index++)
         ((uint8_t *)storage->data)[index] = (index * 17 + 3) & 255;
      saved_resource = *resource;
      saved_level = storage->levels[0];
      saved_size = storage->size;
      saved_data = storage->data;
      struct pipe_sampler_view *view = create_view(pipe, resource);
      known |= change_view(name, view);
      pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 0, &view);
      pipe_sampler_view_reference(&view, NULL);
      struct pipe_sampler_state sampler = sampler_template();
      known |= change_sampler(name, &sampler);
      void *state = pipe->create_sampler_state(pipe, &sampler);
      assert(state);
      void *states[] = {state};
      pipe->bind_sampler_states(pipe, PIPE_SHADER_FRAGMENT, 0, 1, states);
      ((struct pipe_sampler_state *)state)->min_img_filter = sampler.min_img_filter ==
         PIPE_TEX_FILTER_LINEAR ? PIPE_TEX_FILTER_NEAREST : PIPE_TEX_FILTER_LINEAR;
      pipe->delete_sampler_state(pipe, state);
      assert(!memcmp(&sampler, &ctx->sampler_states[1][0], sizeof(sampler)));
      operation_error = exercise_binding(name, pipe, resource);
      known |= operation_error != 0;
      known |= change_storage(name, resource);
      if (is_case(name, "mapped")) {
         struct pipe_box box = {.width = 16, .height = 16, .depth = 1};
         assert(pipe->texture_map(pipe, resource, 0, PIPE_MAP_READ, &box, &transfer));
         assert(transfer && storage->active_maps == 1);
         known = true;
      }
      if (is_case(name, "feedback_color") || is_case(name, "feedback_depth")) {
         struct pipe_framebuffer_state framebuffer = {.width = 16, .height = 16};
         struct pipe_surface surface = {.texture = resource, .format = resource->format};
         if (is_case(name, "feedback_color")) {
            framebuffer.nr_cbufs = 1;
            framebuffer.cbufs[0] = surface;
         } else {
            framebuffer.zsbuf = surface;
         }
         pipe->set_framebuffer_state(pipe, &framebuffer);
         assert(!ctx->status.error);
         known = true;
      }
      if (is_case(name, "missing_view") || is_case(name, "view_null_entry")) {
         struct pipe_sampler_view *empty[] = {NULL};
         pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 0,
                                 is_case(name, "missing_view") ? NULL : empty);
         assert(!ctx->sampler_views[1][0]);
         known = true;
      }
      if (is_case(name, "missing_state") || is_case(name, "state_null_entry")) {
         void *empty[] = {NULL};
         pipe->bind_sampler_states(pipe, PIPE_SHADER_FRAGMENT, 0, 1,
                                    is_case(name, "missing_state") ? NULL : empty);
         assert(!ctx->sampler_state_valid[1][0]);
         struct pipe_sampler_state zero = {0};
         assert(!memcmp(&zero, &ctx->sampler_states[1][0], sizeof(zero)));
         known = true;
      }
      if (is_case(name, "extra_fs_view") || is_case(name, "extra_vs_view") ||
          is_case(name, "valid_trailing_unbind")) {
         enum pipe_shader_type stage = is_case(name, "extra_vs_view") ?
            PIPE_SHADER_VERTEX : PIPE_SHADER_FRAGMENT;
         unsigned slot = stage == PIPE_SHADER_VERTEX ? 0 : 1;
         struct pipe_sampler_view *bound = ctx->sampler_views[1][0];
         pipe->set_sampler_views(pipe, stage, slot, 1, 0, &bound);
         if (is_case(name, "valid_trailing_unbind")) {
            pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 7, &bound);
            for (unsigned index = 1; index < 8; index++) assert(!ctx->sampler_views[1][index]);
         }
         known = true;
      }
      if (is_case(name, "extra_fs_state") || is_case(name, "extra_vs_state")) {
         enum pipe_shader_type stage = is_case(name, "extra_vs_state") ?
            PIPE_SHADER_VERTEX : PIPE_SHADER_FRAGMENT;
         void *additional[] = {&sampler};
         pipe->bind_sampler_states(pipe, stage, stage == PIPE_SHADER_VERTEX ? 0 : 1, 1, additional);
         known = true;
      }
      if (is_case(name, "valid_empty_binds")) {
         pipe->bind_sampler_states(pipe, PIPE_SHADER_FRAGMENT, 8, 0, NULL);
         pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 8, 0, 0, NULL);
         assert(ctx->sampler_state_valid[1][0] && ctx->sampler_views[1][0]);
      }
   }
   assert(known);
   assert(!ctx->status.error);
   struct pipe_resource *index_resource = bind_indices(name, pipe, &info);
   struct pipe_resource *vertex_resource = ctx->vertex_buffers[0].buffer.resource;
   struct pipe_resource saved_vertex = *vertex_resource;
   struct mgc_resource *vertex_storage = mgc_resource(vertex_resource);
   void *saved_vertex_data = vertex_storage->data;
   size_t saved_vertex_size = vertex_storage->size;
   mutate_instancing(name, ctx, &info);
   pipe->draw_vbo(pipe, &info, 7, NULL, draws, num_draws);
   struct mgc_runtime_status status;
   mgc_context_get_runtime_status(pipe, &status);
   vertex_resource->screen = saved_vertex.screen;
   vertex_resource->target = saved_vertex.target;
   vertex_resource->width0 = saved_vertex.width0;
   vertex_storage->data = saved_vertex_data;
   vertex_storage->size = saved_vertex_size;
   vertex_storage->active_maps = 0;
   if (index_resource) {
      assert(index_resource->reference.count == 1);
      pipe_resource_reference(&index_resource, NULL);
   }
   if (resource) {
      if (is_case(name, "valid_snapshot")) memset(saved_data, 0xee, saved_size);
      if (transfer) pipe->texture_unmap(pipe, transfer);
      struct mgc_resource *storage = mgc_resource(resource);
      resource->width0 = saved_resource.width0;
      resource->height0 = saved_resource.height0;
      resource->depth0 = saved_resource.depth0;
      resource->nr_samples = saved_resource.nr_samples;
      resource->nr_storage_samples = saved_resource.nr_storage_samples;
      storage->levels[0] = saved_level;
      storage->size = saved_size;
      storage->data = saved_data;
      pipe_resource_reference(&resource, NULL);
   }
   pipe->destroy(pipe);
   assert(mgc_screen(screen)->allocated_bytes == 0);
   screen->destroy(screen);
   glsl_type_singleton_decref();
   printf("{\"accepted\":%" PRIu64 ",\"rejected\":%" PRIu64
          ",\"sequence\":%" PRIu64 ",\"error\":%d,\"operation_error\":%d,\"callbacks\":%u,\"init_error\":%d}\n",
          status.accepted_draws, status.rejected_draws, status.sequence,
          status.error, operation_error, callbacks, init_error);
   return 0;
}
