#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "util/ralloc.h"
#include "util/u_inlines.h"
#include "mgc_compiler.h"
#include "mgc_context.h"
#include "mgc_public.h"
#include "mgc_resource.h"
#include "mgc_screen.h"

#include "hooks.c"

static bool
features_callback(void *data, const struct mgc_draw_artifact *artifact)
{
   assert(artifact->runtime_abi_version == 1 && artifact->shader_abi_version == 1);
   (*(unsigned *)data)++;
   return true;
}

struct features_probe {
   struct mgc_context *ctx;
   struct pipe_resource *color;
   const char *name;
   unsigned rejected_maps;
};

static void *
features_probe_run(void *data)
{
   struct features_probe *probe = data;
   bool reserved = false;
   for (unsigned attempt = 0; attempt < 7000; attempt++) {
      simple_mtx_lock(&probe->ctx->screen->lock);
      reserved = mgc_resource(probe->color)->active_host_executions != 0;
      simple_mtx_unlock(&probe->ctx->screen->lock);
      if (reserved) break;
      usleep(1000);
   }
   assert(reserved);
   usleep(25000);
   if (case_is(probe->name, "native_state")) {
      simple_mtx_lock(&probe->ctx->screen->lock);
      probe->ctx->blend_color.color[0] = 0.875;
      simple_mtx_unlock(&probe->ctx->screen->lock);
   } else if (case_is(probe->name, "native_data")) {
      simple_mtx_lock(&probe->ctx->screen->lock);
      ((uint8_t *)mgc_resource(probe->color)->data)[0] ^= 1;
      simple_mtx_unlock(&probe->ctx->screen->lock);
   } else {
      struct pipe_context *other = probe->ctx->base.screen->context_create(probe->ctx->base.screen, NULL, 0);
      assert(other);
      const unsigned modes[] = {PIPE_MAP_READ, PIPE_MAP_WRITE, PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED};
      struct pipe_box box = {.width = 1, .height = 1, .depth = 1};
      for (unsigned index = 0; index < ARRAY_SIZE(modes); index++) {
         struct pipe_transfer *transfer = NULL;
         assert(!other->texture_map(other, probe->color, 0, modes[index], &box, &transfer) && !transfer);
         probe->rejected_maps++;
         mgc_context_clear_error(other);
      }
      other->destroy(other);
   }
   return NULL;
}

int
main(int argc, char **argv)
{
   assert(argc == 7);
   const char *name = argv[1];
   unsigned width = strtoul(argv[2], NULL, 10), height = strtoul(argv[3], NULL, 10);
   unsigned source_format = strtoul(argv[4], NULL, 10), color_format = strtoul(argv[5], NULL, 10);
   unsigned depth_format = strtoul(argv[6], NULL, 10);
   const enum pipe_format source_formats[] = {PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_B8G8R8A8_UNORM,
      PIPE_FORMAT_R8G8B8_UNORM, PIPE_FORMAT_R8_UNORM};
   const enum pipe_format color_formats[] = {PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_B8G8R8A8_UNORM,
      PIPE_FORMAT_B5G6R5_UNORM, PIPE_FORMAT_R8_UNORM};
   const enum pipe_format depth_formats[] = {PIPE_FORMAT_NONE, PIPE_FORMAT_Z16_UNORM, PIPE_FORMAT_Z24_UNORM_S8_UINT};
   assert(source_format < 4 && color_format < 4 && depth_format < 3);
   glsl_type_singleton_init_or_ref();
   struct pipe_screen *screen = mgc_create_screen(NULL);
   assert(screen);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   assert(pipe);
   struct mgc_context *ctx = mgc_context(pipe);
   int initial_error = ctx->status.error;
   if (initial_error) {
      mgc_context_clear_error(pipe);
      struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
      struct pipe_draw_start_count_bias draw = {.count = 3};
      pipe->draw_vbo(pipe, &info, 7, NULL, &draw, 1);
      printf("{\"accepted\":0,\"rejected\":%" PRIu64 ",\"sequence\":%" PRIu64 ",\"error\":%d,\"initial_error\":%d,\"callbacks\":0}\n",
         ctx->status.rejected_draws, ctx->status.sequence, ctx->status.error, initial_error);
      pipe->destroy(pipe); screen->destroy(screen); glsl_type_singleton_decref(); return 0;
   }
   bool textured = getenv("MGC_HOST_TEXTURE") && strcmp(getenv("MGC_HOST_TEXTURE"), "0");
   features_shader(pipe, false, name, textured);
   features_shader(pipe, true, name, textured);
   unsigned callbacks = 0;
   if (case_is(name, "custom")) mgc_context_set_artifact_callback(pipe, features_callback, &callbacks);
   const float vertices[] = {-1, -1, 0, 1, 1, -1, 0, 1, 0, 1, 0, 1};
   struct pipe_resource *vertex = pipe_buffer_create(screen, PIPE_BIND_VERTEX_BUFFER, PIPE_USAGE_DEFAULT, sizeof(vertices));
   assert(vertex); memcpy(mgc_resource(vertex)->data, vertices, sizeof(vertices));
   struct pipe_vertex_buffer buffer = {0};
   pipe_resource_reference(&buffer.buffer.resource, vertex);
   pipe->set_vertex_buffers(pipe, 1, &buffer);
   struct pipe_vertex_element element = {.src_format = PIPE_FORMAT_R32G32B32A32_FLOAT, .src_stride = 16};
   if (case_is(name, "max_vertices") || case_is(name, "over_vertices") || case_is(name, "indexed")) element.src_stride = 0;
   if (case_is(name, "divisor")) { element.instance_divisor = 2; element.src_stride = 0; }
   void *elements = pipe->create_vertex_elements_state(pipe, 1, &element);
   assert(elements); pipe->bind_vertex_elements_state(pipe, elements); pipe->delete_vertex_elements_state(pipe, elements);
   struct pipe_resource *color = features_resource(pipe, color_formats[color_format], 3, 2, PIPE_BIND_RENDER_TARGET, 0);
   struct pipe_resource *depth = depth_format ? features_resource(pipe, depth_formats[depth_format], 3, 2, PIPE_BIND_DEPTH_STENCIL, 0) : NULL;
   features_pipeline(pipe, color, depth, name);
   struct pipe_resource *textures[4] = {0};
   struct mgc_resource saved[4];
   for (unsigned slot = 0; textured && slot < 4; slot++) {
      if (case_is(name, "aliases") && slot) pipe_resource_reference(&textures[slot], textures[0]);
      else textures[slot] = features_resource(pipe, source_formats[source_format], width, height,
         PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET, util_logbase2(MAX2(width, height)));
      saved[slot] = *mgc_resource(textures[slot]);
      struct pipe_sampler_view view = {.format = textures[slot]->format, .target = PIPE_TEXTURE_2D,
         .swizzle_r = PIPE_SWIZZLE_X, .swizzle_g = PIPE_SWIZZLE_Y, .swizzle_b = PIPE_SWIZZLE_Z,
         .swizzle_a = PIPE_SWIZZLE_W, .u.tex.last_level = textures[slot]->last_level};
      if (case_is(name, "canonical_swizzle")) {
         if (source_format == 3) view.swizzle_g = view.swizzle_b = PIPE_SWIZZLE_0;
         if (source_format >= 2) view.swizzle_a = PIPE_SWIZZLE_1;
      }
      struct pipe_sampler_view *sampler_view = pipe->create_sampler_view(pipe, textures[slot], &view);
      assert(sampler_view); pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, slot, 1, 0, &sampler_view);
      pipe_sampler_view_reference(&sampler_view, NULL);
      unsigned version = ctx->shaders[1]->shader->abi.host_texture_version;
      struct pipe_sampler_state sampler = {.min_img_filter = version == 11 ? PIPE_TEX_FILTER_NEAREST : PIPE_TEX_FILTER_LINEAR,
         .mag_img_filter = version == 11 ? PIPE_TEX_FILTER_NEAREST : PIPE_TEX_FILTER_LINEAR,
         .min_mip_filter = version >= 13 ? PIPE_TEX_MIPFILTER_LINEAR : PIPE_TEX_MIPFILTER_NEAREST,
         .max_anisotropy = version == 14 ? 4 : 0,
         .wrap_s = PIPE_TEX_WRAP_REPEAT, .wrap_t = PIPE_TEX_WRAP_MIRROR_REPEAT, .max_lod = 1000};
      void *state = pipe->create_sampler_state(pipe, &sampler);
      assert(state); pipe->bind_sampler_states(pipe, PIPE_SHADER_FRAGMENT, slot, 1, &state); pipe->delete_sampler_state(pipe, state);
   }
   struct mgc_shader *shader = ctx->shaders[1]->shader;
   if (getenv("MGC_TEST_CONTRACT_OLD"))
      features_replace(shader, getenv("MGC_TEST_CONTRACT_OLD"), getenv("MGC_TEST_CONTRACT_NEW"));
   if (!strncmp(name, "feature_", 8)) features_replace(shader, "\"max_instances\":256", "\"max_instances\":257");
   if (case_is(name, "layout_version")) features_replace(shader, "\"version\":4", "\"version\":3");
   if (case_is(name, "duplicate_feature")) features_replace(shader, "\"host_features\":", "\"host_features\":{},\"host_features\":");
   if (case_is(name, "texture_contract")) features_replace(shader, "\"max_levels\":6", "\"max_levels\":5");
   if (case_is(name, "binding_contract")) features_replace(shader, "\"texture\":0,\"sampler\":0", "\"texture\":0,\"sampler\":1");
   if (case_is(name, "texture_base")) shader->abi.texture_base++;
   if (case_is(name, "texture_samples")) shader->host_texture_samples = 33;
   if (case_is(name, "shader_abi")) shader->abi_version = 1;
   if (case_is(name, "missing_view")) pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 0, 1, NULL);
   if (case_is(name, "missing_sampler")) pipe->bind_sampler_states(pipe, PIPE_SHADER_FRAGMENT, 0, 1, NULL);
   if (case_is(name, "min_filter")) ctx->sampler_states[1][0].min_img_filter ^= 1;
   if (case_is(name, "mip_filter")) ctx->sampler_states[1][0].min_mip_filter = PIPE_TEX_MIPFILTER_NONE;
   if (case_is(name, "max_lod")) ctx->sampler_states[1][0].max_lod = 0;
   if (!strncmp(name, "anisotropy_", 11)) ctx->sampler_states[1][0].max_anisotropy = strtoul(name + 11, NULL, 10);
   if (case_is(name, "swizzle")) ctx->sampler_views[1][0]->swizzle_r = PIPE_SWIZZLE_Z;
   if (case_is(name, "level_offset")) mgc_resource(textures[0])->levels[textures[0]->last_level].offset = SIZE_MAX;
   if (case_is(name, "level_stride")) mgc_resource(textures[0])->levels[textures[0]->last_level].stride = 0;
   if (case_is(name, "active_execution")) mgc_resource(textures[0])->active_host_executions = 1;
   struct pipe_transfer *transfer = NULL;
   if (case_is(name, "mapped")) {
      struct pipe_box box = {.width = 1, .height = 1, .depth = 1};
      assert(pipe->texture_map(pipe, textures[0], 0, PIPE_MAP_WRITE, &box, &transfer));
   }
   if (case_is(name, "feedback")) {
      struct pipe_framebuffer_state framebuffer = ctx->framebuffer;
      framebuffer.cbufs[0].texture = textures[0]; framebuffer.cbufs[0].format = textures[0]->format;
      pipe->set_framebuffer_state(pipe, &framebuffer);
   }
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
   struct pipe_draw_start_count_bias draw = {.count = 3};
   if (case_is(name, "max_instances")) info.instance_count = 256;
   if (case_is(name, "over_instances")) info.instance_count = 257;
   if (case_is(name, "max_vertices") || case_is(name, "over_vertices")) { info.instance_count = 256; draw.count = case_is(name, "max_vertices") ? 256 : 257; }
   if (case_is(name, "divisor")) { info.instance_count = 4; info.start_instance = 7; }
   if (case_is(name, "restart")) info.primitive_restart = true;
   if (case_is(name, "vertex_mapped")) mgc_resource(vertex)->active_maps = 1;
   struct pipe_resource *indices = NULL;
   if (case_is(name, "indexed")) {
      const uint16_t values[] = {2, 3, 4};
      indices = pipe_buffer_create(screen, PIPE_BIND_INDEX_BUFFER, PIPE_USAGE_DEFAULT, sizeof(values));
      assert(indices); memcpy(mgc_resource(indices)->data, values, sizeof(values));
      info.index_size = 2; info.index.resource = indices; draw.index_bias = -1;
   }
   struct features_probe probe = {.ctx = ctx, .color = color, .name = name};
   pthread_t thread;
   bool probing = case_is(name, "native_state") || case_is(name, "native_data") || case_is(name, "native_maps");
   if (probing) assert(!pthread_create(&thread, NULL, features_probe_run, &probe));
   pipe->draw_vbo(pipe, &info, 7, NULL, &draw, 1);
   if (probing) assert(!pthread_join(thread, NULL));
   struct mgc_runtime_status status; mgc_context_get_runtime_status(pipe, &status);
   if (status.error) fprintf(stderr, "draw rejection: %s\n", status.message);
   printf("{\"accepted\":%" PRIu64 ",\"rejected\":%" PRIu64 ",\"sequence\":%" PRIu64 ",\"error\":%d,\"initial_error\":%d,\"callbacks\":%u,\"rejected_maps\":%u,\"color_hex\":\"",
      status.accepted_draws, status.rejected_draws, status.sequence, status.error, initial_error, callbacks, probe.rejected_maps);
   struct mgc_resource *storage = mgc_resource(color);
   for (size_t index = 0; index < storage->size; index++) printf("%02x", ((uint8_t *)storage->data)[index]);
   printf("\",\"depth_hex\":\"");
   if (depth) for (size_t index = 0; index < mgc_resource(depth)->size; index++) printf("%02x", ((uint8_t *)mgc_resource(depth)->data)[index]);
   printf("\",\"pins\":%u}\n", storage->active_host_executions + (depth ? mgc_resource(depth)->active_host_executions : 0));
   if (transfer) pipe->texture_unmap(pipe, transfer);
   mgc_resource(vertex)->active_maps = 0;
   for (unsigned slot = 0; slot < 4; slot++) {
      if (textures[slot]) {
         mgc_resource(textures[slot])->levels[textures[slot]->last_level] = saved[slot].levels[textures[slot]->last_level];
         mgc_resource(textures[slot])->active_host_executions = 0;
         pipe_resource_reference(&textures[slot], NULL);
      }
   }
   pipe_resource_reference(&vertex, NULL); pipe_resource_reference(&indices, NULL);
   pipe_resource_reference(&color, NULL); pipe_resource_reference(&depth, NULL);
   pipe->destroy(pipe); assert(mgc_screen(screen)->allocated_bytes == 0);
   screen->destroy(screen); glsl_type_singleton_decref(); return 0;
}
