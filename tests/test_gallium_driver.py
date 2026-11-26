"""Exercise real Gallium hooks, without pretending the host driver renders."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "mesa/build"
DRIVER = ROOT / "mesa/src/gallium/drivers/mgc"
CASES = (
    "layout", "reject_layout", "buffer_map", "texture_map", "explicit_buffer",
    "explicit_texture", "budget", "bindings", "fences", "compile_error",
    "artifacts", "bad_draw", "copy_clear", "cross_context", "map_cleanup",
    "instances", "empty_draw", "command_limits", "shader_contract", "map_hazard",
    "display_formats",
)


SOURCE = r'''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "frontend/sw_winsys.h"
#include "util/u_inlines.h"
#include "mgc_context.h"
#include "mgc_compiler.h"
#include "mgc_public.h"
#include "mgc_resource.h"
#include "mgc_screen.h"

static struct pipe_resource
texture_template(void)
{
   return (struct pipe_resource){
      .target = PIPE_TEXTURE_2D_ARRAY, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 8, .height0 = 4, .depth0 = 1, .array_size = 3, .last_level = 3,
      .bind = PIPE_BIND_SAMPLER_VIEW, .usage = PIPE_USAGE_DEFAULT,
   };
}

static struct pipe_resource *
buffer(struct pipe_screen *screen, unsigned size)
{
   struct pipe_resource *resource =
      pipe_buffer_create(screen, PIPE_BIND_VERTEX_BUFFER, PIPE_USAGE_DEFAULT, size);
   assert(resource);
   return resource;
}

static void
layout_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource templat = texture_template();
   assert(screen->can_create_resource(screen, &templat));
   struct pipe_resource *resource = screen->resource_create(screen, &templat);
   assert(resource);
   struct mgc_resource *res = mgc_resource(resource);
   assert(res->size == 588);
   const size_t offsets[] = {0, 384, 512, 576};
   const unsigned strides[] = {32, 16, 8, 4};
   for (unsigned level = 0; level < 4; level++) {
      assert(res->levels[level].offset == offsets[level]);
      assert(res->levels[level].stride == strides[level]);
      assert(res->levels[level].layers == 3);
   }
   for (unsigned index = 0; index < res->size; index++)
      assert(((uint8_t *)res->data)[index] == 0);
   assert((uintptr_t)res->data % 64 == 0);
   pipe_resource_reference(&resource, NULL);
   templat.target = PIPE_TEXTURE_3D;
   templat.array_size = 1;
   templat.depth0 = 8;
   resource = screen->resource_create(screen, &templat);
   assert(resource);
   assert(mgc_resource(resource)->levels[0].layers == 8);
   assert(mgc_resource(resource)->levels[1].layers == 4);
   assert(mgc_resource(resource)->levels[3].layers == 1);
   pipe_resource_reference(&resource, NULL);
}

static void
reject_layout_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource good = texture_template();
   for (unsigned test = 0; test < 12; test++) {
      struct pipe_resource bad = good;
      switch (test) {
      case 0: bad.width0 = 0; break;
      case 1: bad.width0 = UINT_MAX; break;
      case 2: bad.last_level = 4; break;
      case 3: bad.nr_samples = 2; break;
      case 4: bad.format = PIPE_FORMAT_DXT1_RGB; break;
      case 5: bad.flags = PIPE_RESOURCE_FLAG_SPARSE; break;
      case 6: bad.array_size = 257; break;
      case 7: bad.target = PIPE_TEXTURE_CUBE; break;
      case 8: bad.bind = PIPE_BIND_SHARED; break;
      case 9: bad.depth0 = 2; break;
      case 10: bad.flags = PIPE_RESOURCE_FLAG_MAP_PERSISTENT; break;
      case 11: bad.target = PIPE_TEXTURE_RECT; break;
      }
      assert(!screen->can_create_resource(screen, &bad));
      assert(!screen->resource_create(screen, &bad));
   }
   struct pipe_resource huge = {
      .target = PIPE_BUFFER, .format = PIPE_FORMAT_R8_UNORM,
      .width0 = MGC_MAX_RESOURCE_BYTES + 1, .height0 = 1, .depth0 = 1, .array_size = 1,
   };
   assert(!screen->can_create_resource(screen, &huge));
   assert(!screen->is_format_supported(screen, PIPE_FORMAT_DXT1_RGB,
                                      PIPE_TEXTURE_2D, 0, 0, PIPE_BIND_SAMPLER_VIEW));
   assert(!screen->is_format_supported(screen, PIPE_FORMAT_R8_UNORM,
                                      PIPE_TEXTURE_2D, 0, 2, 0));
}

static void
buffer_map_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource *resource = buffer(screen, 32);
   struct pipe_box box = {.x = 4, .width = 8, .height = 1, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   uint8_t *mapped = pipe->buffer_map(pipe, resource, 0, PIPE_MAP_WRITE, &box, &transfer);
   assert(mapped && transfer && resource->reference.count == 2);
   assert(transfer->stride == 0 && transfer->layer_stride == 0);
   pipe_resource_reference(&resource, NULL);
   memset(mapped, 0xa5, 8);
   assert(((uint8_t *)mgc_resource(transfer->resource)->data)[4] == 0xa5);
   pipe->buffer_unmap(pipe, transfer);
   resource = buffer(screen, 32);
   for (unsigned test = 0; test < 9; test++) {
      struct pipe_box bad = box;
      unsigned level = 0, usage = PIPE_MAP_READ;
      switch (test) {
      case 0: bad.x = -1; break;
      case 1: bad.width = 29; break;
      case 2: bad.y = 1; break;
      case 3: bad.width = 0; break;
      case 4: level = 1; break;
      case 5: usage = 0; break;
      case 6: usage |= PIPE_MAP_FLUSH_EXPLICIT; break;
      case 7: usage |= PIPE_MAP_PERSISTENT; break;
      case 8: usage = PIPE_MAP_WRITE | PIPE_MAP_THREAD_SAFE; break;
      }
      transfer = (void *)(uintptr_t)1;
      assert(!pipe->buffer_map(pipe, resource, level, usage, &bad, &transfer));
      assert(transfer == NULL && resource->reference.count == 1);
   }
   pipe_resource_reference(&resource, NULL);
}

static void
texture_map_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource templat = texture_template();
   struct pipe_resource *resource = screen->resource_create(screen, &templat);
   assert(resource);
   struct pipe_box box = {.x = 1, .y = 1, .z = 2, .width = 2, .height = 1, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   uint8_t *mapped = pipe->texture_map(pipe, resource, 1, PIPE_MAP_WRITE, &box, &transfer);
   assert(mapped && transfer->stride == 16 && transfer->layer_stride == 32);
   assert(mapped == (uint8_t *)mgc_resource(resource)->data + 384 + 64 + 16 + 4);
   memset(mapped, 0x72, 8);
   pipe->texture_unmap(pipe, transfer);
   struct pipe_surface templat_surface = {
      .format = resource->format, .level = 1, .first_layer = 2, .last_layer = 2,
   };
   struct pipe_surface *surface = pipe->create_surface(pipe, resource, &templat_surface);
   assert(surface && surface->level == 1 && surface->first_layer == 2);
   templat_surface.level = 4;
   assert(!pipe->create_surface(pipe, resource, &templat_surface));
   pipe_resource_reference(&resource, NULL);
   assert(surface->texture->reference.count == 1);
   pipe->surface_destroy(pipe, surface);
}

static void
explicit_buffer_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource *resource = buffer(screen, 32);
   memset(mgc_resource(resource)->data, 0x11, 32);
   struct pipe_box box = {.x = 4, .width = 8, .height = 1, .depth = 1};
   struct pipe_box flush = {.x = 2, .width = 3, .height = 1, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   uint8_t *mapped = pipe->buffer_map(pipe, resource, 0,
      PIPE_MAP_WRITE | PIPE_MAP_FLUSH_EXPLICIT, &box, &transfer);
   assert(mapped);
   assert(((uintptr_t)mapped - box.x) % 64 == 0);
   memset(mapped, 0xa5, 8);
   assert(((uint8_t *)mgc_resource(resource)->data)[6] == 0x11);
   pipe->transfer_flush_region(pipe, transfer, &flush);
   flush.x = 7;
   pipe->transfer_flush_region(pipe, transfer, &flush);
   pipe->buffer_unmap(pipe, transfer);
   for (unsigned index = 0; index < 32; index++)
      assert(((uint8_t *)mgc_resource(resource)->data)[index] ==
             (index >= 6 && index < 9 ? 0xa5 : 0x11));
   pipe_resource_reference(&resource, NULL);
}

static void
explicit_texture_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource templat = texture_template();
   struct pipe_resource *resource = screen->resource_create(screen, &templat);
   struct pipe_box box = {.x = 1, .z = 1, .width = 3, .height = 2, .depth = 2};
   struct pipe_box flush = {.x = 1, .y = 1, .z = 1, .width = 1, .height = 1, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   uint8_t *mapped = pipe->texture_map(pipe, resource, 1,
      PIPE_MAP_WRITE | PIPE_MAP_FLUSH_EXPLICIT, &box, &transfer);
   assert(mapped);
   size_t location = transfer->layer_stride + transfer->stride + 4;
   memset(mapped + location, 0x66, 4);
   mapped[0] = 0x99;
   pipe->transfer_flush_region(pipe, transfer, &flush);
   pipe->texture_unmap(pipe, transfer);
   const uint8_t *bytes = mgc_resource(resource)->data;
   assert(bytes[384 + 32 + 4] == 0);
   assert(bytes[384 + 64 + 16 + 8] == 0x66);
   pipe_resource_reference(&resource, NULL);
}

static void
budget_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   size_t baseline = mgc_screen(screen)->allocated_bytes;
   assert(mgc_reserve_memory(screen, MGC_MAX_SCREEN_BYTES - baseline));
   struct pipe_resource templat = texture_template();
   assert(!screen->resource_create(screen, &templat));
   assert(!mgc_reserve_memory(screen, 1));
   mgc_release_memory(screen, MGC_MAX_SCREEN_BYTES - baseline);
   struct pipe_resource *resource = buffer(screen, 32);
   size_t reserve = MGC_MAX_SCREEN_BYTES - mgc_screen(screen)->allocated_bytes;
   assert(mgc_reserve_memory(screen, reserve));
   struct pipe_box box = {.width = 8, .height = 1, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   assert(!pipe->buffer_map(pipe, resource, 0, PIPE_MAP_WRITE, &box, &transfer));
   assert(!transfer && resource->reference.count == 1);
   mgc_release_memory(screen, reserve);
   pipe_resource_reference(&resource, NULL);
   assert(mgc_screen(screen)->allocated_bytes == baseline);
}

static void
bindings_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource *resource = buffer(screen, 32);
   struct pipe_constant_buffer constant = {.buffer = resource, .buffer_size = 16};
   pipe->set_constant_buffer(pipe, PIPE_SHADER_VERTEX, 0, false, &constant);
   assert(resource->reference.count == 2);
   pipe_resource_reference(&resource, NULL);
   assert(mgc_context(pipe)->constants[0][0].buffer->reference.count == 1);
   pipe->set_constant_buffer(pipe, PIPE_SHADER_VERTEX, 0, false, NULL);
   uint32_t user[] = {1, 2, 3, 4};
   constant = (struct pipe_constant_buffer){.user_buffer = user, .buffer_size = sizeof(user)};
   pipe->set_constant_buffer(pipe, PIPE_SHADER_FRAGMENT, 0, false, &constant);
   user[0] = 99;
   assert(((uint32_t *)mgc_resource(mgc_context(pipe)->constants[1][0].buffer)->data)[0] == 1);
   pipe->set_constant_buffer(pipe, PIPE_SHADER_FRAGMENT, 0, false, NULL);
   resource = buffer(screen, 32);
   constant = (struct pipe_constant_buffer){.buffer = resource, .buffer_size = 16};
   pipe->set_constant_buffer(pipe, PIPE_SHADER_VERTEX, 0, true, &constant);
   assert(resource->reference.count == 1);
   pipe->set_constant_buffer(pipe, PIPE_SHADER_VERTEX, 0, false, NULL);
   resource = buffer(screen, 32);
   struct pipe_vertex_buffer vertex = {.buffer.resource = resource};
   pipe->set_vertex_buffers(pipe, 1, &vertex);
   assert(resource->reference.count == 1);
   pipe->set_vertex_buffers(pipe, 0, NULL);
   struct pipe_resource templat = texture_template();
   resource = screen->resource_create(screen, &templat);
   struct pipe_sampler_view view_template = {
      .format = resource->format, .target = resource->target,
      .u.tex = {.first_level = 0, .last_level = 3, .first_layer = 0, .last_layer = 2},
   };
   struct pipe_sampler_view *view = pipe->create_sampler_view(pipe, resource, &view_template);
   assert(view);
   pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 0, &view);
   pipe_sampler_view_reference(&view, NULL);
   pipe_resource_reference(&resource, NULL);
   assert(mgc_context(pipe)->sampler_views[1][0]->texture->reference.count == 1);
   pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 0, 1, NULL);
   resource = screen->resource_create(screen, &templat);
   struct pipe_framebuffer_state fb = {.width = 4, .height = 2, .nr_cbufs = 1};
   fb.cbufs[0] = (struct pipe_surface){
      .texture = resource, .format = resource->format, .level = 1,
      .first_layer = 1, .last_layer = 1,
   };
   pipe->set_framebuffer_state(pipe, &fb);
   assert(resource->reference.count == 2);
   pipe_resource_reference(&resource, NULL);
   fb = (struct pipe_framebuffer_state){0};
   pipe->set_framebuffer_state(pipe, &fb);
}

static void
fences_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_fence_handle *fence = NULL, *alias = NULL;
   pipe->flush(pipe, &fence, PIPE_FLUSH_DEFERRED);
   assert(fence && screen->fence_finish(screen, NULL, fence, 0));
   uint64_t first = 0, next = 0;
   int error = 0;
   assert(mgc_fence_get_status(screen, fence, &first, &error) && first && !error);
   screen->fence_reference(screen, &alias, fence);
   screen->fence_reference(screen, &alias, alias);
   assert(((struct mgc_fence *)alias)->reference.count == 2);
   mgc_context_error(pipe, PIPE_ERROR_BAD_INPUT, "test failure");
   pipe->flush(pipe, &fence, 0);
   assert(mgc_fence_get_status(screen, fence, &next, &error));
   assert(next > first && error == PIPE_ERROR_BAD_INPUT);
   assert(mgc_fence_get_status(screen, alias, &next, &error));
   assert(next == first && !error);
   pipe->flush(pipe, &fence, PIPE_FLUSH_FENCE_FD);
   assert(!fence && strstr(mgc_context(pipe)->status.message, "unsupported"));
   struct pipe_screen *other = mgc_create_screen(NULL);
   assert(!other->fence_finish(other, pipe, alias, 0));
   other->destroy(other);
   screen->fence_reference(screen, &fence, NULL);
   pipe->memory_barrier(pipe, PIPE_BARRIER_MAPPED_BUFFER);
   pipe->texture_barrier(pipe, 0);
   assert(screen->fence_finish(screen, pipe, alias, UINT64_MAX));
   screen->fence_reference(screen, &alias, NULL);
}

static void *
shader(struct pipe_context *pipe, gl_shader_stage stage, bool invalid)
{
   nir_builder builder = nir_builder_init_simple_shader(stage,
      mgc_test_get_compiler_options(), "runtime_test");
   nir_def *value = invalid ? nir_i2f32(&builder, nir_load_sample_id(&builder)) :
      nir_imm_float(&builder, 0.5);
   nir_store_output(&builder, value, nir_imm_int(&builder, 0), .component = 0);
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR, .ir.nir = builder.shader, .report_compile_error = true,
   };
   void *result = stage == MESA_SHADER_VERTEX ? pipe->create_vs_state(pipe, &state) :
      pipe->create_fs_state(pipe, &state);
   if (invalid) {
      assert(!result && state.error_message && *state.error_message);
      free(state.error_message);
   } else {
      assert(result && !state.error_message);
   }
   return result;
}

static void
compile_error_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   assert(!shader(pipe, MESA_SHADER_FRAGMENT, true));
   struct mgc_runtime_status status;
   mgc_context_get_runtime_status(pipe, &status);
   assert(status.error == PIPE_ERROR_BAD_INPUT && strstr(status.message, "load_sample_id"));
   mgc_context_clear_error(pipe);
   mgc_context_get_runtime_status(pipe, &status);
   assert(!status.error && !status.message[0]);
}

struct capture { unsigned count; unsigned drawid[4]; uint64_t sequence; bool reject; };

static bool
capture_draw(void *data, const struct mgc_draw_artifact *artifact)
{
   struct capture *capture = data;
   assert(artifact->runtime_abi_version == MGC_RUNTIME_ABI_VERSION);
   assert(artifact->shader_abi_version == MGC_ABI_VERSION);
   assert(strstr(artifact->assembly[0], "stop_core"));
   assert(strstr(artifact->assembly[1], "stop_core"));
   assert(artifact->sequence > capture->sequence);
   capture->sequence = artifact->sequence;
   capture->drawid[capture->count++] = artifact->drawid;
   if (artifact->constants[1][0].data) {
      assert(artifact->constants[1][0].size == 16);
      assert(((const uint32_t *)artifact->constants[1][0].data)[0] == 7);
   }
   return !capture->reject;
}

static void
bind_shaders(struct pipe_context *pipe)
{
   void *vertex = shader(pipe, MESA_SHADER_VERTEX, false);
   void *fragment = shader(pipe, MESA_SHADER_FRAGMENT, false);
   pipe->bind_vs_state(pipe, vertex);
   pipe->bind_fs_state(pipe, fragment);
   pipe->delete_vs_state(pipe, vertex);
   pipe->delete_fs_state(pipe, fragment);
}

static void
artifacts_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   bind_shaders(pipe);
   uint32_t user[] = {7, 8, 9, 10};
   struct pipe_constant_buffer constant = {.user_buffer = user, .buffer_size = sizeof(user)};
   pipe->set_constant_buffer(pipe, PIPE_SHADER_FRAGMENT, 0, false, &constant);
   user[0] = 99;
   struct capture capture = {0};
   mgc_context_set_artifact_callback(pipe, capture_draw, &capture);
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1,
                                 .increment_draw_id = true};
   struct pipe_draw_start_count_bias draws[] = {{0, 3, 0}, {3, 3, 0}};
   pipe->draw_vbo(pipe, &info, 41, NULL, draws, 2);
   assert(capture.count == 2 && capture.drawid[0] == 41 && capture.drawid[1] == 42);
   struct mgc_runtime_status status;
   mgc_context_get_runtime_status(pipe, &status);
   assert(status.accepted_draws == 2 && !status.rejected_draws && !status.error);
   capture.reject = true;
   pipe->draw_vbo(pipe, &info, 43, NULL, draws, 1);
   mgc_context_get_runtime_status(pipe, &status);
   assert(status.accepted_draws == 2 && status.rejected_draws == 1 && status.error);
}

static void
bad_draw_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   bind_shaders(pipe);
   struct capture capture = {0};
   mgc_context_set_artifact_callback(pipe, capture_draw, &capture);
   struct pipe_resource *resource = buffer(screen, 16);
   struct pipe_vertex_buffer vertex = {.buffer.resource = resource};
   pipe->set_vertex_buffers(pipe, 1, &vertex);
   struct pipe_vertex_element element = {.src_format = PIPE_FORMAT_R32_FLOAT,
                                         .src_stride = 4, .vertex_buffer_index = 0};
   void *elements = pipe->create_vertex_elements_state(pipe, 1, &element);
   assert(elements);
   pipe->bind_vertex_elements_state(pipe, elements);
   pipe->delete_vertex_elements_state(pipe, elements);
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
   struct pipe_draw_start_count_bias draw = {0, 5, 0};
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   assert(!capture.count);
   draw.count = 3;
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   assert(capture.count == 1);
   struct pipe_resource *indices = buffer(screen, 4);
   uint16_t values[] = {0, 15};
   memcpy(mgc_resource(indices)->data, values, sizeof(values));
   info.index_size = 2;
   info.index.resource = indices;
   info.take_index_buffer_ownership = true;
   draw.count = 2;
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   assert(capture.count == 1);
   info = (struct pipe_draw_info){.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
   mgc_context_set_artifact_callback(pipe, NULL, NULL);
   draw.count = 3;
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   struct mgc_runtime_status status;
   mgc_context_get_runtime_status(pipe, &status);
   assert(strstr(status.message, "not rendered"));
}

static void
copy_clear_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource *resource = buffer(screen, 32);
   uint8_t pattern = 0xa5;
   pipe->clear_buffer(pipe, resource, 4, 12, &pattern, 1);
   uint8_t *bytes = mgc_resource(resource)->data;
   for (unsigned index = 0; index < 32; index++)
      assert(bytes[index] == (index >= 4 && index < 16 ? 0xa5 : 0));
   for (unsigned index = 0; index < 16; index++)
      bytes[index] = index;
   struct pipe_box box = {.width = 12, .height = 1, .depth = 1};
   pipe->resource_copy_region(pipe, resource, 0, 4, 0, 0, resource, 0, &box);
   for (unsigned index = 0; index < 12; index++)
      assert(bytes[4 + index] == index);
   box.width = 33;
   pipe->resource_copy_region(pipe, resource, 0, 0, 0, 0, resource, 0, &box);
   assert(mgc_context(pipe)->status.error == PIPE_ERROR_BAD_INPUT);
   pipe_resource_reference(&resource, NULL);
}

static void
cross_context_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_screen *other = mgc_create_screen(NULL);
   struct pipe_resource *foreign = buffer(other, 16);
   struct pipe_box box = {.width = 8, .height = 1, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   assert(!pipe->buffer_map(pipe, foreign, 0, PIPE_MAP_READ, &box, &transfer));
   assert(!transfer);
   pipe_resource_reference(&foreign, NULL);
   other->destroy(other);
   struct pipe_context *peer = screen->context_create(screen, NULL, 0);
   struct pipe_resource *resource = buffer(screen, 16);
   uint8_t *mapped = pipe->buffer_map(pipe, resource, 0,
      PIPE_MAP_WRITE | PIPE_MAP_FLUSH_EXPLICIT, &box, &transfer);
   assert(mapped);
   memset(mapped, 0x55, 8);
   pipe->transfer_flush_region(peer, transfer, &box);
   assert(((uint8_t *)mgc_resource(resource)->data)[0] == 0);
   peer->buffer_unmap(peer, transfer);
   assert(resource->reference.count == 2);
   pipe->buffer_unmap(pipe, transfer);
   peer->destroy(peer);
   pipe_resource_reference(&resource, NULL);
}

static void
map_cleanup_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_context *peer = screen->context_create(screen, NULL, 0);
   struct pipe_resource *resource = buffer(screen, 16);
   struct pipe_box box = {.width = 8, .height = 1, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   assert(peer->buffer_map(peer, resource, 0,
      PIPE_MAP_WRITE | PIPE_MAP_FLUSH_EXPLICIT, &box, &transfer));
   pipe_resource_reference(&resource, NULL);
   peer->destroy(peer);
}

static void
instances_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   bind_shaders(pipe);
   struct capture capture = {0};
   mgc_context_set_artifact_callback(pipe, capture_draw, &capture);
   struct pipe_resource *resource = buffer(screen, 16);
   struct pipe_vertex_buffer vertex = {.buffer.resource = resource};
   pipe->set_vertex_buffers(pipe, 1, &vertex);
   struct pipe_vertex_element element = {
      .src_format = PIPE_FORMAT_R32_FLOAT, .src_stride = 4,
      .vertex_buffer_index = 0, .instance_divisor = 2,
   };
   void *elements = pipe->create_vertex_elements_state(pipe, 1, &element);
   pipe->bind_vertex_elements_state(pipe, elements);
   pipe->delete_vertex_elements_state(pipe, elements);
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 3,
                                 .start_instance = 3};
   struct pipe_draw_start_count_bias draw = {0, 3, 0};
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   assert(!capture.count);
   info.start_instance = 2;
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   assert(capture.count == 1);
}

static void
empty_draw_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
   struct pipe_draw_start_count_bias draw = {0, 0, 0};
   pipe->draw_vbo(pipe, &info, 0, NULL, NULL, 0);
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   draw.count = 3;
   info.instance_count = 0;
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   assert(!mgc_context(pipe)->status.error && !mgc_context(pipe)->status.rejected_draws);
}

static void
command_limits_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   bind_shaders(pipe);
   struct capture capture = {0};
   mgc_context_set_artifact_callback(pipe, capture_draw, &capture);
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
   struct pipe_draw_start_count_bias draws[] = {{0, 600000, 0}, {0, 600000, 0}};
   pipe->draw_vbo(pipe, &info, 0, NULL, draws, 2);
   assert(!capture.count && strstr(mgc_context(pipe)->status.message, "budget"));
   draws[0].count = 3;
   struct pipe_draw_indirect_info indirect = {0};
   pipe->draw_vbo(pipe, &info, 0, &indirect, draws, 1);
   assert(!capture.count);
   info.increment_draw_id = true;
   draws[1].count = 3;
   pipe->draw_vbo(pipe, &info, UINT_MAX, NULL, draws, 2);
   assert(!capture.count);
   info.increment_draw_id = false;
   info.instance_count = 1048577;
   pipe->draw_vbo(pipe, &info, 0, NULL, draws, 1);
   assert(!capture.count);
}

static void
shader_contract_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_shader_state state = {.type = PIPE_SHADER_IR_NIR, .report_compile_error = true};
   assert(!pipe->create_vs_state(pipe, &state) && state.error_message);
   free(state.error_message);
   nir_builder builder = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      mgc_test_get_compiler_options(), "wrong_stage");
   state = (struct pipe_shader_state){.type = PIPE_SHADER_IR_NIR, .ir.nir = builder.shader,
                                      .report_compile_error = true};
   assert(!pipe->create_vs_state(pipe, &state) && state.error_message);
   assert(strstr(state.error_message, "stage"));
   free(state.error_message);
   assert(!screen->context_create(screen, NULL, PIPE_CONTEXT_COMPUTE_ONLY));
   assert(!screen->caps.buffer_map_persistent_coherent);
   assert(!screen->caps.allow_mapped_buffers_during_execution);
   assert(!screen->caps.user_vertex_buffers);
   assert(screen->caps.texture_transfer_modes == PIPE_TEXTURE_TRANSFER_DEFAULT);
}

static void
map_hazard_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   bind_shaders(pipe);
   struct capture capture = {0};
   mgc_context_set_artifact_callback(pipe, capture_draw, &capture);
   struct pipe_resource *resource = buffer(screen, 16);
   struct pipe_constant_buffer constant = {.buffer = resource, .buffer_size = 16};
   pipe->set_constant_buffer(pipe, PIPE_SHADER_VERTEX, 0, false, &constant);
   struct pipe_box box = {.width = 16, .height = 1, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   assert(pipe->buffer_map(pipe, resource, 0, PIPE_MAP_WRITE, &box, &transfer));
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
   struct pipe_draw_start_count_bias draw = {0, 3, 0};
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   assert(!capture.count && strstr(mgc_context(pipe)->status.message, "mapped"));
   pipe->buffer_unmap(pipe, transfer);
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);
   assert(capture.count == 1);
   pipe_resource_reference(&resource, NULL);
}

static bool
display_format(struct sw_winsys *winsys, unsigned bind, enum pipe_format format)
{
   return format == PIPE_FORMAT_B8G8R8A8_UNORM;
}

static void
display_formats_test(struct pipe_screen *screen, struct pipe_context *pipe)
{
   const char *host_texture = getenv("MGC_HOST_TEXTURE");
   const char *compile_target = getenv("MGC_COMPILE_TARGET");
   bool texture_lod = host_texture &&
      (!strcmp(host_texture, "9") || !strcmp(host_texture, "10")) &&
      (!compile_target || !strcmp(compile_target, "legacy-mgcsim-v1"));
   const char *host_instancing = getenv("MGC_HOST_INSTANCING");
   bool instancing = host_instancing && !strcmp(host_instancing, "1") &&
      (!compile_target || !strcmp(compile_target, "legacy-mgcsim-v1"));
   assert(screen->caps.fragment_shader_texture_lod == texture_lod);
   assert(screen->caps.vertex_color_unclamped == texture_lod);
   assert(screen->caps.vs_instanceid == instancing);
   assert(screen->caps.vertex_element_instance_divisor == instancing);
   struct sw_winsys winsys = {.is_displaytarget_format_supported = display_format};
   struct pipe_screen *display = mgc_create_screen(&winsys);
   assert(display->caps.fragment_shader_texture_lod == texture_lod);
   assert(display->caps.vertex_color_unclamped == texture_lod);
   assert(display->caps.vs_instanceid == instancing);
   assert(display->caps.vertex_element_instance_divisor == instancing);
   unsigned bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_DISPLAY_TARGET;
   assert(display->is_format_supported(display, PIPE_FORMAT_B8G8R8A8_UNORM,
                                      PIPE_TEXTURE_2D, 0, 0, bind));
   assert(!display->is_format_supported(display, PIPE_FORMAT_R8G8B8A8_UNORM,
                                       PIPE_TEXTURE_2D, 0, 0, bind));
   assert(!screen->is_format_supported(screen, PIPE_FORMAT_B8G8R8A8_UNORM,
                                      PIPE_TEXTURE_2D, 0, 0, bind));
   assert(!display->is_format_supported(display, PIPE_FORMAT_B8G8R8A8_UNORM,
                                       PIPE_TEXTURE_2D, 0, 0, bind | PIPE_BIND_SHARED));
   display->destroy(display);
}

int
main(int argc, char **argv)
{
   assert(argc == 2);
   glsl_type_singleton_init_or_ref();
   struct pipe_screen *screen = mgc_create_screen(NULL);
   assert(screen);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   assert(pipe);
   typedef void (*test_func)(struct pipe_screen *, struct pipe_context *);
   const struct { const char *name; test_func run; } tests[] = {
      {"layout", layout_test}, {"reject_layout", reject_layout_test},
      {"buffer_map", buffer_map_test}, {"texture_map", texture_map_test},
      {"explicit_buffer", explicit_buffer_test}, {"explicit_texture", explicit_texture_test},
      {"budget", budget_test}, {"bindings", bindings_test}, {"fences", fences_test},
      {"compile_error", compile_error_test}, {"artifacts", artifacts_test},
      {"bad_draw", bad_draw_test}, {"copy_clear", copy_clear_test},
      {"cross_context", cross_context_test}, {"map_cleanup", map_cleanup_test},
      {"instances", instances_test}, {"empty_draw", empty_draw_test},
      {"command_limits", command_limits_test}, {"shader_contract", shader_contract_test},
      {"map_hazard", map_hazard_test}, {"display_formats", display_formats_test},
   };
   bool found = false;
   for (unsigned index = 0; index < sizeof(tests) / sizeof(tests[0]); index++) {
      if (!strcmp(argv[1], tests[index].name)) {
         tests[index].run(screen, pipe);
         found = true;
         break;
      }
   }
   assert(found);
   pipe->destroy(pipe);
   assert(mgc_screen(screen)->allocated_bytes == 0);
   screen->destroy(screen);
   glsl_type_singleton_decref();
   puts("ok");
   return 0;
}
'''


def run_checked(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def runtime_helper(tmp_path_factory):
    if not (BUILD / "compile_commands.json").is_file():
        pytest.skip("configure Mesa with the mgc Gallium driver first")
    ninja = ROOT / ".venv/bin/ninja"
    if not ninja.is_file():
        executable = shutil.which("ninja")
        if not executable:
            pytest.skip("ninja is required to build the Gallium runtime test")
        ninja = Path(executable)
    environment = os.environ.copy()
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment["PATH"]
    run_checked([str(ninja), "-C", str(BUILD),
                 "src/gallium/drivers/mgc/libmgc.a"], env=environment)
    commands = json.loads((BUILD / "compile_commands.json").read_text())
    entry = next((item for item in commands if item["file"].endswith("/mgc_context.c")), None)
    if entry is None:
        pytest.skip("Mesa was not configured with mgc")
    arguments = shlex.split(entry["command"])
    compile_flags = []
    cursor = 1
    while cursor < len(arguments):
        argument = arguments[cursor]
        if argument in ("-o", "-c", "-MF", "-MQ", "-MT"):
            cursor += 2
        elif argument == "-MD":
            cursor += 1
        else:
            compile_flags.append(argument)
            cursor += 1
    temp = tmp_path_factory.mktemp("gallium-runtime")
    source = temp / "runtime.c"
    source.write_text(SOURCE)
    object_file = temp / "runtime.o"
    sanitized = os.environ.get("MGC_RUNTIME_SANITIZE") == "1"
    if sanitized:
        compile_flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    run_checked([arguments[0], *compile_flags, "-I" + str(DRIVER),
                 "-o", str(object_file), "-c", str(source)], cwd=BUILD)
    link_line = run_checked([str(ninja), "-C", str(BUILD), "-t", "commands",
                             "src/gallium/drivers/mgc/mgc_test_arith"], env=environment)
    link = shlex.split(link_line.strip().splitlines()[-1])
    helper = temp / "runtime"
    link[link.index("-o") + 1] = str(helper)
    original_object = "src/gallium/drivers/mgc/mgc_test_arith.p/mgc_test_arith.c.o"
    link[link.index(original_object)] = str(object_file)
    if sanitized:
        driver_objects = []
        for filename in ("mgc_context.c", "mgc_resource.c", "mgc_screen.c",
                         "mgc_state.c", "mgc_compiler.c", "mgc_vector_compiler.c", "mgc_host_executor.c"):
            output = temp / (filename + ".o")
            run_checked([arguments[0], *compile_flags, "-o", str(output),
                         "-c", str(DRIVER / filename)], cwd=BUILD)
            driver_objects.append(str(output))
        position = link.index("src/gallium/drivers/mgc/libmgc.a")
        link[position:position + 1] = driver_objects
        link += ["-fsanitize=address,undefined"]
    run_checked(link, cwd=BUILD)
    return helper


@pytest.mark.parametrize("case", CASES)
def test_gallium_runtime(runtime_helper, case):
    result = subprocess.run([str(runtime_helper), case], text=True, capture_output=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"
