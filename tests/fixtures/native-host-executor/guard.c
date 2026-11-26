#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

int mgc_texture_fixture_main(int argc, char **argv);
#define main mgc_texture_fixture_main
#include "../texture-capture/guard.c"
#undef main

int __real_getsockopt(int descriptor, int level, int option, void *value, socklen_t *length);
int __wrap_getsockopt(int descriptor, int level, int option, void *value, socklen_t *length);

int
__wrap_getsockopt(int descriptor, int level, int option, void *value, socklen_t *length)
{
   int result = __real_getsockopt(descriptor, level, option, value, length);
   const char *mode = getenv("MGC_TEST_PEER");
   if (mode && level == SOL_SOCKET && option == SO_PEERCRED && !result) {
      if (!strcmp(mode, "uid")) ((struct ucred *)value)->uid = geteuid() + 1;
      else if (!strcmp(mode, "size")) *length = sizeof(struct ucred) - 1;
      else if (!strcmp(mode, "error")) return -1;
   }
   return result;
}

struct pin_probe {
   struct pipe_context *pipe;
   struct pipe_resource *resource;
   struct pipe_resource *depth;
   struct mgc_context *owner;
   const char *name;
   unsigned rejected;
   bool recovered;
};

static unsigned
host_pins(struct pipe_resource *resource)
{
   struct mgc_screen *screen = mgc_screen(resource->screen);
   simple_mtx_lock(&screen->lock);
   unsigned count = mgc_resource(resource)->active_host_executions;
   simple_mtx_unlock(&screen->lock);
   return count;
}

static void *
probe_pin(void *data)
{
   struct pin_probe *probe = data;
   for (unsigned attempt = 0; attempt < 2000 && !host_pins(probe->resource); attempt++)
      usleep(1000);
   assert(host_pins(probe->resource) == 1);
   if (probe->depth)
      assert(host_pins(probe->depth) == 1);
   const unsigned modes[] = {PIPE_MAP_READ, PIPE_MAP_WRITE, PIPE_MAP_WRITE | PIPE_MAP_FLUSH_EXPLICIT};
   struct pipe_box box = {.width = 3, .height = 2, .depth = 1};
   struct mgc_screen *screen = mgc_screen(probe->pipe->screen);
   size_t budget = screen->allocated_bytes;
   union pipe_color_union color = {.f = {1, 1, 1, 1}};
   probe->pipe->clear(probe->pipe, PIPE_CLEAR_COLOR0 | (probe->depth ? PIPE_CLEAR_DEPTH : 0),
                      NULL, &color, 1, 0);
   assert(mgc_context(probe->pipe)->status.error == PIPE_ERROR_BAD_INPUT);
   mgc_context_clear_error(probe->pipe);
   probe->rejected++;
   struct pipe_resource *resources[] = {probe->resource, probe->depth};
   for (unsigned target = 0; target < (probe->depth ? 2 : 1); target++) {
      assert(mgc_resource_is_mapped(resources[target]));
      for (unsigned index = 0; index < 3; index++) {
         struct pipe_transfer *transfer = (struct pipe_transfer *)(uintptr_t)1;
         assert(!probe->pipe->texture_map(probe->pipe, resources[target], 0, modes[index], &box, &transfer));
         assert(!transfer && !mgc_context(probe->pipe)->transfers);
         assert(mgc_context(probe->pipe)->status.error == PIPE_ERROR_BAD_INPUT);
         mgc_context_clear_error(probe->pipe);
         probe->rejected++;
      }
      assert(screen->allocated_bytes == budget);
      struct pipe_resource template = {
         .target = PIPE_TEXTURE_2D, .format = resources[target]->format,
         .width0 = 3, .height0 = 2, .depth0 = 1, .array_size = 1,
         .bind = target ? PIPE_BIND_DEPTH_STENCIL : PIPE_BIND_RENDER_TARGET,
         .usage = PIPE_USAGE_DEFAULT,
      };
      struct pipe_resource *other = probe->pipe->screen->resource_create(probe->pipe->screen, &template);
      assert(other);
      memset(mgc_resource(other)->data, 0x71, mgc_resource(other)->size);
      probe->pipe->resource_copy_region(probe->pipe, other, 0, 0, 0, 0, resources[target], 0, &box);
      assert(mgc_context(probe->pipe)->status.error == PIPE_ERROR_BAD_INPUT);
      for (size_t index = 0; index < mgc_resource(other)->size; index++)
         assert(((uint8_t *)mgc_resource(other)->data)[index] == 0x71);
      mgc_context_clear_error(probe->pipe);
      probe->rejected++;
      probe->pipe->resource_copy_region(probe->pipe, resources[target], 0, 0, 0, 0, other, 0, &box);
      assert(mgc_context(probe->pipe)->status.error == PIPE_ERROR_BAD_INPUT);
      assert(!mgc_context(probe->pipe)->transfers);
      mgc_context_clear_error(probe->pipe);
      probe->rejected++;
      pipe_resource_reference(&other, NULL);
   }
   struct pipe_resource *replacement = NULL;
   unsigned saved_height = probe->depth ? probe->depth->height0 : 0;
   unsigned saved_function = probe->owner->depth_stencil_state.depth_func;
   if (is_case(probe->name, "z_pin_depth_identity") || is_case(probe->name, "z_pin_color_identity")) {
      bool color_target = is_case(probe->name, "z_pin_color_identity");
      struct pipe_resource template = {
         .target = PIPE_TEXTURE_2D,
         .format = color_target ? PIPE_FORMAT_R8G8B8A8_UNORM : PIPE_FORMAT_Z16_UNORM,
         .width0 = 3, .height0 = 2, .depth0 = 1, .array_size = 1,
         .bind = color_target ? PIPE_BIND_RENDER_TARGET : PIPE_BIND_DEPTH_STENCIL,
         .usage = PIPE_USAGE_DEFAULT,
      };
      replacement = probe->pipe->screen->resource_create(probe->pipe->screen, &template);
      assert(replacement);
      simple_mtx_lock(&screen->lock);
      pipe_resource_reference(color_target ? &probe->owner->framebuffer.cbufs[0].texture :
         &probe->owner->framebuffer.zsbuf.texture, replacement);
      simple_mtx_unlock(&screen->lock);
   }
   simple_mtx_lock(&screen->lock);
   if (is_case(probe->name, "z_pin_depth_geometry")) probe->depth->height0++;
   if (is_case(probe->name, "z_pin_state")) probe->owner->depth_stencil_state.depth_func = PIPE_FUNC_ALWAYS;
   simple_mtx_unlock(&screen->lock);
   for (unsigned attempt = 0; attempt < 7000 && host_pins(probe->resource); attempt++)
      usleep(1000);
   assert(!host_pins(probe->resource));
   simple_mtx_lock(&screen->lock);
   if (probe->depth) probe->depth->height0 = saved_height;
   probe->owner->depth_stencil_state.depth_func = saved_function;
   if (replacement) {
      bool color_target = is_case(probe->name, "z_pin_color_identity");
      pipe_resource_reference(color_target ? &probe->owner->framebuffer.cbufs[0].texture :
         &probe->owner->framebuffer.zsbuf.texture, color_target ? probe->resource : probe->depth);
   }
   simple_mtx_unlock(&screen->lock);
   pipe_resource_reference(&replacement, NULL);
   for (unsigned target = 0; target < (probe->depth ? 2 : 1); target++) {
      assert(!host_pins(resources[target]));
      struct pipe_transfer *transfer = NULL;
      assert(probe->pipe->texture_map(probe->pipe, resources[target], 0, PIPE_MAP_READ, &box, &transfer));
      assert(transfer);
      probe->pipe->texture_unmap(probe->pipe, transfer);
   }
   assert(!mgc_context(probe->pipe)->status.error && !mgc_context(probe->pipe)->transfers);
   probe->recovered = true;
   return NULL;
}

static void
bind_draw_state(struct pipe_context *pipe, const char *name, unsigned width, unsigned height)
{
   struct pipe_blend_state blend = {.rt[0].colormask = 15};
   if (is_case(name, "source_over") || is_case(name, "z_source_over")) {
      blend.rt[0].blend_enable = 1;
      blend.rt[0].rgb_func = blend.rt[0].alpha_func = PIPE_BLEND_ADD;
      blend.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_SRC_ALPHA;
      blend.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_INV_SRC_ALPHA;
      blend.rt[0].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
      blend.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_INV_SRC_ALPHA;
   }
   if (is_case(name, "blend_mask") || is_case(name, "clear_mask") || is_case(name, "clear_z_mask"))
      blend.rt[0].colormask = 7;
   if (is_case(name, "blend_logicop")) blend.logicop_enable = 1;
   if (is_case(name, "blend_dither")) blend.dither = 1;
   if (is_case(name, "blend_alpha_coverage")) blend.alpha_to_coverage = 1;
   if (is_case(name, "blend_independent")) blend.independent_blend_enable = 1;
   if (is_case(name, "blend_factors")) {
      blend.rt[0].blend_enable = 1;
      blend.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_ONE;
   }
   void *state = pipe->create_blend_state(pipe, &blend);
   assert(state);
   pipe->bind_blend_state(pipe, state);
   ((struct pipe_blend_state *)state)->logicop_enable = 1;
   pipe->delete_blend_state(pipe, state);
   assert(!memcmp(&blend, &mgc_context(pipe)->blend_state, sizeof(blend)));
   struct pipe_rasterizer_state rasterizer = {
      .half_pixel_center = 1, .depth_clip_near = 1, .depth_clip_far = 1,
      .fill_front = PIPE_POLYGON_MODE_FILL, .fill_back = PIPE_POLYGON_MODE_FILL,
   };
   if (is_case(name, "raster_scissor")) rasterizer.scissor = 1;
   if (is_case(name, "raster_cull")) rasterizer.cull_face = PIPE_FACE_BACK;
   if (is_case(name, "raster_discard")) rasterizer.rasterizer_discard = 1;
   if (is_case(name, "raster_fill")) rasterizer.fill_front = PIPE_POLYGON_MODE_LINE;
   if (is_case(name, "raster_offset")) rasterizer.offset_tri = 1;
   if (is_case(name, "raster_multisample")) rasterizer.multisample = 1;
   if (is_case(name, "raster_clip")) rasterizer.clip_plane_enable = 1;
   if (is_case(name, "raster_halfz")) rasterizer.clip_halfz = 1;
   if (is_case(name, "raster_edge")) rasterizer.bottom_edge_rule = 1;
   if (is_case(name, "raster_center")) rasterizer.half_pixel_center = 0;
   if (is_case(name, "raster_clamp")) rasterizer.depth_clamp = 1;
   state = pipe->create_rasterizer_state(pipe, &rasterizer);
   assert(state);
   pipe->bind_rasterizer_state(pipe, state);
   ((struct pipe_rasterizer_state *)state)->scissor = 1;
   pipe->delete_rasterizer_state(pipe, state);
   assert(!memcmp(&rasterizer, &mgc_context(pipe)->rasterizer_state, sizeof(rasterizer)));
   struct pipe_depth_stencil_alpha_state depth = {0};
   if (!strncmp(name, "clear_z_", 8)) depth.depth_writemask = 1;
   if (!strncmp(name, "z_", 2)) {
      depth.depth_enabled = 1;
      depth.depth_writemask = 1;
      depth.depth_func = PIPE_FUNC_LESS;
   }
   if (is_case(name, "z_disabled")) depth.depth_enabled = 0;
   if (is_case(name, "z_write_disabled")) depth.depth_writemask = 0;
   if (is_case(name, "clear_z_write_disabled")) depth.depth_writemask = 0;
   if (is_case(name, "z_equal")) depth.depth_func = PIPE_FUNC_EQUAL;
   if (is_case(name, "z_lequal")) depth.depth_func = PIPE_FUNC_LEQUAL;
   if (is_case(name, "z_bounds")) depth.depth_bounds_test = 1;
   if (is_case(name, "z_stencil")) depth.stencil[0].enabled = 1;
   if (is_case(name, "depth_enabled")) depth.depth_enabled = 1;
   if (is_case(name, "depth_bounds")) depth.depth_bounds_test = 1;
   if (is_case(name, "stencil_front")) depth.stencil[0].enabled = 1;
   if (is_case(name, "stencil_back")) depth.stencil[1].enabled = 1;
   if (is_case(name, "alpha_test")) depth.alpha_enabled = 1;
   state = pipe->create_depth_stencil_alpha_state(pipe, &depth);
   assert(state);
   pipe->bind_depth_stencil_alpha_state(pipe, state);
   ((struct pipe_depth_stencil_alpha_state *)state)->depth_enabled = 1;
   pipe->delete_depth_stencil_alpha_state(pipe, state);
   assert(!memcmp(&depth, &mgc_context(pipe)->depth_stencil_state, sizeof(depth)));
   struct pipe_viewport_state viewport = {
      .scale = {width * 0.5f, height * 0.5f, 0.5f},
      .translate = {width * 0.5f, height * 0.5f, 0.5f},
      .swizzle_x = PIPE_VIEWPORT_SWIZZLE_POSITIVE_X,
      .swizzle_y = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Y,
      .swizzle_z = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Z,
      .swizzle_w = PIPE_VIEWPORT_SWIZZLE_POSITIVE_W,
   };
   if (is_case(name, "viewport_partial")) viewport.scale[0] -= 0.5;
   if (is_case(name, "viewport_flip")) viewport.scale[1] *= -1;
   if (is_case(name, "viewport_nan")) viewport.scale[0] = NAN;
   if (is_case(name, "viewport_swizzle")) viewport.swizzle_y = PIPE_VIEWPORT_SWIZZLE_POSITIVE_X;
   if (is_case(name, "viewport_depth")) viewport.translate[2] = 0;
   pipe->set_viewport_states(pipe, 0, 1, &viewport);
   memset(&viewport, 0, sizeof(viewport));
   if (is_case(name, "null_blend")) pipe->bind_blend_state(pipe, NULL);
   if (is_case(name, "null_raster")) pipe->bind_rasterizer_state(pipe, NULL);
   if (is_case(name, "null_depth")) pipe->bind_depth_stencil_alpha_state(pipe, NULL);
   if (is_case(name, "missing_viewport")) mgc_context(pipe)->viewport_valid = false;
   if (is_case(name, "sample_mask")) pipe->set_sample_mask(pipe, 0);
}

int
main(int argc, char **argv)
{
   assert(argc == 2);
   const char *name = argv[1];
   bool has_depth = !strncmp(name, "z_", 2) || !strncmp(name, "clear_z_", 8) ||
      is_case(name, "default_z_clear") || is_case(name, "custom_z_clear");
   glsl_type_singleton_init_or_ref();
   struct pipe_screen *screen = mgc_create_screen(NULL);
   assert(screen);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   assert(pipe);
   struct mgc_context *ctx = mgc_context(pipe);
   bind_shader(pipe, false, false, false);
   bind_shader(pipe, true, false, false);
   bind_vertices(pipe);
   if (is_case(name, "write_timeout")) {
      struct pipe_resource *vertices = pipe_buffer_create(screen, PIPE_BIND_VERTEX_BUFFER,
         PIPE_USAGE_DEFAULT, MGC_HOST_CAPTURE_MAX_BUFFER_BYTES);
      assert(vertices);
      memcpy(mgc_resource(vertices)->data, mgc_resource(ctx->vertex_buffers[0].buffer.resource)->data, 48);
      struct pipe_vertex_buffer buffer = {.buffer.resource = vertices};
      pipe->set_vertex_buffers(pipe, 1, &buffer);
   }
   struct pipe_resource template = {
      .target = PIPE_TEXTURE_2D, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 3, .height0 = 2, .depth0 = 1, .array_size = 1,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
      .usage = PIPE_USAGE_DEFAULT,
   };
   if (is_case(name, "small") || is_case(name, "z_small")) template.width0 = template.height0 = 1;
   if (is_case(name, "maximum") || is_case(name, "z_maximum")) template.width0 = template.height0 = 128;
   if (is_case(name, "rt_large")) template.width0 = 129;
   if (is_case(name, "padded") || is_case(name, "z_padded")) {
      template.width0 = 8;
      template.height0 = 4;
   }
   if (is_case(name, "rt_format")) template.format = PIPE_FORMAT_B8G8R8A8_UNORM;
   if (is_case(name, "rt_rectangle")) template.target = PIPE_TEXTURE_RECT;
   if (is_case(name, "rt_array")) { template.target = PIPE_TEXTURE_2D_ARRAY; template.array_size = 2; }
   if (is_case(name, "rt_mips")) template.last_level = 1;
   struct pipe_resource *resource = screen->resource_create(screen, &template);
   assert(resource);
   struct mgc_resource *storage = mgc_resource(resource);
   for (size_t index = 0; index < storage->size; index++)
      ((uint8_t *)storage->data)[index] = (index * 17 + 3) & 255;
   struct pipe_resource saved_resource = *resource;
   struct mgc_resource_level saved_level = storage->levels[0];
   size_t saved_size = storage->size;
   void *saved_data = storage->data;
   if (is_case(name, "padded") || is_case(name, "z_padded")) {
      resource->width0 = storage->levels[0].width = 3;
      resource->height0 = storage->levels[0].height = 2;
      storage->levels[0].offset = 4;
   }
   struct pipe_framebuffer_state framebuffer = {
      .width = resource->width0, .height = resource->height0, .nr_cbufs = 1,
      .cbufs[0] = {.texture = resource, .format = resource->format},
   };
   struct pipe_resource *depth_resource = NULL;
   struct mgc_resource *depth_storage = NULL;
   struct pipe_resource saved_depth = {0};
   struct mgc_resource_level saved_depth_level = {0};
   size_t saved_depth_size = 0;
   void *saved_depth_data = NULL;
   if (has_depth && !is_case(name, "z_missing")) {
      struct pipe_resource depth_template = template;
      depth_template.format = PIPE_FORMAT_Z16_UNORM;
      depth_template.bind = PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW;
      depth_resource = screen->resource_create(screen, &depth_template);
      assert(depth_resource);
      depth_storage = mgc_resource(depth_resource);
      for (size_t index = 0; index < depth_storage->size; index++)
         ((uint8_t *)depth_storage->data)[index] = (index * 23 + 5) & 255;
      saved_depth = *depth_resource;
      saved_depth_level = depth_storage->levels[0];
      saved_depth_size = depth_storage->size;
      saved_depth_data = depth_storage->data;
      if (is_case(name, "z_padded")) {
         depth_resource->width0 = depth_storage->levels[0].width = 3;
         depth_resource->height0 = depth_storage->levels[0].height = 2;
         depth_storage->levels[0].offset = 2;
      }
      framebuffer.zsbuf.texture = depth_resource;
      framebuffer.zsbuf.format = PIPE_FORMAT_Z16_UNORM;
   }
   if (is_case(name, "rt_depth_attachment") || is_case(name, "clear_depth_attachment"))
      framebuffer.zsbuf = framebuffer.cbufs[0];
   pipe->set_framebuffer_state(pipe, &framebuffer);
   assert(!ctx->status.error);
   unsigned width = framebuffer.width, height = framebuffer.height;
   bind_draw_state(pipe, name, width, height);
   if (depth_resource) {
      if (is_case(name, "z_format") || is_case(name, "clear_z_format"))
         depth_resource->format = PIPE_FORMAT_Z32_FLOAT;
      if (is_case(name, "z_surface_format")) ctx->framebuffer.zsbuf.format = PIPE_FORMAT_Z32_FLOAT;
      if (is_case(name, "z_size")) depth_resource->width0++;
      if (is_case(name, "z_large")) depth_resource->width0 = 129;
      if (is_case(name, "z_array")) depth_resource->array_size = 2;
      if (is_case(name, "z_mips")) depth_resource->last_level = 1;
      if (is_case(name, "z_samples")) depth_resource->nr_samples = 2;
      if (is_case(name, "z_storage_samples")) depth_resource->nr_storage_samples = 2;
      if (is_case(name, "z_shared")) depth_resource->bind |= PIPE_BIND_SHARED;
      if (is_case(name, "z_persistent")) depth_resource->flags |= PIPE_RESOURCE_FLAG_MAP_PERSISTENT;
      if (is_case(name, "z_stride")) depth_storage->levels[0].stride = 1;
      if (is_case(name, "z_offset")) depth_storage->levels[0].offset = saved_depth_size;
      if (is_case(name, "z_null_data")) depth_storage->data = NULL;
      if (is_case(name, "z_allocation_size")) depth_storage->size = 1;
      if (is_case(name, "z_pinned") || is_case(name, "clear_z_pinned"))
         depth_storage->active_host_executions = 1;
      if (is_case(name, "z_alias")) {
         struct pipe_sampler_view *view = create_view(pipe, depth_resource);
         pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 0, &view);
         pipe_sampler_view_reference(&view, NULL);
      }
   }
   if (is_case(name, "clear_z_color_pinned")) storage->active_host_executions = 1;
   if (is_case(name, "rt_shared") || is_case(name, "clear_shared")) resource->bind |= PIPE_BIND_SHARED;
   if (is_case(name, "rt_samples")) resource->nr_samples = 2;
   if (is_case(name, "rt_storage_samples")) resource->nr_storage_samples = 2;
   if (is_case(name, "rt_persistent")) resource->flags |= PIPE_RESOURCE_FLAG_MAP_PERSISTENT;
   if (is_case(name, "rt_stride")) storage->levels[0].stride = 1;
   if (is_case(name, "rt_offset")) storage->levels[0].offset = saved_size;
   if (is_case(name, "rt_null_data")) storage->data = NULL;
   if (is_case(name, "rt_size")) storage->size = 1;
   if (is_case(name, "rt_alias") || is_case(name, "clear_alias")) {
      struct pipe_sampler_view *view = create_view(pipe, resource);
      pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, 1, 0, &view);
      pipe_sampler_view_reference(&view, NULL);
   }
   struct pipe_transfer *transfer = NULL;
   if (is_case(name, "rt_mapped") || is_case(name, "clear_mapped") ||
       is_case(name, "clear_z_color_mapped")) {
      struct pipe_box box = {.width = 3, .height = 2, .depth = 1};
      assert(pipe->texture_map(pipe, resource, 0, PIPE_MAP_READ, &box, &transfer));
   }
   struct pipe_transfer *depth_transfer = NULL;
   if (is_case(name, "z_mapped") || is_case(name, "clear_z_mapped")) {
      struct pipe_box box = {.width = 3, .height = 2, .depth = 1};
      assert(pipe->texture_map(pipe, depth_resource, 0, PIPE_MAP_READ, &box, &depth_transfer));
   }
   unsigned callbacks = 0;
   if (is_case(name, "custom") || is_case(name, "custom_clear") || is_case(name, "custom_z_clear"))
      mgc_context_set_artifact_callback(pipe, custom_callback, &callbacks);
   if (is_case(name, "capture_only")) {
      ctx->rasterizer_state.scissor = 1;
      ctx->blend_state.rt[0].colormask = 0;
      ctx->viewport_valid = false;
   }
   if (is_case(name, "hardware_target"))
      ctx->shaders[1]->shader->abi.target = MGC_TARGET_FP16_HARDWARE_V1;
   if (is_case(name, "abi3_fs_abi"))
      ctx->shaders[1]->shader->abi_version = MGC_ABI_VERSION;
   if (is_case(name, "abi3_fs_option"))
      ctx->shaders[1]->shader->abi.host_instancing_version = 0;
   if (is_case(name, "abi3_fs_hardware"))
      ctx->shaders[1]->shader->abi.target = MGC_TARGET_FP16_HARDWARE_V1;
   if (is_case(name, "abi3_without_optin")) {
      for (unsigned stage = 0; stage < MGC_RUNTIME_STAGES; stage++) {
         ctx->shaders[stage]->shader->abi_version = MGC_HOST_INSTANCING_ABI_VERSION;
         ctx->shaders[stage]->shader->abi.host_instancing_version = MGC_HOST_INSTANCING_VERSION;
      }
   }
   pthread_t worker;
   struct pin_probe probe = {0};
   bool concurrent = is_case(name, "pin_success") || is_case(name, "pin_error") ||
      !strncmp(name, "z_pin_", 6);
   if (concurrent) {
      probe.pipe = screen->context_create(screen, NULL, 0);
      probe.resource = resource;
      probe.depth = depth_resource;
      probe.owner = ctx;
      probe.name = name;
      assert(probe.pipe);
      probe.pipe->set_framebuffer_state(probe.pipe, &framebuffer);
      bind_draw_state(probe.pipe, name, width, height);
      assert(!pthread_create(&worker, NULL, probe_pin, &probe));
   }
   struct pipe_resource *borrowed = resource;
   struct pipe_resource *borrowed_depth = depth_resource;
   pipe_resource_reference(&resource, NULL);
   pipe_resource_reference(&depth_resource, NULL);
   if (!strncmp(name, "clear_", 6) || is_case(name, "default_clear") || is_case(name, "custom_clear") ||
       is_case(name, "default_z_clear") || is_case(name, "custom_z_clear")) {
      union pipe_color_union color = {.f = {-1, 0.5, 2, 1}};
      if (is_case(name, "clear_zero")) memset(&color, 0, sizeof(color));
      if (is_case(name, "clear_nan") || is_case(name, "clear_z_color_nan")) color.f[0] = NAN;
      if (is_case(name, "clear_unbound") || is_case(name, "clear_z_unbound")) {
         pipe->bind_blend_state(pipe, NULL);
         pipe->bind_rasterizer_state(pipe, NULL);
         pipe->bind_depth_stencil_alpha_state(pipe, NULL);
      }
      struct pipe_scissor_state scissor = {.maxx = width, .maxy = height};
      unsigned buffers = PIPE_CLEAR_COLOR0;
      if (has_depth && !is_case(name, "clear_z_coloronly")) buffers |= PIPE_CLEAR_DEPTH;
      if (is_case(name, "clear_z_depthonly") || is_case(name, "clear_z_nullcolor")) buffers = PIPE_CLEAR_DEPTH;
      if (is_case(name, "clear_depth")) buffers |= PIPE_CLEAR_DEPTH;
      if (is_case(name, "clear_stencil") || is_case(name, "clear_z_stencil")) buffers |= PIPE_CLEAR_STENCIL;
      if (is_case(name, "clear_other_color")) buffers = PIPE_CLEAR_COLOR1;
      if (is_case(name, "clear_empty")) buffers = 0;
      double clear_depth = 1;
      if (is_case(name, "clear_z_half")) clear_depth = 0.5;
      if (is_case(name, "clear_z_zero")) clear_depth = 0;
      if (is_case(name, "clear_z_nan")) clear_depth = NAN;
      if (is_case(name, "clear_z_inf")) clear_depth = INFINITY;
      if (is_case(name, "clear_z_below")) clear_depth = -0.01;
      if (is_case(name, "clear_z_above")) clear_depth = 1.01;
      pipe->clear(pipe, buffers, is_case(name, "clear_scissor") || is_case(name, "clear_z_scissor") ? &scissor : NULL,
                  is_case(name, "clear_null") || is_case(name, "clear_z_nullcolor") ? NULL : &color, clear_depth, 0);
   } else if (!strncmp(name, "unit_", 5)) {
      ctx->status.sequence = mgc_next_sequence(screen);
      uint64_t sequence = is_case(name, "unit_zero_sequence") ? 0 :
         is_case(name, "unit_wrong_sequence") ? ctx->status.sequence + 1 : ctx->status.sequence;
      size_t json_size = is_case(name, "unit_json_empty") ? 0 :
         is_case(name, "unit_json_large") ? MGC_HOST_CAPTURE_MAX_FILE_BYTES + 1u : 2;
      assert(!mgc_host_executor_execute(ctx, sequence,
         is_case(name, "unit_json_null") ? NULL : "{}", json_size));
   } else {
      struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
      struct pipe_draw_start_count_bias draw = {0, 3, 0};
      pipe->draw_vbo(pipe, &info, 7, NULL, &draw, 1);
   }
   struct mgc_runtime_status status;
   mgc_context_get_runtime_status(pipe, &status);
   if (concurrent) {
      assert(!pthread_join(worker, NULL));
      probe.pipe->destroy(probe.pipe);
   }
   if (is_case(name, "clear_z_color_pinned")) {
      assert(storage->active_host_executions == 1);
      storage->active_host_executions = 0;
   }
   assert(!storage->active_host_executions);
   if (depth_storage) {
      if (is_case(name, "z_pinned") || is_case(name, "clear_z_pinned")) {
         assert(depth_storage->active_host_executions == 1);
         depth_storage->active_host_executions = 0;
      }
      assert(!depth_storage->active_host_executions);
   }
   if (transfer) pipe->texture_unmap(pipe, transfer);
   if (depth_transfer) pipe->texture_unmap(pipe, depth_transfer);
   borrowed->width0 = saved_resource.width0;
   borrowed->height0 = saved_resource.height0;
   borrowed->bind = saved_resource.bind;
   borrowed->flags = saved_resource.flags;
   borrowed->nr_samples = saved_resource.nr_samples;
   borrowed->nr_storage_samples = saved_resource.nr_storage_samples;
   storage->levels[0] = saved_level;
   storage->size = saved_size;
   storage->data = saved_data;
   if (borrowed_depth) {
      borrowed_depth->format = saved_depth.format;
      borrowed_depth->width0 = saved_depth.width0;
      borrowed_depth->height0 = saved_depth.height0;
      borrowed_depth->array_size = saved_depth.array_size;
      borrowed_depth->last_level = saved_depth.last_level;
      borrowed_depth->bind = saved_depth.bind;
      borrowed_depth->flags = saved_depth.flags;
      borrowed_depth->nr_samples = saved_depth.nr_samples;
      borrowed_depth->nr_storage_samples = saved_depth.nr_storage_samples;
      depth_storage->levels[0] = saved_depth_level;
      depth_storage->size = saved_depth_size;
      depth_storage->data = saved_depth_data;
   }
   struct pipe_fence_handle *fence = NULL;
   pipe->flush(pipe, &fence, 0);
   uint64_t fence_sequence;
   int fence_error;
   assert(fence && mgc_fence_get_status(screen, fence, &fence_sequence, &fence_error));
   assert(fence_error == status.error && fence_sequence >= status.sequence);
   screen->fence_reference(screen, &fence, NULL);
   printf("{\"accepted\":%" PRIu64 ",\"rejected\":%" PRIu64 ",\"sequence\":%" PRIu64
          ",\"error\":%d,\"callbacks\":%u,\"pin_rejections\":%u,\"pin_recovered\":%s,\"storage_hex\":\"",
          status.accepted_draws, status.rejected_draws, status.sequence,
          status.error, callbacks, probe.rejected, probe.recovered ? "true" : "false");
   for (size_t index = 0; index < saved_size; index++) printf("%02x", ((uint8_t *)saved_data)[index]);
   printf("\",\"depth_hex\":\"");
   for (size_t index = 0; index < saved_depth_size; index++) printf("%02x", ((uint8_t *)saved_depth_data)[index]);
   puts("\"}");
   struct pipe_box box = {
      .width = saved_resource.width0, .height = saved_resource.height0, .depth = 1,
   };
   transfer = NULL;
   assert(pipe->texture_map(pipe, borrowed, 0, PIPE_MAP_READ, &box, &transfer));
   assert(transfer);
   pipe->texture_unmap(pipe, transfer);
   assert(!ctx->transfers && !storage->active_maps && !storage->active_host_executions);
   if (borrowed_depth) {
      box.width = saved_depth.width0;
      box.height = saved_depth.height0;
      assert(pipe->texture_map(pipe, borrowed_depth, 0, PIPE_MAP_READ, &box, &transfer));
      assert(transfer);
      pipe->texture_unmap(pipe, transfer);
      assert(!ctx->transfers && !depth_storage->active_maps && !depth_storage->active_host_executions);
   }
   pipe->destroy(pipe);
   assert(mgc_screen(screen)->allocated_bytes == 0);
   screen->destroy(screen);
   glsl_type_singleton_decref();
   return 0;
}
