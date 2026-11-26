#include "mgc_host_executor.h"

static void
configure_native_blend_equation(struct pipe_context *pipe)
{
   assert(MGC_HOST_BLEND_DISABLED == 0 && MGC_HOST_BLEND_SOURCE_OVER == 1);
   assert(MGC_HOST_BLEND_ADDITIVE == 2 && MGC_HOST_BLEND_SUBTRACT == 3);
   assert(MGC_HOST_BLEND_REVERSE_SUBTRACT == 4 && MGC_HOST_BLEND_MINIMUM == 5);
   assert(MGC_HOST_BLEND_MAXIMUM == 6);
   const char *option = getenv("MGC_TEST_BLEND_MODE");
   assert(option && strlen(option) == 1 && option[0] >= '0' && option[0] <= '6');
   unsigned mode = option[0] - '0';
   const unsigned equations[] = {
      PIPE_BLEND_ADD, PIPE_BLEND_ADD, PIPE_BLEND_ADD, PIPE_BLEND_SUBTRACT,
      PIPE_BLEND_REVERSE_SUBTRACT, PIPE_BLEND_MIN, PIPE_BLEND_MAX,
   };
   struct pipe_blend_state blend = {.rt[0].colormask = 15};
   struct pipe_rt_blend_state *target = &blend.rt[0];
   target->blend_enable = mode != MGC_HOST_BLEND_DISABLED;
   target->rgb_func = target->alpha_func = equations[mode];
   target->rgb_src_factor = target->rgb_dst_factor = PIPE_BLENDFACTOR_ONE;
   target->alpha_src_factor = target->alpha_dst_factor = PIPE_BLENDFACTOR_ONE;
   if (mode == MGC_HOST_BLEND_SOURCE_OVER) {
      target->rgb_src_factor = PIPE_BLENDFACTOR_SRC_ALPHA;
      target->rgb_dst_factor = target->alpha_dst_factor = PIPE_BLENDFACTOR_INV_SRC_ALPHA;
   }
   const char *mutation = getenv("MGC_TEST_BLEND_MUTATION");
   assert(mutation);
   bool known = is_case(mutation, "none");
   const char *fields[] = {"rgb_src", "rgb_dst", "alpha_src", "alpha_dst"};
   const char *factor_names[] = {"zero", "src_alpha", "inv_src_alpha"};
   const unsigned factors[] = {
      PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_SRC_ALPHA, PIPE_BLENDFACTOR_INV_SRC_ALPHA,
   };
   for (unsigned field = 0; field < ARRAY_SIZE(fields); field++) {
      for (unsigned factor = 0; factor < ARRAY_SIZE(factors); factor++) {
         char expected[64];
         snprintf(expected, sizeof(expected), "factor_%s_%s", fields[field], factor_names[factor]);
         if (!is_case(mutation, expected))
            continue;
         if (field == 0) target->rgb_src_factor = factors[factor];
         if (field == 1) target->rgb_dst_factor = factors[factor];
         if (field == 2) target->alpha_src_factor = factors[factor];
         if (field == 3) target->alpha_dst_factor = factors[factor];
         known = true;
      }
   }
   if (is_case(mutation, "mixed_rgb")) {
      target->rgb_func = (equations[mode] + 1) % 5;
      known = true;
   }
   if (is_case(mutation, "mixed_alpha")) {
      target->alpha_func = (equations[mode] + 1) % 5;
      known = true;
   }
   if (is_case(mutation, "unknown_equation_7")) {
      target->rgb_func = target->alpha_func = 7;
      known = true;
   }
   if (is_case(mutation, "color_mask")) { target->colormask = 7; known = true; }
   if (is_case(mutation, "logicop")) { blend.logicop_enable = 1; known = true; }
   if (is_case(mutation, "dither")) { blend.dither = 1; known = true; }
   if (is_case(mutation, "coverage")) { blend.alpha_to_coverage = 1; known = true; }
   if (is_case(mutation, "coverage_dither")) { blend.alpha_to_coverage_dither = 1; known = true; }
   if (is_case(mutation, "alpha_to_one")) { blend.alpha_to_one = 1; known = true; }
   if (is_case(mutation, "independent")) { blend.independent_blend_enable = 1; known = true; }
   if (is_case(mutation, "max_rt")) { blend.max_rt = 1; known = true; }
   if (is_case(mutation, "advanced")) { blend.advanced_blend_func = 1; known = true; }
   if (is_case(mutation, "coherent")) { blend.blend_coherent = 1; known = true; }
   if (is_case(mutation, "extra_target")) { blend.rt[1].blend_enable = 1; known = true; }
   assert(known);
   void *state = pipe->create_blend_state(pipe, &blend);
   assert(state);
   pipe->bind_blend_state(pipe, state);
   ((struct pipe_blend_state *)state)->rt[0].colormask = 0;
   ((struct pipe_blend_state *)state)->rt[0].rgb_func = 7;
   pipe->delete_blend_state(pipe, state);
   assert(mgc_context(pipe)->blend_state_valid);
   assert(!memcmp(&blend, &mgc_context(pipe)->blend_state, sizeof(blend)));
}

static void
change_native_blend_during_ipc(struct mgc_context *owner, const char *name)
{
   struct pipe_rt_blend_state *target = &owner->blend_state.rt[0];
   if (is_case(name, "z_pin_blend_equation"))
      target->rgb_func = target->alpha_func = (target->rgb_func + 1) % 5;
   if (is_case(name, "z_pin_blend_factor"))
      target->rgb_dst_factor = PIPE_BLENDFACTOR_ZERO;
}
