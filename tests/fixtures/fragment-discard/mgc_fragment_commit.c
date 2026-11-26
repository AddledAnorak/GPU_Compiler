#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "mgc_compiler.h"

_Static_assert(sizeof(((struct mgc_compile_options *)0)->fragment_status_version) == sizeof(uint32_t),
               "fragment status option must be uint32_t");
_Static_assert(offsetof(struct mgc_compile_options, fragment_status_version) >=
               offsetof(struct mgc_compile_options, target) + sizeof(enum mgc_compile_target),
               "fragment status option must be appended after target");

static nir_def *
input(nir_builder *builder, unsigned component)
{
   return nir_load_input(builder, 1, 32, nir_imm_int(builder, 0),
                         .base = 0, .component = component, .dest_type = nir_type_float32,
                         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
}

static void
output(nir_builder *builder, nir_def *value, unsigned base, unsigned component)
{
   nir_store_output(builder, value, nir_imm_int(builder, 0),
                    .base = base, .component = component, .write_mask = 1,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = builder->shader->info.stage == MESA_SHADER_VERTEX ?
                                       VARYING_SLOT_POS : FRAG_RESULT_COLOR,
                                     .num_slots = 1});
}

static void
rgba(nir_builder *builder, unsigned color, bool zero)
{
   for (unsigned component = 0; component < 4; component++)
      output(builder, nir_imm_float(builder, !zero && component == color ? 1.0f : 0.0f),
             0, component);
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "defaults")) {
      struct mgc_compile_options legacy = mgc_default_compile_options();
      struct mgc_compile_options hardware = mgc_default_hardware_compile_options();
      printf("{\"legacy\":%u,\"hardware\":%u}\n",
             legacy.fragment_status_version, hardware.fragment_status_version);
      return 0;
   }
   if (argc != 4) {
      fprintf(stderr, "usage: %s <mode> <default|version> <legacy|hardware>\n", argv[0]);
      return 2;
   }
   const char *mode = argv[1];
   bool vertex = !strcmp(mode, "vertex");
   nir_builder builder = nir_builder_init_simple_shader(
      vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT,
      mgc_test_get_compiler_options(), "mgc_fragment_commit");
   nir_variable *color_output = nir_variable_create(
      builder.shader, nir_var_shader_out, glsl_vec4_type(), "commit_output");
   color_output->data.location = vertex ? VARYING_SLOT_POS : FRAG_RESULT_COLOR;

   if (!strcmp(mode, "zero") || vertex) {
      rgba(&builder, 0, true);
   } else if (!strcmp(mode, "collision") || !strcmp(mode, "boundary")) {
      output(&builder, nir_imm_float(&builder, 1), 16383,
             !strcmp(mode, "collision") ? 3 : 2);
   } else {
      nir_variable *varying = nir_variable_create(
         builder.shader, nir_var_shader_in, glsl_vec4_type(), "commit_input");
      varying->data.location = VARYING_SLOT_VAR0;
      varying->data.interpolation = INTERP_MODE_SMOOTH;
      nir_def *coordinate = input(&builder, 0);
      nir_def *condition = nir_flt_imm(&builder, coordinate, 0);

      if (!strcmp(mode, "early-demote-if") || !strcmp(mode, "late-demote-if") ||
          !strcmp(mode, "early-terminate-if") || !strcmp(mode, "late-terminate-if")) {
         if (!strncmp(mode, "late-", 5))
            rgba(&builder, 0, false);
         if (strstr(mode, "terminate"))
            nir_terminate_if(&builder, condition);
         else
            nir_demote_if(&builder, condition);
         rgba(&builder, 1, false);
      } else if (!strcmp(mode, "late-cfg")) {
         nir_def *other_condition = nir_flt_imm(&builder, input(&builder, 1), 0);
         rgba(&builder, 0, false);
         nir_if *branch = nir_push_if(&builder, condition);
         nir_demote_if(&builder, other_condition);
         rgba(&builder, 1, false);
         nir_push_else(&builder, branch);
         rgba(&builder, 2, false);
         nir_pop_if(&builder, branch);
      } else if (!strcmp(mode, "returns")) {
         nir_if *branch = nir_push_if(&builder, condition);
         rgba(&builder, 0, false);
         nir_jump(&builder, nir_jump_return);
         nir_push_else(&builder, branch);
         rgba(&builder, 2, false);
         nir_jump(&builder, nir_jump_return);
         nir_pop_if(&builder, branch);
      } else if (!strcmp(mode, "loop")) {
         nir_def *other_condition = nir_flt_imm(&builder, input(&builder, 1), 0);
         nir_loop *loop = nir_push_loop(&builder);
         nir_if *exit_if = nir_push_if(&builder, nir_fge_imm(&builder, coordinate, 0));
         nir_jump(&builder, nir_jump_break);
         nir_pop_if(&builder, exit_if);
         nir_demote_if(&builder, other_condition);
         nir_pop_loop(&builder, loop);
         rgba(&builder, 0, false);
      } else if (!strcmp(mode, "demote") || !strcmp(mode, "terminate")) {
         if (!strcmp(mode, "demote"))
            nir_demote(&builder);
         else
            nir_terminate(&builder);
         rgba(&builder, 0, false);
      } else {
         fprintf(stderr, "unknown mode: %s\n", mode);
         ralloc_free(builder.shader);
         return 2;
      }
   }

   struct mgc_compile_options options = !strcmp(argv[3], "hardware") ?
      mgc_default_hardware_compile_options() : mgc_default_compile_options();
   if (strcmp(argv[2], "default"))
      options.fragment_status_version = strtoul(argv[2], NULL, 10);
   nir_print_shader(builder.shader, stderr);
   struct mgc_shader *shader = mgc_compile_nir_with_options(builder.shader, &options);
   if (shader->layout_json)
      fprintf(stderr, "layout=%s\n", shader->layout_json);
   int failed = shader->error != NULL;
   if (failed)
      fprintf(stderr, "%s\n", shader->error);
   else
      fputs(shader->asm_text, stdout);
   mgc_shader_destroy(shader);
   return failed;
}
