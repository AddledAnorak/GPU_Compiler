#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "mgc_compiler.h"

static nir_def *
input(nir_builder *builder, unsigned component)
{
   return nir_load_input(builder, 1, 32, nir_imm_int(builder, 0),
                         .base = 0, .component = component,
                         .dest_type = nir_type_float32,
                         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
}

static void
output(nir_builder *builder, nir_def *value, unsigned component)
{
   nir_store_output(builder, value, nir_imm_int(builder, 0),
                    .base = 0, .component = component, .write_mask = 1,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_COLOR, .num_slots = 1});
}

static nir_def *
chain(nir_builder *builder, nir_def *value, nir_def *step, unsigned count, bool subtract)
{
   for (unsigned iteration = 0; iteration < count; iteration++) {
      value = subtract ? nir_fsub(builder, value, step) : nir_fadd(builder, value, step);
      output(builder, value, 0);
   }
   return value;
}

static void
sequential(nir_builder *builder, unsigned count, bool exclusive)
{
   nir_def *seed = input(builder, 0);
   nir_def *step = input(builder, 1);
   nir_if *branch = nir_push_if(builder, nir_flt_imm(builder, seed, 0));
   chain(builder, seed, step, count, exclusive);
   nir_push_else(builder, branch);
   if (exclusive)
      chain(builder, seed, step, count, false);
   else
      output(builder, seed, 0);
   nir_pop_if(builder, branch);
}

static void
predicate_chain(nir_builder *builder, unsigned count)
{
   nir_def *seed = input(builder, 0);
   nir_def *step = input(builder, 1);
   nir_if *branch = nir_push_if(builder, nir_flt_imm(builder, seed, 0));
   nir_def *value = seed;
   for (unsigned iteration = 0; iteration < count; iteration++) {
      nir_def *comparison = nir_flt_imm(builder, value, 0);
      nir_def *inverse = nir_inot(builder, comparison);
      nir_def *selected = nir_bcsel(builder, inverse, nir_fneg(builder, step), step);
      value = nir_fadd(builder, value, selected);
      output(builder, value, 0);
      output(builder, nir_bcsel(builder, comparison, step, nir_fneg(builder, step)), 1);
   }
   nir_push_else(builder, branch);
   output(builder, seed, 0);
   output(builder, step, 1);
   nir_pop_if(builder, branch);
}

static void
rotate_loop(nir_builder *builder, bool three_values)
{
   nir_def *bound = input(builder, 3);
   nir_variable *values[3];
   unsigned width = three_values ? 3 : 2;
   for (unsigned component = 0; component < width; component++) {
      values[component] = nir_local_variable_create(builder->impl, glsl_float_type(), "carried");
      nir_store_var(builder, values[component], input(builder, component), 1);
   }
   nir_variable *index = nir_local_variable_create(builder->impl, glsl_int_type(), "index");
   nir_store_var(builder, index, nir_imm_int(builder, 0), 1);
   nir_loop *loop = nir_push_loop(builder);
   nir_def *current = nir_load_var(builder, index);
   nir_if *exit = nir_push_if(builder, nir_fge(builder, nir_i2f32(builder, current), bound));
   nir_jump(builder, nir_jump_break);
   nir_pop_if(builder, exit);
   nir_def *previous[3];
   for (unsigned component = 0; component < width; component++)
      previous[component] = nir_load_var(builder, values[component]);
   for (unsigned component = 0; component < width; component++)
      nir_store_var(builder, values[component], previous[(component + 1) % width], 1);
   nir_store_var(builder, index, nir_iadd_imm(builder, current, 1), 1);
   nir_pop_loop(builder, loop);
   for (unsigned component = 0; component < width; component++)
      output(builder, nir_load_var(builder, values[component]), component);
}

static void
predicate_loop(nir_builder *builder)
{
   nir_def *first = input(builder, 0);
   nir_def *second = input(builder, 1);
   nir_def *bound = input(builder, 3);
   nir_variable *flag = nir_local_variable_create(builder->impl, glsl_bool_type(), "flag");
   nir_variable *sum = nir_local_variable_create(builder->impl, glsl_float_type(), "sum");
   nir_variable *index = nir_local_variable_create(builder->impl, glsl_int_type(), "index");
   nir_store_var(builder, flag, nir_flt(builder, first, second), 1);
   nir_store_var(builder, sum, nir_imm_float(builder, 0), 1);
   nir_store_var(builder, index, nir_imm_int(builder, 0), 1);
   nir_loop *loop = nir_push_loop(builder);
   nir_def *current = nir_load_var(builder, index);
   nir_if *exit = nir_push_if(builder, nir_fge(builder, nir_i2f32(builder, current), bound));
   nir_jump(builder, nir_jump_break);
   nir_pop_if(builder, exit);
   nir_def *previous_flag = nir_load_var(builder, flag);
   nir_def *selected = nir_bcsel(builder, previous_flag, first, second);
   nir_store_var(builder, sum, nir_fadd(builder, nir_load_var(builder, sum), selected), 1);
   nir_store_var(builder, flag, nir_inot(builder, previous_flag), 1);
   nir_store_var(builder, index, nir_iadd_imm(builder, current, 1), 1);
   nir_pop_loop(builder, loop);
   output(builder, nir_load_var(builder, sum), 0);
   output(builder, nir_bcsel(builder, nir_load_var(builder, flag), first, second), 1);
}

static void
pressure(nir_builder *builder, unsigned count, bool predicates)
{
   nir_def **values = calloc(count, sizeof(*values));
   nir_def *seed = input(builder, 0);
   nir_def *condition = nir_flt_imm(builder, seed, 0);
   for (unsigned index = 0; index < count; index++) {
      values[index] = predicates ? nir_flt_imm(builder, seed, (float)index) :
         nir_load_ubo(builder, 1, 32, nir_imm_int(builder, 0), nir_imm_int(builder, index * 4),
                      .align_mul = 4, .range = 4);
   }
   nir_if *branch = nir_push_if(builder, condition);
   for (unsigned index = 0; index < count; index++)
      output(builder, predicates ? nir_bcsel(builder, values[index], seed, nir_fneg(builder, seed)) : values[index], 0);
   nir_push_else(builder, branch);
   output(builder, seed, 0);
   nir_pop_if(builder, branch);
   free(values);
}

int
main(int argc, char **argv)
{
   if (argc != 3)
      return 2;
   unsigned count = strtoul(argv[2], NULL, 10);
   if (!count || count > 4097)
      return 2;
   nir_builder builder = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, mgc_test_get_compiler_options(), "cfg_allocation");
   nir_variable *varying = nir_variable_create(builder.shader, nir_var_shader_in,
                                               glsl_vec4_type(), "input");
   varying->data.location = VARYING_SLOT_VAR0;
   varying->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color = nir_variable_create(builder.shader, nir_var_shader_out,
                                             glsl_vec4_type(), "output");
   color->data.location = FRAG_RESULT_COLOR;
   if (!strcmp(argv[1], "sequential") || !strcmp(argv[1], "exclusive"))
      sequential(&builder, count, !strcmp(argv[1], "exclusive"));
   else if (!strcmp(argv[1], "predicate-chain"))
      predicate_chain(&builder, count);
   else if (!strcmp(argv[1], "swap-loop") || !strcmp(argv[1], "rotate-loop"))
      rotate_loop(&builder, !strcmp(argv[1], "rotate-loop"));
   else if (!strcmp(argv[1], "predicate-loop"))
      predicate_loop(&builder);
   else if (!strcmp(argv[1], "scalar-pressure") || !strcmp(argv[1], "predicate-pressure"))
      pressure(&builder, count, !strcmp(argv[1], "predicate-pressure"));
   else
      return 2;
   NIR_PASS_V(builder.shader, nir_lower_vars_to_ssa);
   unsigned definitions = 0;
   nir_foreach_block(block, builder.impl) {
      nir_foreach_instr(instr, block) {
         if (nir_instr_def(instr) && instr->type != nir_instr_type_load_const)
            definitions++;
      }
   }
   struct mgc_compile_options options = mgc_default_compile_options();
   options.fragment_status_version = MGC_FRAGMENT_STATUS_VERSION;
   struct mgc_shader *shader = mgc_compile_nir_with_options(builder.shader, &options);
   fprintf(stderr, "stats={\"definitions\":%u,\"registers\":%u,\"predicates\":%u,\"blocks\":%u}\n",
           definitions, shader->stats.registers, shader->stats.predicates, shader->stats.blocks);
   if (shader->layout_json)
      fprintf(stderr, "layout=%s\n", shader->layout_json);
   bool failed = shader->error != NULL;
   if (failed)
      fprintf(stderr, "%s\n", shader->error);
   else
      fputs(shader->asm_text, stdout);
   mgc_shader_destroy(shader);
   return failed;
}
