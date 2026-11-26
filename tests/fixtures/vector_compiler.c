#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "mgc_compiler.h"

int
main(int argc, char **argv)
{
   if (argc != 2)
      return 2;
   nir_builder builder = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      mgc_test_get_compiler_options(), "vector-target-tests");
   nir_variable *input_variable = nir_variable_create(builder.shader, nir_var_shader_in, glsl_vec4_type(), "v_input");
   input_variable->data.location = VARYING_SLOT_VAR0;
   input_variable->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *output_variable = nir_variable_create(builder.shader, nir_var_shader_out, glsl_vec4_type(), "color");
   output_variable->data.location = FRAG_RESULT_DATA0;
   output_variable->data.interpolation = INTERP_MODE_SMOOTH;
   nir_def *left = nir_load_input(&builder, 1, 32, nir_imm_int(&builder, 0),
      .base = 0, .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_def *right = nir_load_input(&builder, 1, 32, nir_imm_int(&builder, 0),
      .base = 0, .component = 1, .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_def *value = left;
   if (!strcmp(argv[1], "add"))
      value = nir_fadd(&builder, left, right);
   else if (!strcmp(argv[1], "subtract"))
      value = nir_fsub(&builder, left, right);
   else if (!strcmp(argv[1], "multiply"))
      value = nir_fmul(&builder, left, right);
   else if (!strcmp(argv[1], "minimum"))
      value = nir_fmin(&builder, left, right);
   else if (!strcmp(argv[1], "maximum"))
      value = nir_fmax(&builder, left, right);
   else if (!strcmp(argv[1], "negate"))
      value = nir_fneg(&builder, left);
   else if (!strcmp(argv[1], "absolute"))
      value = nir_fabs(&builder, left);
   else if (!strcmp(argv[1], "rsqrt"))
      value = nir_frsq(&builder, left);
   else if (!strcmp(argv[1], "sqrt"))
      value = nir_fsqrt(&builder, left);
   else if (!strcmp(argv[1], "divide"))
      value = nir_fdiv(&builder, left, right);
   else if (!strcmp(argv[1], "integer"))
      value = nir_iadd(&builder, nir_f2i32(&builder, left), nir_imm_int(&builder, 1));
   else if (!strcmp(argv[1], "fractional-power"))
      value = nir_fpow(&builder, left, nir_imm_float(&builder, 0.7f));
   else if (!strcmp(argv[1], "chain")) {
      for (unsigned iteration = 0; iteration < 240; iteration++)
         value = nir_fadd(&builder, nir_fmul(&builder, value, nir_imm_float(&builder, 0.99951171875f)), right);
   } else if (!strcmp(argv[1], "late-invalid")) {
      nir_store_output(&builder, nir_imm_float(&builder, 0.5f), nir_imm_int(&builder, 0),
         .base = 0, .component = 0, .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      value = nir_frsq(&builder, left);
   } else if (!strcmp(argv[1], "constant-inf"))
      value = nir_imm_float(&builder, __builtin_inff());
   else if (!strcmp(argv[1], "output-reserved")) {
      nir_store_output(&builder, value, nir_imm_int(&builder, 0),
         .base = 8192, .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   } else if (strcmp(argv[1], "copy"))
      return 2;
   nir_store_output(&builder, value, nir_imm_int(&builder, 0),
      .base = 0, .component = !strcmp(argv[1], "late-invalid") ? 1 : 0,
      .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   struct mgc_compile_options options = mgc_default_hardware_compile_options();
   options.target = MGC_TARGET_FP16_HARDWARE_VECTOR;
   struct mgc_shader *shader = mgc_compile_nir_with_options(builder.shader, &options);
   if (!shader || shader->error || !shader->asm_text) {
      fprintf(stderr, "%s\n", shader && shader->error ? shader->error : "vector compilation failed");
      if (shader && shader->asm_text)
         fprintf(stderr, "error exposed partial assembly\n");
      mgc_shader_destroy(shader);
      return 1;
   }
   fprintf(stdout, "%s", shader->asm_text);
   fprintf(stderr, "%s\n", shader->layout_json);
   mgc_shader_destroy(shader);
   return 0;
}
