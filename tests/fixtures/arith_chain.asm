; mgc backend, MGC-FP16-v1, stage=MESA_SHADER_FRAGMENT
movi_f16 2.0, r0
movi_f16 3.0, r1
fmul16 r0, r1, r2
wait_fmul
fadd16 #4, r2, r3
wait_fadd
movi_f16 2.0, r4
fdiv16 r3, r4, r5
wait_fdiv
movi_i32 1048576, r6
store r5, r6
stop_core
