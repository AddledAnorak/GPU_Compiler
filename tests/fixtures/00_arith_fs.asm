; mgc backend, MGC-FP16-v1, stage=MESA_SHADER_FRAGMENT
movi_f16 3.25, r0
movi_f16 3.25, r1
movi_f16 3.25, r2
movi_i32 1048576, r3
store r0, r3
movi_i32 1048577, r4
store r1, r4
movi_i32 1048578, r5
store r2, r5
movi_i32 1048579, r6
store #3, r6
stop_core
