# GLSL -> NIR Dictionary

Student A's Weeks 3-8 tracing and validation deliverable. Built from real
traces captured by
`tools/dump_nir.sh` (`docs/glsl-to-nir/traces/*.nir.txt`), Mesa 25.2.0,
softpipe reference driver, surfaceless EGL, `NIR_DEBUG=print`. Generated traces
are ignored by Git; run `tools/dump_nir.sh` from the repository root to recreate
all twelve. The excerpts below use the **last printed fragment-shader snapshot**
unless explicitly marked as an earlier snapshot.

Each trace contains **many** vertex and fragment NIR snapshots: the initial
`glsl_to_nir` result, intermediate linker/optimization passes, and later
softpipe-specific processing. Search for `shader: MESA_SHADER_FRAGMENT` and
inspect its last occurrence for the final printed fragment snapshot; search
for `glsl_to_nir` to see the initial result. The excerpts abbreviate the dumps
to show the relevant instructions. These are **softpipe observations**. The
`mgc` driver requests different NIR compiler options and runs its own lowering
(`mesa/src/gallium/drivers/mgc/mgc_screen.c` and `mgc_context.c`), so do not
assume its input is byte-for-byte identical to the last softpipe snapshot.

## `00_arith` -- arithmetic exists early, then folds to one constant

The fragment shader computes `(2.0 * 3.0 + 0.5) / 2.0` and writes it to RGB.
The initial `glsl_to_nir` fragment snapshot contains:

```
32  %5  = fmul %2, %4
32  %8  = fadd %5, %7
32  %11 = fdiv %8, %10
```

After Mesa's optimization passes, the last printed fragment snapshot is:

```
32x4  %1 = load_const (...) = (3.250000, 3.250000, 3.250000, 1.000000)
           @store_output (%1, ...)  // gl_FragColor
```

This maps GLSL `*`, `+`, `/` to early NIR `fmul`, `fadd`, `fdiv`, but the
arithmetic has disappeared from the last softpipe snapshot. A live `mgc`
compile of this example likewise emits constants rather than arithmetic. The
`mgc` compiler's separate NIR-builder arithmetic test exercises those
operations without letting GLSL fold them first (`mgc_test_arith.c`). The
paired vertex shader writes a constant `gl_Position`; its last printed NIR is
likewise `load_const` followed by `@store_output`.

## Vertex-stage handoff -- positions and varyings

In `01_flat_color/vert.glsl`, `gl_Position = a_position` becomes
`@load_input` for `a_position`, then `@store_output` for `gl_Position` in the
last printed vertex snapshot. In `02_varying/vert.glsl`, the additional
`v_color = a_color` becomes another `@load_input` / `@store_output` pair; its
fragment shader receives that varying through `@load_interpolated_input`.
This describes the information handoff across stages in these examples, not
an implemented vertex-output path in the current `mgc` driver.

## `01_flat_color` -- baseline constant write

```
32x4  %1 = load_const (1.000000, 0.000000, 0.000000, 1.000000)
           @store_output (%1, ...)  // gl_FragColor
```

The final fragment body has one color `load_const` + one `@store_output`.
No arithmetic remains: `gl_FragColor = vec4(1,0,0,1)` is constant. This is a
minimal compiler-path check because the current `mgc` backend only needs its
constant path (`docs/isa/isa.yaml`'s constant bank, index 3 = 1.0, index 2 =
0.0) and `store`.

## `02_varying` -- interpolated input passthrough

```
32x2  %1 = @load_barycentric_pixel (interp_mode=INTERP_MODE_SMOOTH)
32x3  %3 = @load_interpolated_input (%1, 0) (io location=VARYING_SLOT_VAR0 ...)
32x4  %5 = mov %3.xyzx
           @store_reg (%5, %0) (wrmask=xyz)
32x4  %6 = mov 1.0.xxxx
           @store_reg (%6, %0) (wrmask=w)
32x4  %7 = @load_reg (%0)
           @store_output (%7, ...)
```

Confirms varyings become the `@load_interpolated_input` intrinsic, gated by
a `@load_barycentric_pixel` intrinsic that supplies the interpolation
weights. Both are NIR *intrinsics*, not ALU ops -- outside the Weeks 3-4
arithmetic-only backend described in `nir_opcode_triage.md`. Note the
`@decl_reg`/`@load_reg`/`@store_reg` pattern: in this shader,
`gl_FragColor`'s 4 components get assembled into one vec4 register across two
partial writes (`wrmask=xyz` then `wrmask=w`) before `@store_output`. Constant
shaders such as `00_arith` and `01_flat_color` instead write a vec4 directly.

## `03_uniform` -- uniform load + multiply

```
32x4   %3 = @load_ubo_vec4 (0, 0) (base=0, component=0)     // u_tint
32x3   %5 = @load_interpolated_input (...)                  // v_color
32x4   %9 = fmul %8, %3
           @store_output (%9, ...)
```

In this softpipe trace, `u_tint` arrives as `@load_ubo_vec4` -- **not**
`load_uniform` -- through `uniform_0` at `base=0`. Both the softpipe and
`mgc` compiler options request uniform-to-UBO lowering; other uniforms and
layouts need their own trace checks. `nir_opcode_triage.md` proposes a direct
mapping from NIR `fmul` to MGC `fmul16`; that is a backend design claim, not
something the softpipe trace validates.

## `04_conditional` -- if/else fully flattens to `bcsel`

```
32    %4 = flt32 %2, %3            // u_threshold < interpolated luma
32x4  %7 = b32csel %4.xxxx, 1.0.xxxx, (0,1,0,1).xxxy
           @store_output (%7, ...)
```

For this shader, GLSL `if (luma > u_threshold) ... else ...` becomes a scalar
comparison plus `b32csel` (select between two values) in the last softpipe
fragment snapshot. The comparison is reversed without changing its meaning:
`u_threshold < luma`. The luma arithmetic from the GLSL fragment source is
visible in the last **vertex** snapshot, which stores one value for
interpolation; the last fragment snapshot loads that value. No `nir_if` node
remains in this sample's last fragment block. This does not establish that
every conditional becomes a select, or that `mgc` receives identical NIR.

**Backend mapping:** for this class of shader -- a scalar boolean condition
selecting between two already-computed values, with no side effects or
divergent work in either branch -- `mgc` uses the ISA's
conditional-execution prefix (`cond_enable`/
`cond_invert`/`cond_cr`, `isa.yaml` fields 26/25/24-22), mapping `flt32` to
`fcomp_l` writing a CR and `bcsel` to a pair of conditionally-executed
`copy`s. The current `mgc` compiler implements this restricted compare-fed
`bcsel` form in isolated NIR, but this full GLSL shader still needs its input
loads implemented. The condition encoding remains provisional. Real branches
are absent from this one final snapshot. `10_branch_discard` below shows what
happens when a conditional discards a fragment.

## `05_loop` -- fully evaluated at compile time, not just unrolled

```
32x4  %5 = fmul %4.xyzx, 2.083333.xxxx
           @store_reg (%5, %0) (wrmask=xyz)
```

The prediction in the shader's own header comment ("Mesa's optimizer will
very likely unroll this entirely") **undersold what actually happens**. The
loop doesn't just unroll to 4 `fmul`+`fadd` pairs -- the entire
`for (i=0;i<4;i++) accum += v_color * (1.0/float(i+1))` reduces to a
*single* `fmul` against the constant `2.083333` (= 1 + 1/2 + 1/3 + 1/4 =
25/12). Constant folding after unrolling collapsed the whole accumulation
into one scalar multiply. There is no `loop {}` construct, no
`nir_opt_loop_unroll` artifact, and no residual control flow anywhere in the
final NIR -- confirmed by grepping the trace for `loop {`, which only
appears in intermediate passes, never in the final block.

**Conclusion:** this particular shader tells us nothing about whether the
backend needs real loop control flow -- it tells us Mesa's optimizer can
make the question moot for compile-time-constant trip counts. A genuine
test of loop lowering needs a runtime-variable trip count. `09_runtime_loop`
below supplies one and retains a loop in its final printed NIR.

## `06_builtin_math` -- confirms the emulate-bucket, plus an FMA surprise

```
32x3   %5 = fadd %3, 0.001.xxx
32     %6 = fdot3_replicated %5, %5
32     %7 = frsq %6                          // normalize()'s 1/sqrt
32x3   %8 = fmul %5, %7.xxx
32    %10 = fsin %9                          // sin(u_time)
32    %12 = ffma %10, 0.5, 0.5               // * 0.5 + 0.5, fused
32x4  %13 = fmul %8.xyzx, %12.xxxx
```

In this trace, `normalize()` lowers to `fdot3_replicated` followed by `frsq`
(reciprocal square root), and `sin()` becomes `fsin`. The provisional
`docs/isa/isa.yaml` lists no native `frsq` or `fsin` instruction. The current
`mgc` backend expands `fsin` through Mesa's `lower_sincos`; `frsq` still needs
a backend path. This example does not prove that every use of `normalize()`, `length()`,
or `distance()` takes the same path.

**Additional finding:** `* 0.5 + 0.5` appears as one fused `ffma` node in
this softpipe trace, not separate `fmul` and `fadd` nodes. `mgc` requests
`lower_ffma16 = lower_ffma32 = true` because its proposed ISA has no FMA
instruction. That is a driver-specific lowering choice; the softpipe trace
alone does not show the exact form reaching `mgc`.

## Week 5-6: control flow and transcendental functions

These three added shader pairs answer questions the simpler Week 3-4 samples
could not. They are still **softpipe traces**, not proof that the MGC backend
can compile or execute the resulting NIR. Reproduce them with
`tools/dump_nir.sh` and inspect `09_runtime_loop.nir.txt`,
`10_branch_discard.nir.txt`, and `11_transcendentals.nir.txt` under
`docs/glsl-to-nir/traces/`.

### `09_runtime_loop` -- a loop that survives optimization

The fragment shader uses `for (int i = 0; i < u_limit; i++)`, where `u_limit`
is a uniform value unknown when the shader is compiled. Its **initial**
`glsl_to_nir` fragment snapshot contains a structured `loop { ... }` with an
integer less-than test (`ilt`), an `if` with `break`, and an `iadd` for the
counter. The **last** fragment snapshot still contains:

```
loop {
    32    %11 = @load_ubo_vec4 (...)       // runtime u_limit
    32    %13 = ige32 %12, %11             // i >= u_limit
    if %13 {
        break
    } else {
    }
    32    %15 = iadd %14, 1               // advance i
    32    %17 = i2f32 %16                 // float(i + 1)
    32x3  %19 = ffma %5, %17.xxx, %18    // accumulate color
}
```

Mesa reverses the loop-exit comparison but keeps the loop and its break.
The uniform is loaded from a UBO in this softpipe snapshot; the multiply plus
add has fused to `ffma`. Unlike `05_loop`, this example demonstrates NIR
control flow that a backend would need to handle if given an equivalent
shader. It does not show how MGC will lower that control flow.

### `10_branch_discard` -- a conditional with a side effect

The fragment shader discards the fragment when `v_color.r > u_threshold`.
Its **initial** fragment snapshot contains `if %5 { @terminate } else { }`.
The **last** fragment snapshot has a comparison followed by:

```
32  %5 = flt32 %4, %3.x             // u_threshold < v_color.r
         @terminate_if (%5)
```

Here Mesa removes the structured `if` but preserves its behavior with the
conditional `terminate_if` intrinsic: a true condition stops this fragment
before its color output. By contrast, `04_conditional` becomes `b32csel`
because both branches only choose a color. These two observed patterns show
why one value-only conditional is not a general rule for all `if` statements.

### `11_transcendentals` -- `sin`, `cos`, and `atan`

The fragment shader writes `vec4(sin(x), cos(x), atan(x), 1.0)` with
`x = u_phase + v_color.r`. The **initial** `glsl_to_nir` fragment snapshot
already has `fsin` and `fcos`. It has **no `fatan`**: Mesa has expanded
`atan(x)` into `fabs`, `frcp`, `fge`, `bcsel`, a polynomial sequence of `ffma`
operations, and bitwise operations to restore the sign. In the **last**
softpipe fragment snapshot, the recognizable core is:

```
32x4  %4 = fsin %3.xxxx
32x4  %5 = fcos %3.xxxx
32    %6 = fabs %3
32    %7 = frcp %3
32    %9 = fge32 1.0, %6
32   %10 = b32csel %9, %3, %7
          ... polynomial ffma operations, then sign restoration ...
```

In this sample, Mesa also moves the linear `u_phase + v_color.r` calculation
to the vertex stage and interpolates its result into the final fragment
shader. The transcendental operations stay in the fragment stage. This trace
shows the NIR forms produced by this Mesa/softpipe setup; it does not establish
native hardware support for any of the three functions. The provisional ISA
in `docs/isa/isa.yaml` lists no dedicated `sin`, `cos`, or `atan` opcode.

## Week 7-8: validate Phong and texture NIR across both shader stages

Run `tools/dump_nir.sh` from the repository root, then
`python3 tools/validate_week78_nir.py`. The first command regenerates the
Git-ignored Mesa/softpipe traces; the second checks the final vertex and
fragment snapshots and prints the observed data flow. Both shader pairs
compiled and linked when these traces were regenerated on Mesa 25.2.0.
The checks use the **last snapshot of each stage** because the dump repeats
the same shader after many passes. The initial fragment snapshot is also
checked for a dot product or texture sample, respectively.

### `07_lighting_phong`: positions and normals become a lit color

The vertex shader receives `a_position` and `a_normal`, loads the `u_model`
and `u_mvp` matrices from uniform-buffer data, and writes `gl_Position` plus
two packed varying slots. Mesa puts all three normal components and
`v_world_pos.x` in `VARYING_SLOT_VAR0`; `v_world_pos.yz` goes in
`VARYING_SLOT_VAR1`. The fragment shader interpolates both slots, so the
normal and world position reach each fragment despite that packing.

In the last fragment snapshot the three GLSL `normalize()` calls each become
a dot product, `frsq` (one divided by the square root), and multiply. The
diffuse and specular dot products pass through `fmax` to clamp them at zero.
`reflect()` contributes arithmetic including `ffma`; `pow(..., 32.0)`
becomes `fpow` with exponent 32. Uniform-buffer loads supply light position,
view position, and object color; the final arithmetic stores `gl_FragColor`.
The checked static counts are 6 `fdot3_replicated`, 3 `frsq`, 2 `fmax`,
1 `fpow`, and 2 `ffma`. These counts describe one NIR snapshot, not the
number of instructions executed for a rendered frame. `frsq` and `fpow`
remain backend concerns for MGC.

### `08_texture`: coordinates become a sampled color

The vertex shader transforms `a_position` by `u_mvp` for `gl_Position` and
passes `a_texcoord` into `VARYING_SLOT_VAR0`. The final fragment snapshot has
a direct chain:

```
%2 = @load_interpolated_input (...)  // interpolated VARYING_SLOT_VAR0
%3 = (float32)tex %2 (backend1), 0 (texture), 0 (sampler)
     @store_output (%3, ...)        // gl_FragColor
```

The initial fragment snapshot identifies the sampler through a dereference;
the final one uses texture and sampler index 0. `tex` is a NIR texture
instruction, not an ALU operation or intrinsic. MGC has no handler for it,
and the provisional ISA documents no sampler opcode. The trace establishes
what Mesa asks a backend to do; it does not establish how the FPGA will sample.

This is **NIR trace validation**, not a pixel-output or MGC execution test.
The minimal harness draws one triangle but does not initialize the Phong
uniforms or bind a texture, and it does not read back pixels. The softpipe
trace cannot prove MGC input loading, texture sampling, or rendered colors.

## Trace summary: possible backend requirements

| Shader | New backend requirement |
|---|---|
| `00_arith` | early `fmul`/`fadd`/`fdiv`, folded to constant + output in final NIR |
| `01_flat_color` | constant bank + `store` for a minimal compiler-path check |
| `02_varying` | `@load_interpolated_input` / `@load_barycentric_pixel` -- outside the Weeks 3-4 arithmetic prototype |
| `03_uniform` | softpipe shows `@load_ubo_vec4`; check the `mgc` path separately before implementing uniform loads |
| `04_conditional` | scalar compare + `b32csel` in softpipe; MGC handles the restricted compare/select form, but not this shader's input loads |
| `05_loop` | nothing -- constant folding removes it; not a useful loop-lowering test as written |
| `06_builtin_math` | softpipe shows `frsq`, `fsin`, and `ffma`; MGC lowers `fsin` but still needs `frsq` |
| `07_lighting_phong` | vertex inputs/matrices feed packed varyings; fragment has 3 `frsq` and 1 `fpow`, both needing a backend strategy |
| `08_texture` | vertex UV reaches fragment `tex` then color; NIR `tex` has no current MGC handler |
| `09_runtime_loop` | runtime-bound NIR `loop` and `break` survive the final softpipe snapshot |
| `10_branch_discard` | side-effecting conditional becomes `@terminate_if`, not `b32csel` |
| `11_transcendentals` | `fsin` and `fcos` remain; `atan` is already expanded into arithmetic and sign operations |
