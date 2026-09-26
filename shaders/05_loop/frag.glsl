// Exercises NIR loop lowering. With a compile-time-constant trip count,
// Mesa's optimizer will very likely unroll this entirely (nir_opt_loop_unroll)
// before it reaches the backend -- worth confirming in the trace, since it
// changes whether the backend ever needs real loop control flow at all in
// the common case.
#version 100
precision mediump float;
varying vec3 v_color;

void main() {
    vec3 accum = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        accum += v_color * (1.0 / float(i + 1));
    }
    gl_FragColor = vec4(accum, 1.0);
}
