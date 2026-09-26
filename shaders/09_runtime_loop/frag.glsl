// Unlike 05_loop, this loop's trip count is unknown at compile time.
#version 100
precision mediump float;
varying vec3 v_color;
uniform int u_limit;

void main() {
    vec3 accum = vec3(0.0);
    for (int i = 0; i < u_limit; i++) {
        accum += v_color * float(i + 1);
    }
    gl_FragColor = vec4(accum, 1.0);
}
