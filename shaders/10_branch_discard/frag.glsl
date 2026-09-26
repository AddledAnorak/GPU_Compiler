// The discard is a side effect, unlike 04_conditional's value-only branches.
#version 100
precision mediump float;
varying vec3 v_color;
uniform float u_threshold;

void main() {
    if (v_color.r > u_threshold) {
        discard;
    }
    gl_FragColor = vec4(v_color, 1.0);
}
