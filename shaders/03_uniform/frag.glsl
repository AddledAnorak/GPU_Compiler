// Exercises uniform loading (constant-buffer/register-preload path --
// i.e. our movi_* pseudo-ops / ploadu_l+loadu_h) mixed with an add.
#version 100
precision mediump float;
uniform vec4 u_tint;
varying vec3 v_color;

void main() {
    gl_FragColor = vec4(v_color, 1.0) * u_tint;
}
