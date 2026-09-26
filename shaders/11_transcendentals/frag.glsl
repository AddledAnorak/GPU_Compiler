// Compare Mesa's NIR for the three Week 5-6 functions in one shader.
#version 100
precision mediump float;
varying vec3 v_color;
uniform float u_phase;

void main() {
    float x = u_phase + v_color.r;
    gl_FragColor = vec4(sin(x), cos(x), atan(x), 1.0);
}
