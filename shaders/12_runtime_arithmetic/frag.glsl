#version 100
precision mediump float;
varying vec3 v_color;
uniform float u_bias;
uniform float u_divisor;

void main() {
    float value = (v_color.r + u_bias) / (u_divisor + v_color.g);
    gl_FragColor = vec4(value, -value, v_color.b, 1.0);
}
