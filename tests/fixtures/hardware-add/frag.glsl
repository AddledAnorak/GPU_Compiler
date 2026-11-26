#version 100
precision mediump float;
uniform float u_bias;
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(v_color + vec3(u_bias), 1.0);
}
