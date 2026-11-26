#version 100
attribute vec4 a_position;
varying vec2 v_uv;
void main() {
    gl_Position = a_position;
    v_uv = (a_position.xy + vec2(1.0)) * 0.5;
}
