#version 100
attribute vec4 a_position;
varying vec3 v_color;
void main() {
    gl_Position = a_position;
    v_color = a_position.xyz + vec3(0.5);
}
