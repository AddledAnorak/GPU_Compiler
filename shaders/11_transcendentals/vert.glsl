// Supply a varying input so the fragment math cannot fold to constants.
#version 100
attribute vec4 a_position;
attribute vec3 a_color;
varying vec3 v_color;

void main() {
    v_color = a_color;
    gl_Position = a_position;
}
