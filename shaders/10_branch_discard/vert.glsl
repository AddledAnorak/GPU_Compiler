// Pass the color used by the fragment shader's conditional discard.
#version 100
attribute vec4 a_position;
attribute vec3 a_color;
varying vec3 v_color;

void main() {
    v_color = a_color;
    gl_Position = a_position;
}
