// Vertex side is a plain passthrough -- the interesting part of this test
// pair is entirely in frag.glsl (constant-trip-count loop lowering). Only
// here to feed v_color into the fragment shader's accumulation loop.
#version 100
attribute vec4 a_position;
attribute vec3 a_color;
varying vec3 v_color;

void main() {
    v_color = a_color;
    gl_Position = a_position;
}
