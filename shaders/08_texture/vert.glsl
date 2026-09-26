// Week 7-8 validation target from the project plan: texture mapping.
// Note (docs/isa/OPEN_QUESTIONS.md #13 / ENCODING_GUESSES.md): there is no
// sample instruction in MGC-FP16-v1 at all -- this exercises what the
// *frontend* produces (a NIR texture instruction). Implementing sampling
// for MGC is backend/driver work, outside this frontend trace.
#version 100
attribute vec4 a_position;
attribute vec2 a_texcoord;
uniform mat4 u_mvp;
varying vec2 v_texcoord;

void main() {
    v_texcoord = a_texcoord;
    gl_Position = u_mvp * a_position;
}
