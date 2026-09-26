// Trivial case: no inputs, no computation. This is the shader used in
// FP16_ISA_V1.pdf-style minimal-example thinking -- the first thing that
// should trace cleanly from GLSL through NIR to a single fmov-style write.
#version 100
precision mediump float;

void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
