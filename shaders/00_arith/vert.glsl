// Pure-constant position write, no attribute reads at all. Deliberately
// different from the other shader pairs' vertex shaders (which all read
// a_position via load_input): the mgc backend has no vertex-output
// storage convention yet (OPEN_QUESTIONS.md item 8) and no attribute-
// fetch support (Week 5-8), so this is the only vertex shader shape that
// can link successfully against it today. See mgc_compiler.c's handling
// of store_output in the vertex stage (a documented no-op, not a real
// destination) and the Weeks 1-4 plan's Phase 4 verification.
#version 100

void main() {
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
