// Trivial FP16 arithmetic through the real GLSL -> Mesa -> NIR -> mgc
// pipeline, complementing the standalone NIR-builder unit test
// (mesa/src/gallium/drivers/mgc/mgc_test_arith.c) that proves the same
// (2*3+0.5)/2 = 3.25 computation without going through GLSL at all. Both
// exist because GLSL constant-folds pure-literal arithmetic before it
// reaches any driver (see docs/glsl-to-nir/dictionary.md's 05_loop
// finding) -- this shader is expected to arrive at the backend already
// folded to a single constant, which is itself the useful thing to
// confirm about the real compiler, while mgc_test_arith is what actually
// exercises the multi-instruction fmul16/fadd16/fdiv16 + wait_* path.
// See the Weeks 1-4 plan, Phase 4 verification.
#version 100
precision mediump float;

void main() {
    float c = (2.0 * 3.0 + 0.5) / 2.0;
    gl_FragColor = vec4(c, c, c, 1.0);
}
