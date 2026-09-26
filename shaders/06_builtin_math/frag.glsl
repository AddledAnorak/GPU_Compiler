// Exercises the built-ins that become the emulate-bucket in
// docs/isa/nir_opcode_triage.md: fsin/fcos have no hardware support
// (frcp/normalize/length bottom out on a missing fsqrt). This shader is
// the canary for Week 5-6 -- if it doesn't compile once transcendentals
// are implemented, nothing depending on lighting or wave effects will.
#version 100
precision mediump float;
varying vec3 v_color;
uniform float u_time;

void main() {
    float wave = sin(u_time) * 0.5 + 0.5;
    vec3 n = normalize(v_color + vec3(0.001)); // exercises length()/fsqrt path
    gl_FragColor = vec4(n * wave, 1.0);
}
