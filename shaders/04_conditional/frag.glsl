// Exercises NIR's if/else lowering -- the key question for the dictionary
// (docs/glsl-to-nir/dictionary.md) is whether this stays structured
// (nir_if) or flattens to bcsel, since that decides whether the backend
// needs real branches (goto/jump + CR) or just conditional execution.
#version 100
precision mediump float;
varying vec3 v_color;
uniform float u_threshold;

void main() {
    float luma = v_color.r * 0.3 + v_color.g * 0.59 + v_color.b * 0.11;
    if (luma > u_threshold) {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
