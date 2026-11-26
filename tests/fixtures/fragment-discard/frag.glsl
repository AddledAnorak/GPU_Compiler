#version 100
precision mediump float;
varying vec3 v_position;
void main() {
    if (v_position.x < 0.0)
        discard;
    gl_FragColor = vec4(0.25, 0.5, 0.75, 0.0);
}
