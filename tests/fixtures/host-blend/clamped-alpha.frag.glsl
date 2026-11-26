#version 100
precision mediump float;
varying vec3 v_position;
void main() {
    gl_FragColor = vec4(0.5 - v_position.z, 0.0, 0.5 + v_position.z, 0.5 + v_position.z * 2.0);
}
