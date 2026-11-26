#version 100
precision mediump float;
uniform sampler2D u_texture;
varying vec2 v_uv;
void main() {
    gl_FragColor = texture2D(u_texture, v_uv * 4.0 - vec2(1.5));
}
