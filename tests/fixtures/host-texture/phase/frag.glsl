#version 100
precision mediump float;
uniform sampler2D u_texture;
varying vec2 v_uv;
void main() {
    vec2 coordinates = v_uv.yx * vec2(-1.0, 1.0) + vec2(1.0, 0.0);
    vec2 copied = coordinates;
    gl_FragColor = texture2D(u_texture, copied);
}
