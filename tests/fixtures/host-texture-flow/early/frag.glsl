#version 100
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_texture;
void main() {
    if (v_uv.x < 0.5)
        discard;
    gl_FragColor = texture2D(u_texture, v_uv);
}
