#version 100
precision mediump float;
varying vec2 v_texcoord;
uniform sampler2D u_tex;

void main() {
    gl_FragColor = texture2D(u_tex, v_texcoord);
}
