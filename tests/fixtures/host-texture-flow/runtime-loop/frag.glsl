#version 100
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_texture;
void main() {
    gl_FragColor = vec4(0.0);
    for (int index = 0; index < int(v_uv.y * 3.0 + 1.0); index++) {
        gl_FragColor = texture2D(u_texture, v_uv);
        if (v_uv.x < 0.5)
            discard;
    }
}
