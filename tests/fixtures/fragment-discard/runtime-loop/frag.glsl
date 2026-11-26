#version 100
precision mediump float;
varying vec3 v_position;
void main() {
    gl_FragColor = vec4(0.0);
    for (int index = 0; index < int(v_position.z + 1.5); index++) {
        if (v_position.z > 0.0)
            gl_FragColor += vec4(0.0, 0.0, 0.5, 0.5);
        else
            gl_FragColor += vec4(1.0, 0.0, 0.0, 0.0);
        if (v_position.x < 0.0 && v_position.z < 0.0)
            discard;
    }
}
