// Week 7-8 validation target from the project plan: Phong lighting.
#version 100
attribute vec4 a_position;
attribute vec3 a_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
varying vec3 v_normal;
varying vec3 v_world_pos;

void main() {
    v_normal = mat3(u_model) * a_normal;
    vec4 world_pos = u_model * a_position;
    v_world_pos = world_pos.xyz;
    gl_Position = u_mvp * a_position;
}
