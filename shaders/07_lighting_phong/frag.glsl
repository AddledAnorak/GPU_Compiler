#version 100
precision mediump float;
varying vec3 v_normal;
varying vec3 v_world_pos;

uniform vec3 u_light_pos;
uniform vec3 u_view_pos;
uniform vec3 u_object_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 light_dir = normalize(u_light_pos - v_world_pos);

    float diff = max(dot(n, light_dir), 0.0);
    vec3 diffuse = diff * u_object_color;

    vec3 view_dir = normalize(u_view_pos - v_world_pos);
    vec3 reflect_dir = reflect(-light_dir, n);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    vec3 specular = vec3(0.5) * spec;

    vec3 ambient = 0.1 * u_object_color;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}
