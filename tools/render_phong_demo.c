/* Render shaders/07_lighting_phong with Mesa softpipe to an offscreen PPM.
 *
 * From the repository root, after building Mesa and this executable:
 *   render_phong_demo <output.ppm> <light_x> <light_y> <light_z>
 * The Python/Pillow conversion to PNG is deliberately outside this small
 * GLES2 program. This is a visual reference, not an MGC hardware renderer.
 */
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#define SIZE 640
#define RINGS 56
#define SECTORS 112

typedef struct {
    GLfloat position[4];
    GLfloat normal[3];
} Vertex;

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) { perror(path); exit(1); }
    if (fseek(file, 0, SEEK_END)) fail("could not seek shader");
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET)) fail("could not size shader");
    char *source = malloc((size_t)size + 1);
    if (!source) fail("out of memory reading shader");
    if (fread(source, 1, (size_t)size, file) != (size_t)size)
        fail("could not read shader");
    source[size] = '\0';
    fclose(file);
    return source;
}

static GLuint compile_shader(GLenum type, const char *path) {
    char *source = read_file(path);
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, (const GLchar **)&source, NULL);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[4096] = {0};
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "%s: %s\n", path, log);
        fail("shader compilation failed");
    }
    free(source);
    return shader;
}

static Vertex sphere_point(int ring, int sector) {
    const float pi = 3.14159265358979323846f;
    float theta = pi * ring / RINGS;
    float phi = 2.0f * pi * sector / SECTORS;
    float x = sinf(theta) * cosf(phi);
    float y = cosf(theta);
    float z = sinf(theta) * sinf(phi);
    return (Vertex){{x, y, z, 1.0f}, {x, y, z}};
}

static Vertex *make_sphere(int *count) {
    *count = RINGS * SECTORS * 6;
    Vertex *vertices = malloc((size_t)*count * sizeof(*vertices));
    if (!vertices) fail("out of memory creating sphere");
    int i = 0;
    for (int ring = 0; ring < RINGS; ring++) {
        for (int sector = 0; sector < SECTORS; sector++) {
            Vertex a = sphere_point(ring, sector);
            Vertex b = sphere_point(ring + 1, sector);
            Vertex c = sphere_point(ring, sector + 1);
            Vertex d = sphere_point(ring + 1, sector + 1);
            vertices[i++] = a;
            vertices[i++] = b;
            vertices[i++] = c;
            vertices[i++] = c;
            vertices[i++] = b;
            vertices[i++] = d;
        }
    }
    return vertices;
}

static void save_ppm(const char *path, const unsigned char *rgba) {
    FILE *file = fopen(path, "wb");
    if (!file) { perror(path); exit(1); }
    fprintf(file, "P6\n%d %d\n255\n", SIZE, SIZE);
    for (int y = SIZE - 1; y >= 0; y--) {
        for (int x = 0; x < SIZE; x++) {
            const unsigned char *pixel = rgba + 4 * (y * SIZE + x);
            if (fwrite(pixel, 1, 3, file) != 3) fail("could not write PPM");
        }
    }
    fclose(file);
}

int main(int argc, char **argv) {
    if (argc != 5) fail("usage: render_phong_demo <output.ppm> <light_x> <light_y> <light_z>");
    char *end = NULL;
    GLfloat light[3];
    for (int i = 0; i < 3; i++) {
        light[i] = strtof(argv[i + 2], &end);
        if (!end || *end) fail("light coordinates must be numbers");
    }

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, NULL, NULL))
        fail("could not initialize EGL display");
    if (!eglBindAPI(EGL_OPENGL_ES_API)) fail("could not select GLES");
    EGLint config_attributes[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24, EGL_NONE
    };
    EGLConfig config;
    EGLint config_count = 0;
    if (!eglChooseConfig(display, config_attributes, &config, 1, &config_count)
        || config_count == 0) fail("no RGBA8/depth24 GLES2 EGL config");
    EGLint surface_attributes[] = {EGL_WIDTH, SIZE, EGL_HEIGHT, SIZE, EGL_NONE};
    EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
    EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
    if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT
        || !eglMakeCurrent(display, surface, surface, context))
        fail("could not make GLES2 context current");

    fprintf(stderr, "GL_RENDERER: %s\n", glGetString(GL_RENDERER));
    GLuint vertex = compile_shader(GL_VERTEX_SHADER, "shaders/07_lighting_phong/vert.glsl");
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, "shaders/07_lighting_phong/frag.glsl");
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[4096] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "%s\n", log);
        fail("shader link failed");
    }
    glUseProgram(program);

    const GLfloat mvp[16] = {
        0.82f, 0, 0, 0, 0, 0.82f, 0, 0,
        0, 0, -0.82f, 0, 0, 0, 0, 1
    };
    const GLfloat model[16] = {
        1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1
    };
    glUniformMatrix4fv(glGetUniformLocation(program, "u_mvp"), 1, GL_FALSE, mvp);
    glUniformMatrix4fv(glGetUniformLocation(program, "u_model"), 1, GL_FALSE, model);
    glUniform3fv(glGetUniformLocation(program, "u_light_pos"), 1, light);
    glUniform3f(glGetUniformLocation(program, "u_view_pos"), 0.0f, 0.0f, 4.0f);
    glUniform3f(glGetUniformLocation(program, "u_object_color"), 0.17f, 0.45f, 0.80f);

    int count = 0;
    Vertex *vertices = make_sphere(&count);
    GLint position = glGetAttribLocation(program, "a_position");
    GLint normal = glGetAttribLocation(program, "a_normal");
    if (position < 0 || normal < 0) fail("shader attributes not found");
    glEnableVertexAttribArray((GLuint)position);
    glEnableVertexAttribArray((GLuint)normal);
    glVertexAttribPointer((GLuint)position, 4, GL_FLOAT, GL_FALSE,
                          sizeof(Vertex), &vertices[0].position);
    glVertexAttribPointer((GLuint)normal, 3, GL_FLOAT, GL_FALSE,
                          sizeof(Vertex), &vertices[0].normal);

    glViewport(0, 0, SIZE, SIZE);
    glClearColor(0.045f, 0.055f, 0.085f, 1.0f);
    glClearDepthf(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDrawArrays(GL_TRIANGLES, 0, count);
    glFinish();
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "GLES error after draw: 0x%x\n", error);
        return 1;
    }

    unsigned char *pixels = malloc((size_t)SIZE * SIZE * 4);
    if (!pixels) fail("out of memory reading pixels");
    glReadPixels(0, 0, SIZE, SIZE, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if ((error = glGetError()) != GL_NO_ERROR) {
        fprintf(stderr, "GLES error after readback: 0x%x\n", error);
        return 1;
    }
    save_ppm(argv[1], pixels);
    free(pixels);
    free(vertices);
    glDeleteProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display, context);
    eglDestroySurface(display, surface);
    eglTerminate(display);
    return 0;
}
