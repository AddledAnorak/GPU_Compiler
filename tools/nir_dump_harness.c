/* Minimal offscreen GLES2 harness: compiles+links the given vertex and
 * fragment shader sources (glCompileShader triggers glsl_to_nir) and issues
 * one draw call to exercise softpipe. NIR_DEBUG=print records the initial
 * NIR and later snapshots. Used by tools/dump_nir.sh.
 *
 * Build against the locally-built Mesa checkout (mesa/build/), not any
 * system Mesa -- see Phase 0 of the plan for the meson setup command:
 *   PKG_CONFIG_PATH="mesa/build/meson-uninstalled" \
 *     gcc nir_dump_harness.c -o nir_dump_harness \
 *     $(pkg-config --cflags --libs egl glesv2)
 *
 * Usage: nir_dump_harness <vert.glsl> <frag.glsl>
 */
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <stdio.h>
#include <stdlib.h>

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    return buf;
}

static GLuint compile(GLenum type, const char *src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof(log), NULL, log);
        fprintf(stderr, "shader compile failed:\n%s\n", log);
        exit(1);
    }
    return sh;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <vert.glsl> <frag.glsl>\n", argv[0]);
        return 1;
    }

    /* Surfaceless EGL context -- no window system needed, matches
     * softpipe/llvmpipe's role as a pure software rasterizer in
     * FP16_ISA_V1's project doc Step 2. */
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(dpy, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);

    EGLint cfg_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE,
    };
    EGLConfig cfg;
    EGLint num_cfg;
    eglChooseConfig(dpy, cfg_attribs, &cfg, 1, &num_cfg);

    EGLint pbuf_attribs[] = { EGL_WIDTH, 4, EGL_HEIGHT, 4, EGL_NONE };
    EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pbuf_attribs);

    EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attribs);
    eglMakeCurrent(dpy, surf, surf, ctx);

    char *vsrc = read_file(argv[1]);
    char *fsrc = read_file(argv[2]);
    GLuint vs = compile(GL_VERTEX_SHADER, vsrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[4096];
        glGetProgramInfoLog(prog, sizeof(log), NULL, log);
        fprintf(stderr, "link failed:\n%s\n", log);
        return 1;
    }
    glUseProgram(prog);

    /* One triangle, enough to force the fragment shader to actually run
     * through the softpipe pipeline (Step 2 of the project plan). */
    GLfloat verts[] = { -1, -1, 0, 1,  1, -1, 0, 1,  0, 1, 0, 1 };
    GLint pos_loc = glGetAttribLocation(prog, "a_position");
    if (pos_loc >= 0) {
        glEnableVertexAttribArray(pos_loc);
        glVertexAttribPointer(pos_loc, 4, GL_FLOAT, GL_FALSE, 0, verts);
    }
    glViewport(0, 0, 4, 4);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglDestroySurface(dpy, surf);
    eglTerminate(dpy);
    return 0;
}
