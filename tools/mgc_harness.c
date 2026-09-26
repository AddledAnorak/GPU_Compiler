/* Minimal offscreen GLES2 harness for the mgc backend: compiles+links the
 * given vertex and fragment shader sources and stops. Unlike
 * nir_dump_harness.c (which issues a draw call so softpipe's real
 * rasterizer runs), this harness deliberately never calls glDrawArrays --
 * the mgc driver's rendering path (draw_vbo, clear, resource storage) is
 * unimplemented no-op scaffolding, entirely out of scope for the Weeks
 * 1-4 plan (which targets "context created + shader compiled + backend
 * called", not rendering), and calling glDrawArrays crashes inside
 * Mesa's internal state-tracker update-function dispatch (st_util.h's
 * st_validate_state) because the resource/rendering layer was never
 * meant to be exercised.
 *
 * Also unlike nir_dump_harness.c, this doesn't need a pbuffer/EGL
 * surface at all -- eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE,
 * ctx) is enough to reach glLinkProgram, which is where
 * create_fs_state/create_vs_state (mgc_context.c) run and produce the
 * MGC_ASM_DUMP output. See docs/driver/gallium-integration.md and the
 * Weeks 1-4 plan, Phase 4 verification.
 *
 * Build against the locally-built Mesa checkout, same as
 * nir_dump_harness.c:
 *   PKG_CONFIG_PATH="mesa/build/meson-uninstalled" \
 *     gcc mgc_harness.c -o mgc_harness $(pkg-config --cflags --libs egl glesv2)
 *
 * Usage: mgc_harness <vert.glsl> <frag.glsl>
 * Exit status matches glLinkProgram's GL_LINK_STATUS: 0 on success (the
 * expected outcome, once the involved shaders stay inside the backend's
 * arithmetic/constant/store_output subset -- see
 * docs/isa/nir_opcode_triage.md), 1 on failure, with the link log printed
 * either way so an out-of-scope shader's error is visible, not silent.
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

    EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attribs);
    /* No surface at all -- EGL_KHR_surfaceless_context. */
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);

    fprintf(stderr, "GL_RENDERER: %s\n", (const char *)glGetString(GL_RENDERER));

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
    char log[4096];
    glGetProgramInfoLog(prog, sizeof(log), NULL, log);
    if (log[0])
        fprintf(stderr, "%s\n", log);

    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglTerminate(dpy);
    return linked ? 0 : 1;
}
