#define main mgc_original_harness_main
#include "tools/mgc_harness.c"
#undef main

typedef void (GL_APIENTRYP acceptance_draw_proc)(GLenum, GLint, GLsizei, GLsizei);
typedef void (GL_APIENTRYP acceptance_base_draw_proc)(GLenum, GLint, GLsizei, GLsizei, GLuint);

static int acceptance_error(const char *operation) {
    GLenum error = glGetError();
    if (error == GL_NO_ERROR)
        return 1;
    fprintf(stderr, "MGC_ACCEPT_GL_ERROR %s=0x%x\n", operation, error);
    return 0;
}

static int acceptance_word(FILE *stream, unsigned *value) {
    unsigned char bytes[4];
    if (fread(bytes, 1, sizeof(bytes), stream) != sizeof(bytes))
        return 0;
    *value = little_u32(bytes);
    return 1;
}

static int acceptance_textures(GLuint program, GLuint names[4], unsigned *count) {
    const char *path = getenv("MGC_ACCEPT_TEXTURES");
    *count = 0;
    if (!path)
        return 1;
    FILE *stream = fopen(path, "rb");
    unsigned char magic[8];
    if (!stream || fread(magic, 1, 8, stream) != 8 || memcmp(magic, "MGCTX04\0", 8) ||
        !acceptance_word(stream, count) || *count < 1 || *count > 4) {
        if (stream)
            fclose(stream);
        fprintf(stderr, "MGC_ACCEPT_TEXTURE_INVALID header\n");
        return 0;
    }
    const char *mode = getenv("MGC_HOST_TEXTURE");
    unsigned version = mode ? (unsigned)strtoul(mode, NULL, 10) : 0;
    if (version < 11 || version > 14) {
        fclose(stream);
        return 0;
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures((GLsizei)*count, names);
    unsigned used = 0;
    int valid = 1;
    for (unsigned index = 0; valid && index < *count; index++) {
        unsigned binding, width, height, format, levels;
        valid = acceptance_word(stream, &binding) && acceptance_word(stream, &width) &&
                acceptance_word(stream, &height) && acceptance_word(stream, &format) &&
                acceptance_word(stream, &levels);
        if (!valid || binding > 3 || used & (1u << binding) || width < 1 || width > 32 ||
            height < 1 || height > 32 || format > 3 || levels < 1 || levels > 6) {
            valid = 0;
            break;
        }
        used |= 1u << binding;
        const unsigned channels[] = {4, 1, 3, 4};
        const GLenum external[] = {GL_RGBA, 0x1903, GL_RGB, 0x80E1};
        const GLint internal[] = {0x8058, 0x8229, 0x8051, 0x8058};
        const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
        if (format == 1 && !extension_advertised(extensions, "GL_ARB_texture_rg")) {
            fprintf(stderr, "MGC_ACCEPT_EXTENSION_UNAVAILABLE GL_ARB_texture_rg\n");
            valid = 0;
            break;
        }
        glActiveTexture(GL_TEXTURE0 + binding);
        glBindTexture(GL_TEXTURE_2D, names[index]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, version == 11 ? GL_NEAREST : GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        version == 11 ? GL_NEAREST_MIPMAP_NEAREST :
                        version == 12 ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR_MIPMAP_LINEAR);
        if (version == 14) {
            if (!extension_advertised(extensions, "GL_EXT_texture_filter_anisotropic")) {
                fprintf(stderr, "MGC_ACCEPT_EXTENSION_UNAVAILABLE GL_EXT_texture_filter_anisotropic\n");
                valid = 0;
                break;
            }
            GLfloat maximum = 0, anisotropy = 0;
            glGetFloatv(0x84FF, &maximum);
            glTexParameterf(GL_TEXTURE_2D, 0x84FE, 4);
            glGetTexParameterfv(GL_TEXTURE_2D, 0x84FE, &anisotropy);
            int arb = extension_advertised(extensions, "GL_ARB_texture_filter_anisotropic");
            const char *renderer = (const char *)glGetString(GL_RENDERER);
            if (maximum < 4 || anisotropy != 4 ||
                (!strcmp(renderer, "mgc") && (maximum != 4 || arb))) {
                valid = 0;
                break;
            }
            fprintf(stderr, "MGC_ACCEPT_ANISOTROPY maximum=%.0f sampler=%.0f arb=%d\n", maximum, anisotropy, arb);
        }
        for (unsigned level = 0; valid && level < levels; level++) {
            unsigned char texels[32 * 32 * 4];
            size_t size = width * height * channels[format];
            if (fread(texels, 1, size, stream) != size ||
                (width == 1 && height == 1 && level + 1 != levels)) {
                valid = 0;
                break;
            }
            glTexImage2D(GL_TEXTURE_2D, (GLint)level, internal[format], (GLsizei)width,
                         (GLsizei)height, 0, external[format], GL_UNSIGNED_BYTE, texels);
            fprintf(stderr, "MGC_ACCEPT_TEXTURE binding=%u level=%u width=%u height=%u format=%u\n",
                    binding, level, width, height, format);
            if (level + 1 < levels) {
                width = width > 1 ? width / 2 : 1;
                height = height > 1 ? height / 2 : 1;
            }
        }
        if (width != 1 || height != 1)
            valid = 0;
        char uniform[32];
        snprintf(uniform, sizeof(uniform), "u_textures[%u]", binding);
        GLint location = glGetUniformLocation(program, uniform);
        if (location < 0 && binding == 0)
            location = glGetUniformLocation(program, "u_texture");
        if (location < 0) {
            fprintf(stderr, "MGC_ACCEPT_UNIFORM_UNAVAILABLE %s\n", uniform);
            valid = 0;
        } else {
            glUniform1i(location, (GLint)binding);
        }
    }
    if (fgetc(stream) != EOF || ferror(stream) || fclose(stream))
        valid = 0;
    return valid && acceptance_error("texture upload and sampler uniforms");
}

static int acceptance_write(const char *path, const void *bytes, size_t size) {
    FILE *stream = fopen(path, "wb");
    if (!stream)
        return 0;
    int success = fwrite(bytes, 1, size, stream) == size;
    if (fclose(stream))
        success = 0;
    return success;
}

static int acceptance_draw(GLuint program, const char *path, const char *name) {
    unsigned vertex_count = 0;
    const char *mesh_path = getenv("MGC_HOST_MESH");
    struct triangle_vertex *vertices = mesh_path ? read_mesh(mesh_path, &vertex_count) : NULL;
    if (!vertices || vertex_count > 24) {
        free(vertices);
        fprintf(stderr, "MGC_ACCEPT_MESH_INVALID bounded real triangles required\n");
        return 0;
    }
    for (unsigned index = 0; index < vertex_count; index++)
        memcpy(vertices[index].color, vertices[index].normal, sizeof(vertices[index].color));
    GLuint framebuffer = 0, color = 0, depth = 0, buffer = 0, textures[4] = {0};
    unsigned texture_count = 0;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    int rgb565 = !strcmp(name, "rgb565-color-mask");
    int red_only = !strcmp(name, "r8-framebuffer");
    int bgra = !strcmp(name, "bgra8-framebuffer");
    GLenum external = red_only ? 0x1903 : rgb565 ? GL_RGB : bgra ? 0x80E1 : GL_RGBA;
    GLint internal = red_only ? 0x8229 : rgb565 ? 0x8D62 : bgra ? 0x80E1 : 0x8058;
    if (bgra && !extension_advertised((const char *)glGetString(GL_EXTENSIONS),
                                     "GL_EXT_texture_format_BGRA8888")) {
        fprintf(stderr, "MGC_ACCEPT_EXTENSION_UNAVAILABLE GL_EXT_texture_format_BGRA8888\n");
        free(vertices);
        return 0;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, internal, 8, 8, 0, external,
                 rgb565 ? GL_UNSIGNED_SHORT_5_6_5 : GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    int stencil = !strcmp(name, "stencil-z24s8");
    if (stencil) {
        glGenRenderbuffers(1, &depth);
        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glRenderbufferStorage(GL_RENDERBUFFER, 0x88F0, 8, 8);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth);
    }
    if (!acceptance_error("framebuffer allocation") ||
        glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "MGC_ACCEPT_FRAMEBUFFER_INCOMPLETE\n");
        free(vertices);
        return 0;
    }
    glViewport(0, 0, 8, 8);
    glDisable(GL_DITHER);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glFrontFace(GL_CCW);
    if (!bgra) {
        glDisable(0x809D);
        desktop_clamp_color_proc clamp_color = (desktop_clamp_color_proc)eglGetProcAddress("glClampColorARB");
        if (!clamp_color || !extension_advertised((const char *)glGetString(GL_EXTENSIONS),
                                                 "GL_ARB_color_buffer_float")) {
            fprintf(stderr, "MGC_ACCEPT_EXTENSION_UNAVAILABLE GL_ARB_color_buffer_float\n");
            free(vertices);
            return 0;
        }
        clamp_color(0x891A, GL_FALSE);
    }
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glStencilFuncSeparate(GL_FRONT_AND_BACK, GL_ALWAYS, 0, 255);
    glStencilMaskSeparate(GL_FRONT_AND_BACK, 255);
    glStencilOpSeparate(GL_FRONT_AND_BACK, GL_KEEP, GL_KEEP, GL_KEEP);
    glBlendFuncSeparate(GL_ONE, GL_ZERO, GL_ONE, GL_ZERO);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendColor(0, 0, 0, 0);
    int blending = !strcmp(name, "separate-blend-factors");
    if (blending || rgb565)
        glClearColor(.25f, .5f, .75f, .75f);
    else
        glClearColor(0, 0, 0, 0);
    if (bgra) {
        glClearDepthf(1.0f);
    } else {
        desktop_clear_depth_proc clear_depth = (desktop_clear_depth_proc)eglGetProcAddress("glClearDepth");
        if (!clear_depth) {
            free(vertices);
            return 0;
        }
        clear_depth(1.0);
    }
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | (stencil ? GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT : 0));
    if (blending) {
        glEnable(GL_BLEND);
        glBlendColor(.5f, .25f, .75f, .5f);
        glBlendFuncSeparate(GL_CONSTANT_COLOR, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_REVERSE_SUBTRACT);
    }
    if (rgb565)
        glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_TRUE);
    if (stencil) {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_STENCIL_TEST);
        glStencilFuncSeparate(GL_FRONT, GL_ALWAYS, 6, 255);
        glStencilFuncSeparate(GL_BACK, GL_ALWAYS, 11, 255);
        glStencilOpSeparate(GL_FRONT, GL_ZERO, GL_INCR_WRAP, GL_REPLACE);
        glStencilOpSeparate(GL_BACK, GL_ZERO, GL_INVERT, GL_REPLACE);
    }
    glUseProgram(program);
    GLint location = glGetUniformLocation(program, "u_lod");
    if (location >= 0)
        glUniform1f(location, !strcmp(name, "fractional-lod-trilinear") ? .5f : .25f);
    if (!acceptance_textures(program, textures, &texture_count)) {
        free(vertices);
        return 0;
    }
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(*vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(*vertices), (void *)offsetof(struct triangle_vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(*vertices), (void *)offsetof(struct triangle_vertex, color));
    if (!acceptance_error("clear, state and actual vertex upload")) {
        free(vertices);
        return 0;
    }
    if (!strcmp(name, "local-instance-id") || !strcmp(name, "vertex-local-instance-id") || !strcmp(name, "base-instance")) {
        const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
        if (!extension_advertised(extensions, "GL_ARB_draw_instanced")) {
            fprintf(stderr, "MGC_ACCEPT_EXTENSION_UNAVAILABLE GL_ARB_draw_instanced\n");
            free(vertices);
            return 0;
        }
        if (!strcmp(name, "base-instance")) {
            acceptance_base_draw_proc draw = (acceptance_base_draw_proc)eglGetProcAddress("glDrawArraysInstancedBaseInstance");
            if (!draw || !extension_advertised(extensions, "GL_ARB_base_instance") ||
                !extension_advertised(extensions, "GL_ARB_shader_draw_parameters")) {
                fprintf(stderr, "MGC_ACCEPT_EXTENSION_UNAVAILABLE actual base-instance draw/GLSL\n");
                free(vertices);
                return 0;
            }
            draw(GL_TRIANGLES, 0, (GLsizei)vertex_count, 2, 3);
        } else {
            acceptance_draw_proc draw = (acceptance_draw_proc)eglGetProcAddress("glDrawArraysInstancedARB");
            if (!draw) {
                free(vertices);
                return 0;
            }
            draw(GL_TRIANGLES, 0, (GLsizei)vertex_count, 2);
        }
    } else {
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertex_count);
    }
    glFinish();
    int success = acceptance_error("actual draw and finish");
    if (success && path) {
        GLubyte rgba[8 * 8 * 4];
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        success = acceptance_error("actual glReadPixels RGBA8") && acceptance_write(path, rgba, sizeof(rgba));
        if (success && stencil) {
            uint32_t packed[8 * 8];
            glReadPixels(0, 0, 8, 8, 0x84F9, 0x84FA, packed);
            success = acceptance_error("actual glReadPixels Z24S8");
            for (unsigned index = 0; index < 8 * 8; index++)
                packed[index] = (packed[index] >> 8) | (packed[index] << 24);
            const char *depth_path = getenv("MGC_ACCEPT_DEPTH_OUTPUT");
            success = success && depth_path && acceptance_write(depth_path, packed, sizeof(packed));
        }
        if (success)
            fprintf(stderr, "MGC_ACCEPT_READBACK actual_glReadPixels=1 width=8 height=8\n");
    }
    fprintf(stderr, "MGC_ACCEPT_DRAW case=%s vertices=%u texture_count=%u\n", name, vertex_count, texture_count);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDeleteBuffers(1, &buffer);
    glDeleteTextures((GLsizei)texture_count, textures);
    glDeleteTextures(1, &color);
    glDeleteRenderbuffers(1, &depth);
    glDeleteFramebuffers(1, &framebuffer);
    free(vertices);
    return success && acceptance_error("cleanup");
}

int main(int argc, char **argv) {
    const char *name = getenv("MGC_ACCEPT_CASE");
    int capture = argc == 4 && !strcmp(argv[3], "--draw");
    int reference = argc == 5 && !strcmp(argv[3], "--reference");
    int native = argc == 5 && !strcmp(argv[3], "--host-execute");
    const char *features = getenv("MGC_HOST_FEATURES");
    const char *target = getenv("MGC_COMPILE_TARGET");
    if (!name || (!capture && !reference && !native) || !features || strcmp(features, "1") ||
        !target || strcmp(target, "legacy-mgcsim-v1") ||
        (native && !getenv("MGC_HOST_EXECUTOR_SOCKET"))) {
        fprintf(stderr, "MGC_ACCEPT_ENV_INVALID explicit ABI4 legacy feature draw required\n");
        return 1;
    }
    int es_profile = !strcmp(name, "bgra8-framebuffer");
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, NULL, NULL) ||
        !eglBindAPI(es_profile ? EGL_OPENGL_ES_API : EGL_OPENGL_API)) {
        fprintf(stderr, "MGC_ACCEPT_EGL_INIT 0x%x\n", eglGetError());
        return 1;
    }
    EGLint attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
                           es_profile ? EGL_OPENGL_ES2_BIT : EGL_OPENGL_BIT,
                           EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
    EGLConfig config;
    EGLint count = 0;
    EGLint context_attributes[] = {EGL_NONE};
    EGLint es_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    if (!eglChooseConfig(display, attributes, &config, 1, &count) || count != 1)
        return 1;
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT,
                                         es_profile ? es_attributes : context_attributes);
    if (context == EGL_NO_CONTEXT || !eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context)) {
        fprintf(stderr, "MGC_ACCEPT_DESKTOP_UNAVAILABLE 0x%x\n", eglGetError());
        return 1;
    }
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    const char *version = (const char *)glGetString(GL_VERSION);
    const char *language = (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
    fprintf(stderr, "GL_RENDERER: %s\nGL_VERSION: %s\nGL_SHADING_LANGUAGE_VERSION: %s\n",
            renderer ? renderer : "unavailable", version ? version : "unavailable", language ? language : "unavailable");
    const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
    const char *capabilities[] = {"GL_ARB_draw_instanced", "GL_ARB_base_instance",
                                 "GL_ARB_shader_draw_parameters", "GL_EXT_gpu_shader4"};
    for (unsigned index = 0; index < sizeof(capabilities) / sizeof(capabilities[0]); index++)
        fprintf(stderr, "MGC_ACCEPT_CAPABILITY %s=%d\n", capabilities[index],
                extension_advertised(extensions, capabilities[index]));
    int success = renderer && version && language && (!!strstr(version, "OpenGL ES") == es_profile) &&
                  !strcmp(renderer, reference ? "softpipe" : "mgc");
    if (getenv("MGC_HOST_TEXTURE") &&
        !extension_advertised((const char *)glGetString(GL_EXTENSIONS), "GL_ARB_shader_texture_lod")) {
        fprintf(stderr, "MGC_ACCEPT_EXTENSION_UNAVAILABLE GL_ARB_shader_texture_lod\n");
        success = 0;
    }
    if (success) {
        char *vertex_source = read_file(argv[1]);
        char *fragment_source = read_file(argv[2]);
        GLuint vertex = compile(GL_VERTEX_SHADER, vertex_source);
        GLuint fragment = compile(GL_FRAGMENT_SHADER, fragment_source);
        free(vertex_source);
        free(fragment_source);
        GLuint program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glBindAttribLocation(program, 0, "a_position");
        glBindAttribLocation(program, 1, "a_color");
        glLinkProgram(program);
        GLint linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        char log[4096] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        if (log[0])
            fprintf(stderr, "%s\n", log);
        success = linked && acceptance_draw(program, capture ? NULL : argv[4], name);
        glDeleteProgram(program);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display, context);
    eglTerminate(display);
    return success ? 0 : 1;
}
