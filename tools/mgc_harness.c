#define _POSIX_C_SOURCE 200809L

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct triangle_vertex {
    GLfloat position[4];
    GLfloat color[3];
    GLfloat normal[3];
};

struct host_texture {
    unsigned width;
    unsigned height;
    GLubyte texels[16 * 16 * 4];
};

struct host_instances {
    unsigned count;
    unsigned divisors[2];
    unsigned components[2];
    unsigned records[2];
    GLfloat values[2][16 * 4];
};

struct host_mipmaps {
    unsigned level_count;
    unsigned widths[5];
    unsigned heights[5];
    size_t offsets[5];
    GLubyte texels[1364];
};

typedef void (GL_APIENTRYP instance_divisor_proc)(GLuint, GLuint);
typedef void (GL_APIENTRYP instance_draw_proc)(GLenum, GLint, GLsizei, GLsizei);
typedef void (GL_APIENTRYP desktop_clear_depth_proc)(double);
typedef void (GL_APIENTRYP desktop_clamp_color_proc)(GLenum, GLenum);

static uint32_t little_u32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
           (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static int extension_advertised(const char *extensions, const char *name) {
    if (!extensions)
        return 0;
    size_t length = strlen(name);
    for (const char *cursor = extensions; (cursor = strstr(cursor, name)); cursor += length)
        if ((cursor == extensions || cursor[-1] == ' ') && (!cursor[length] || cursor[length] == ' '))
            return 1;
    return 0;
}

static int desktop_120_source(const char *source) {
    while (*source == ' ' || *source == '\t' || *source == '\r' || *source == '\n')
        source++;
    if (strncmp(source, "#version", 8))
        return 0;
    source += 8;
    if (*source != ' ' && *source != '\t')
        return 0;
    while (*source == ' ' || *source == '\t')
        source++;
    return !strncmp(source, "120", 3) &&
           (!source[3] || source[3] == ' ' || source[3] == '\t' || source[3] == '\r' || source[3] == '\n');
}

static int read_instances(const char *path, struct host_instances *instances) {
    unsigned char bytes[28 + 16 * 8 * 4];
    int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    struct stat storage;
    if (descriptor < 0 || fstat(descriptor, &storage) || !S_ISREG(storage.st_mode) ||
        storage.st_size <= 28 || storage.st_size > (off_t)sizeof(bytes)) {
        fprintf(stderr, "instance input requires an ordinary bounded MGCINS01 file\n");
        if (descriptor >= 0)
            close(descriptor);
        return 0;
    }
    size_t length = (size_t)storage.st_size;
    size_t received = 0;
    while (received < length) {
        ssize_t result = read(descriptor, bytes + received, length - received);
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            break;
        received += (size_t)result;
    }
    unsigned char extra;
    ssize_t trailing;
    do {
        trailing = read(descriptor, &extra, 1);
    } while (trailing < 0 && errno == EINTR);
    int closed = close(descriptor);
    if (received != length || trailing != 0 || closed || memcmp(bytes, "MGCINS01", 8))
        goto invalid;
    memset(instances, 0, sizeof(*instances));
    instances->count = little_u32(bytes + 8);
    if (instances->count < 1 || instances->count > 16)
        goto invalid;
    size_t scalar_count = 0;
    for (unsigned stream = 0; stream < 2; stream++) {
        instances->divisors[stream] = little_u32(bytes + 12 + stream * 4);
        instances->components[stream] = little_u32(bytes + 20 + stream * 4);
        if (instances->divisors[stream] < 1 || instances->divisors[stream] > 1048576 ||
            instances->components[stream] < 3 || instances->components[stream] > 4)
            goto invalid;
        instances->records[stream] = (instances->count - 1) / instances->divisors[stream] + 1;
        scalar_count += instances->records[stream] * instances->components[stream];
    }
    if (length != 28 + scalar_count * 4)
        goto invalid;
    size_t cursor = 28;
    for (unsigned stream = 0; stream < 2; stream++) {
        unsigned scalars = instances->records[stream] * instances->components[stream];
        for (unsigned index = 0; index < scalars; index++, cursor += 4) {
            uint32_t word = little_u32(bytes + cursor);
            memcpy(&instances->values[stream][index], &word, sizeof(word));
            if (!isfinite(instances->values[stream][index]))
                goto invalid;
        }
    }
    return 1;
invalid:
    fprintf(stderr, "invalid MGCINS01 count/divisors/vec3-or-vec4/exact finite FP32 instance records\n");
    return 0;
}

static int read_mipmaps(const char *path, struct host_mipmaps *mipmaps) {
    unsigned char bytes[1384];
    int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    struct stat storage;
    if (descriptor < 0 || fstat(descriptor, &storage) || !S_ISREG(storage.st_mode) ||
        storage.st_size < 24 || storage.st_size > (off_t)sizeof(bytes)) {
        fprintf(stderr, "mipmaps require an ordinary bounded MGCMIP1 file\n");
        if (descriptor >= 0)
            close(descriptor);
        return 0;
    }
    size_t length = (size_t)storage.st_size;
    size_t received = 0;
    while (received < length) {
        ssize_t result = read(descriptor, bytes + received, length - received);
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            break;
        received += (size_t)result;
    }
    unsigned char extra;
    ssize_t trailing;
    do {
        trailing = read(descriptor, &extra, 1);
    } while (trailing < 0 && errno == EINTR);
    int closed = close(descriptor);
    if (received != length || trailing != 0 || closed || memcmp(bytes, "MGCMIP1\0", 8))
        goto invalid;
    unsigned width = little_u32(bytes + 8);
    unsigned height = little_u32(bytes + 12);
    mipmaps->level_count = little_u32(bytes + 16);
    if (width < 1 || width > 16 || height < 1 || height > 16 ||
        mipmaps->level_count < 1 || mipmaps->level_count > 5)
        goto invalid;
    size_t payload = 0;
    for (unsigned level = 0; level < mipmaps->level_count; level++) {
        mipmaps->widths[level] = width;
        mipmaps->heights[level] = height;
        mipmaps->offsets[level] = payload;
        payload += width * height * 4;
        if (width == 1 && height == 1 && level + 1 != mipmaps->level_count)
            goto invalid;
        if (level + 1 < mipmaps->level_count) {
            width = width > 1 ? width / 2 : 1;
            height = height > 1 ? height / 2 : 1;
        }
    }
    if (width != 1 || height != 1 || length != 20 + payload || payload > sizeof(mipmaps->texels))
        goto invalid;
    memcpy(mipmaps->texels, bytes + 20, payload);
    return 1;
invalid:
    fprintf(stderr, "invalid MGCMIP1 width/height/count/exact complete RGBA8 mipchain through 1x1\n");
    return 0;
}

static int read_texture(const char *path, struct host_texture *texture) {
    int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    struct stat storage;
    if (descriptor < 0 || fstat(descriptor, &storage) || !S_ISREG(storage.st_mode) ||
        storage.st_size < 12 || storage.st_size > (off_t)(8 + sizeof(texture->texels))) {
        fprintf(stderr, "invalid texture file (expected bounded width/height plus RGBA8): %s\n", path);
        if (descriptor >= 0)
            close(descriptor);
        return 0;
    }
    unsigned char bytes[8 + 16 * 16 * 4];
    size_t size = storage.st_size;
    size_t offset = 0;
    int success = 1;
    while (success && offset < size) {
        ssize_t received = read(descriptor, bytes + offset, size - offset);
        if (received < 0 && errno == EINTR)
            continue;
        if (received <= 0)
            success = 0;
        else
            offset += received;
    }
    unsigned char extra;
    ssize_t trailing;
    do {
        trailing = read(descriptor, &extra, 1);
    } while (trailing < 0 && errno == EINTR);
    if (close(descriptor) || trailing != 0)
        success = 0;
    if (success) {
        texture->width = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
                         (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
        texture->height = (uint32_t)bytes[4] | (uint32_t)bytes[5] << 8 |
                          (uint32_t)bytes[6] << 16 | (uint32_t)bytes[7] << 24;
        success = texture->width >= 1 && texture->width <= 16 &&
                  texture->height >= 1 && texture->height <= 16 &&
                  size == 8 + texture->width * texture->height * 4;
    }
    if (!success) {
        fprintf(stderr, "texture requires dimensions 1..16 and exact tightly packed RGBA8 bytes: %s\n", path);
        return 0;
    }
    memcpy(texture->texels, bytes + 8, size - 8);
    return 1;
}

static struct triangle_vertex *read_mesh(const char *path, unsigned *count) {
    _Static_assert(sizeof(GLfloat) == 4, "mesh input requires 32-bit GL floats");
    int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    struct stat storage;
    if (descriptor < 0 || fstat(descriptor, &storage) || !S_ISREG(storage.st_mode) ||
        storage.st_size <= 0 || storage.st_size > 4096 * 28 || storage.st_size % 28 ||
        (storage.st_size / 28) % 3) {
        fprintf(stderr, "invalid mesh file (expected <=4096 triangle-list FP32 records): %s\n", path);
        if (descriptor >= 0)
            close(descriptor);
        return NULL;
    }
    size_t size = storage.st_size;
    unsigned vertices_count = size / 28;
    unsigned char *bytes = malloc(size);
    struct triangle_vertex *vertices = calloc(vertices_count, sizeof(*vertices));
    int success = bytes && vertices;
    size_t offset = 0;
    while (success && offset < size) {
        ssize_t received = read(descriptor, bytes + offset, size - offset);
        if (received < 0 && errno == EINTR)
            continue;
        if (received <= 0)
            success = 0;
        else
            offset += received;
    }
    unsigned char extra;
    ssize_t trailing;
    do {
        trailing = read(descriptor, &extra, 1);
    } while (trailing < 0 && errno == EINTR);
    int closed = close(descriptor);
    if (trailing != 0 || closed)
        success = 0;
    for (unsigned index = 0; success && index < vertices_count; index++) {
        GLfloat values[7];
        for (unsigned component = 0; component < 7; component++) {
            const unsigned char *source = bytes + index * 28 + component * 4;
            uint32_t word = (uint32_t)source[0] | (uint32_t)source[1] << 8 |
                            (uint32_t)source[2] << 16 | (uint32_t)source[3] << 24;
            memcpy(&values[component], &word, sizeof(word));
            if (!isfinite(values[component]))
                success = 0;
        }
        if (!values[4] && !values[5] && !values[6])
            success = 0;
        memcpy(vertices[index].position, values, sizeof(vertices[index].position));
        memcpy(vertices[index].normal, values + 4, sizeof(vertices[index].normal));
        vertices[index].color[0] = 1;
        vertices[index].color[1] = .5f;
        vertices[index].color[2] = .25f;
    }
    free(bytes);
    if (!success) {
        fprintf(stderr, "could not read finite mesh vertices with nonzero normals: %s\n", path);
        free(vertices);
        return NULL;
    }
    *count = vertices_count;
    return vertices;
}

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0) {
        perror(path);
        exit(1);
    }
    long length = ftell(file);
    if (length < 0 || length > 16 * 1024 * 1024 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "invalid or oversized shader file: %s\n", path);
        fclose(file);
        exit(1);
    }
    char *contents = malloc((size_t)length + 1);
    if (!contents || fread(contents, 1, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "could not read shader file: %s\n", path);
        free(contents);
        fclose(file);
        exit(1);
    }
    contents[length] = '\0';
    fclose(file);
    return contents;
}

static GLuint compile(GLenum type, const char *src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[4096] = {0};
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "shader compile failed:\n%s\n", log);
        exit(1);
    }
    return shader;
}

static void set_uniform_defaults(GLuint program, int mesh_input) {
    const GLfloat identity[] = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    };
    const char *matrices[] = {"u_mvp", "u_model"};
    for (unsigned index = 0; index < sizeof(matrices) / sizeof(matrices[0]); index++) {
        GLint location = glGetUniformLocation(program, matrices[index]);
        if (location >= 0)
            glUniformMatrix4fv(location, 1, GL_FALSE, identity);
    }
    GLint location = glGetUniformLocation(program, "u_tint");
    if (location >= 0)
        glUniform4f(location, .5f, .25f, 1, .5f);
    location = glGetUniformLocation(program, "u_bias");
    if (location >= 0)
        glUniform1f(location, .25f);
    location = glGetUniformLocation(program, "u_divisor");
    if (location >= 0)
        glUniform1f(location, 2);
    const char *positions[] = {"u_light_pos", "u_view_pos"};
    for (unsigned index = 0; index < sizeof(positions) / sizeof(positions[0]); index++) {
        location = glGetUniformLocation(program, positions[index]);
        if (location >= 0)
            glUniform3f(location, 0, 0, mesh_input ? -2 : 2);
    }
    location = glGetUniformLocation(program, "u_object_color");
    if (location >= 0)
        glUniform3f(location, .25f, .5f, .75f);
}

static int framebuffer_extent(const char *name, unsigned *extent) {
    const char *value = getenv(name);
    if (!value) {
        *extent = 16;
        return 1;
    }
    if (!value[0] || strlen(value) > 3) {
        fprintf(stderr, "%s must be a decimal integer in [1, 128]\n", name);
        return 0;
    }
    unsigned parsed = 0;
    for (const char *digit = value; *digit; digit++) {
        if (*digit < '0' || *digit > '9') {
            fprintf(stderr, "%s must be a decimal integer in [1, 128]\n", name);
            return 0;
        }
        parsed = parsed * 10 + (unsigned)(*digit - '0');
    }
    if (parsed < 1 || parsed > 128) {
        fprintf(stderr, "%s must be a decimal integer in [1, 128]\n", name);
        return 0;
    }
    *extent = parsed;
    return 1;
}

static int draw_repeats(unsigned *repeats) {
    const char *value = getenv("MGC_HOST_DRAW_REPEATS");
    *repeats = 1;
    if (!value)
        return 1;
    if (strlen(value) != 1 || value[0] < '1' || value[0] > '4') {
        fprintf(stderr, "MGC_HOST_DRAW_REPEATS must be a decimal integer in [1, 4]\n");
        return 0;
    }
    *repeats = (unsigned)(value[0] - '0');
    return 1;
}

static int clear_rgba(GLfloat components[4]) {
    const char *value = getenv("MGC_HOST_CLEAR_RGBA");
    for (unsigned component = 0; component < 4; component++)
        components[component] = 0;
    if (!value)
        return 1;
    if (!value[0] || strlen(value) > 128)
        goto invalid;
    const char *cursor = value;
    for (unsigned component = 0; component < 4; component++) {
        char *component_end = NULL;
        double parsed = strtod(cursor, &component_end);
        if (component_end == cursor || !isfinite(parsed) || parsed < 0 || parsed > 1 ||
            (component < 3 ? *component_end != ',' : *component_end != '\0'))
            goto invalid;
        for (const char *character = cursor; character < component_end; character++) {
            if (!(*character >= '0' && *character <= '9') && *character != '.' &&
                *character != '+' && *character != '-' && *character != 'e' && *character != 'E')
                goto invalid;
        }
        components[component] = (GLfloat)parsed;
        if (component < 3)
            cursor = component_end + 1;
    }
    return 1;
invalid:
    fprintf(stderr, "MGC_HOST_CLEAR_RGBA requires exactly four comma-separated finite decimal values in [0, 1]\n");
    return 0;
}

static int texture_wrap_value(const char *name, const char *value, GLenum *wrap) {
    if (!value || !strcmp(value, "clamp-to-edge"))
        *wrap = GL_CLAMP_TO_EDGE;
    else if (!strcmp(value, "repeat"))
        *wrap = GL_REPEAT;
    else if (!strcmp(value, "mirrored-repeat"))
        *wrap = GL_MIRRORED_REPEAT;
    else {
        fprintf(stderr, "%s must be clamp-to-edge or repeat or mirrored-repeat\n", name);
        return 0;
    }
    return 1;
}

static int draw_triangle(GLuint program, const char *readback_path, unsigned blend_mode,
                         int native_readback, unsigned width, unsigned height,
                         unsigned repeats, const GLfloat clear_color[4],
                         desktop_clear_depth_proc clear_depth) {
    const char *texture_mode = getenv("MGC_HOST_TEXTURE");
    const char *texture_path = getenv("MGC_HOST_TEXTURE_DATA");
    const char *texture_filter = getenv("MGC_HOST_TEXTURE_FILTER");
    const char *texture_wrap = getenv("MGC_HOST_TEXTURE_WRAP");
    const char *texture_wrap_s = getenv("MGC_HOST_TEXTURE_WRAP_S");
    const char *texture_wrap_t = getenv("MGC_HOST_TEXTURE_WRAP_T");
    struct host_texture texture;
    struct host_mipmaps mipmaps;
    int has_texture = texture_path && texture_path[0];
    if (texture_mode && strcmp(texture_mode, "10") &&
        (strlen(texture_mode) != 1 || texture_mode[0] < '1' || texture_mode[0] > '9')) {
        fprintf(stderr, "MGC_HOST_TEXTURE must be 1, 2, 3, 4, 5 or 6 (paired), 7 or 8 (per-axis), 9 or 10 (mipmapped)\n");
        return 0;
    }
    unsigned texture_version = texture_mode ? (!strcmp(texture_mode, "10") ? 10 : (unsigned)(texture_mode[0] - '0')) : 0;
    int mixed_mode = texture_version >= 7;
    int mipmap_mode = texture_version >= 9;
    int linear_mode = texture_version && !(texture_version % 2);
    GLenum mode_wrap = texture_version == 3 || texture_version == 4 ? GL_REPEAT :
                       texture_version == 5 || texture_version == 6 ? GL_MIRRORED_REPEAT : GL_CLAMP_TO_EDGE;
    if (texture_filter && strcmp(texture_filter, "nearest") && strcmp(texture_filter, "linear")) {
        fprintf(stderr, "MGC_HOST_TEXTURE_FILTER must be nearest or linear\n");
        return 0;
    }
    GLenum paired_wrap = mode_wrap;
    if (texture_wrap && !texture_wrap_value("MGC_HOST_TEXTURE_WRAP", texture_wrap, &paired_wrap))
        return 0;
    GLenum wrap_s = paired_wrap;
    GLenum wrap_t = paired_wrap;
    if (mixed_mode) {
        if (!texture_wrap_s || !texture_wrap_t) {
            fprintf(stderr, "MGC_HOST_TEXTURE versions 7..10 require explicit MGC_HOST_TEXTURE_WRAP_S/T\n");
            return 0;
        }
        if (!texture_wrap_value("MGC_HOST_TEXTURE_WRAP_S", texture_wrap_s, &wrap_s) ||
            !texture_wrap_value("MGC_HOST_TEXTURE_WRAP_T", texture_wrap_t, &wrap_t))
            return 0;
        if (texture_wrap && (paired_wrap != wrap_s || paired_wrap != wrap_t)) {
            fprintf(stderr, "MGC_HOST_TEXTURE_WRAP conflicts with explicit S/T wraps\n");
            return 0;
        }
    } else if (texture_wrap_s || texture_wrap_t) {
        fprintf(stderr, "MGC_HOST_TEXTURE_WRAP_S/T require MGC_HOST_TEXTURE version 7..10\n");
        return 0;
    }
    int linear_filter = texture_filter ? !strcmp(texture_filter, "linear") : linear_mode;
    if (mixed_mode && linear_filter != linear_mode) {
        fprintf(stderr, "texture filter/wrap must match MGC_HOST_TEXTURE version 7..10 opt-in\n");
        return 0;
    }
    if ((texture_mode || linear_filter || wrap_s != GL_CLAMP_TO_EDGE || wrap_t != GL_CLAMP_TO_EDGE) && !has_texture) {
        fprintf(stderr, "host texture draw requires explicit MGC_HOST_TEXTURE_DATA\n");
        return 0;
    }
    if (mipmap_mode && !extension_advertised((const char *)glGetString(GL_EXTENSIONS), "GL_ARB_shader_texture_lod")) {
        fprintf(stderr, "MGC_MIPMAP_EXTENSION_UNAVAILABLE: explicit desktop GLSL120 mipmaps require advertised GL_ARB_shader_texture_lod; no GLES3 or derivative fallback\n");
        return 0;
    }
    const char *compile_target = getenv("MGC_COMPILE_TARGET");
    int mgc_renderer = !strcmp((const char *)glGetString(GL_RENDERER), "mgc");
    const char *instancing_mode = getenv("MGC_HOST_INSTANCING");
    const char *instance_path = getenv("MGC_HOST_INSTANCE_DATA");
    int instancing = instancing_mode != NULL;
    struct host_instances instances;
    instance_divisor_proc attribute_divisor = NULL;
    instance_draw_proc draw_instanced = NULL;
    if ((instancing && strcmp(instancing_mode, "1")) ||
        (instancing != (instance_path && instance_path[0]))) {
        fprintf(stderr, "instancing requires explicit MGC_HOST_INSTANCING=1 and MGC_HOST_INSTANCE_DATA\n");
        return 0;
    }
    if (instancing && mgc_renderer && compile_target && strcmp(compile_target, "legacy-mgcsim-v1")) {
        fprintf(stderr, "host instancing is unsupported on hardware\n");
        return 0;
    }
    if (instancing && !read_instances(instance_path, &instances))
        return 0;
    if (instancing) {
        const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
        const char *alias = NULL;
        if (clear_depth && extension_advertised(extensions, "GL_ARB_instanced_arrays") &&
            extension_advertised(extensions, "GL_ARB_draw_instanced")) {
            alias = "ARB";
            attribute_divisor = (instance_divisor_proc)eglGetProcAddress("glVertexAttribDivisorARB");
            draw_instanced = (instance_draw_proc)eglGetProcAddress("glDrawArraysInstancedARB");
        } else if (!clear_depth && extension_advertised(extensions, "GL_ANGLE_instanced_arrays")) {
            alias = "ANGLE";
            attribute_divisor = (instance_divisor_proc)eglGetProcAddress("glVertexAttribDivisorANGLE");
            draw_instanced = (instance_draw_proc)eglGetProcAddress("glDrawArraysInstancedANGLE");
        } else if (!clear_depth && extension_advertised(extensions, "GL_EXT_instanced_arrays") &&
                   extension_advertised(extensions, "GL_EXT_draw_instanced")) {
            alias = "EXT";
            attribute_divisor = (instance_divisor_proc)eglGetProcAddress("glVertexAttribDivisorEXT");
            draw_instanced = (instance_draw_proc)eglGetProcAddress("glDrawArraysInstancedEXT");
        }
        if (!alias || !attribute_divisor || !draw_instanced) {
            fprintf(stderr, "instancing requires advertised GL_ANGLE_instanced_arrays or "
                            "GL_EXT_instanced_arrays + GL_EXT_draw_instanced on GLES2, or desktop "
                            "GL_ARB_instanced_arrays + GL_ARB_draw_instanced and both matching extension entry points\n");
            return 0;
        }
        fprintf(stderr, "MGC_HOST_INSTANCING_GL alias=%s count=%u offset_divisor=%u color_divisor=%u base_instance=0\n",
                alias, instances.count, instances.divisors[0], instances.divisors[1]);
    }
    if (blend_mode && mgc_renderer && compile_target && strcmp(compile_target, "legacy-mgcsim-v1")) {
        fprintf(stderr, "host blend is unsupported on hardware\n");
        return 0;
    }
    if (has_texture && mgc_renderer && compile_target && strcmp(compile_target, "legacy-mgcsim-v1")) {
        fprintf(stderr, "host textures are unsupported on hardware\n");
        return 0;
    }
    if (has_texture && mgc_renderer &&
        (!texture_mode || linear_filter != linear_mode ||
         (!mixed_mode && (wrap_s != mode_wrap || wrap_t != mode_wrap)))) {
        fprintf(stderr, "texture filter/wrap must match legacy MGC_HOST_TEXTURE version 1..10 opt-in\n");
        return 0;
    }
    if (has_texture && (mipmap_mode ? !read_mipmaps(texture_path, &mipmaps) : !read_texture(texture_path, &texture)))
        return 0;
    const struct triangle_vertex default_vertices[] = {
        {{-.75f, -.75f, 0, 1}, {1, .5f, .25f}, {0, 0, 1}},
        {{ .75f, -.75f, 0, 1}, {1, .5f, .25f}, {0, 0, 1}},
        {{0, .75f, 0, 1}, {1, .5f, .25f}, {0, 0, 1}},
    };
    const struct triangle_vertex *vertices = default_vertices;
    struct triangle_vertex *mesh_vertices = NULL;
    unsigned vertex_count = 3;
    const char *mesh_path = getenv("MGC_HOST_MESH");
    if (mesh_path && mesh_path[0]) {
        mesh_vertices = read_mesh(mesh_path, &vertex_count);
        if (!mesh_vertices)
            return 0;
        vertices = mesh_vertices;
    }
    if (instancing && vertex_count > 4096 / instances.count) {
        fprintf(stderr, "instancing exceeds aggregate vertex_count * instances <= 4096\n");
        free(mesh_vertices);
        return 0;
    }
    GLuint framebuffer = 0;
    GLuint colorbuffer = 0;
    GLuint color_renderbuffer = 0;
    GLuint depthbuffer = 0;
    GLuint vertexbuffer = 0;
    GLuint instance_buffers[2] = {0, 0};
    GLuint sample_texture = 0;
    const char *depth_mode = getenv("MGC_HOST_DEPTH_TEST");
    int depth_test = depth_mode && !strcmp(depth_mode, "1");
    int success = 0;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    if (native_readback) {
        glGenRenderbuffers(1, &color_renderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, color_renderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8_OES, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_renderbuffer);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            fprintf(stderr, "native %ux%u RGBA8 renderbuffer allocation failed: 0x%x\n", width, height, error);
            goto done;
        }
    } else {
        glGenTextures(1, &colorbuffer);
        glBindTexture(GL_TEXTURE_2D, colorbuffer);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorbuffer, 0);
    }
    if (depth_test) {
        glGenRenderbuffers(1, &depthbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbuffer);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            fprintf(stderr, "draw %ux%u GL_DEPTH_COMPONENT16 depth attachment failed: 0x%x\n", width, height, error);
            goto done;
        }
    }
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    GLenum framebuffer_error = glGetError();
    if (status != GL_FRAMEBUFFER_COMPLETE || framebuffer_error != GL_NO_ERROR) {
        if (depth_test)
            fprintf(stderr, "draw framebuffer is incomplete (%ux%u GL_DEPTH_COMPONENT16 depth enabled): 0x%x\n",
                    width, height, status);
        else
            fprintf(stderr, "draw framebuffer is incomplete: 0x%x\n", status);
        if (framebuffer_error != GL_NO_ERROR)
            fprintf(stderr, "draw framebuffer setup failed: 0x%x\n", framebuffer_error);
        goto done;
    }
    glViewport(0, 0, width, height);
    if (readback_path || getenv("MGC_HOST_FEATURES")) {
        glDisable(GL_DITHER);
        glDisable(GL_SCISSOR_TEST);
    }
    if (native_readback) {
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    }
    const GLenum blend_equations[] = {GL_FUNC_ADD, GL_FUNC_ADD, GL_FUNC_ADD, GL_FUNC_SUBTRACT,
                                      GL_FUNC_REVERSE_SUBTRACT, GL_MIN_EXT, GL_MAX_EXT};
    glBlendEquation(blend_equations[blend_mode]);
    if (blend_mode) {
        glEnable(GL_BLEND);
        if (blend_mode == 1)
            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        else
            glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
    } else {
        glDisable(GL_BLEND);
    }
    GLenum blend_error = glGetError();
    if (blend_error != GL_NO_ERROR) {
        fprintf(stderr, "actual GL blend equation mode %u setup failed: 0x%x\n", blend_mode, blend_error);
        goto done;
    }
    if (depth_test) {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    glDisable(GL_CULL_FACE);
    glUseProgram(program);
    set_uniform_defaults(program, mesh_vertices != NULL);
    if (has_texture) {
        GLint location = glGetUniformLocation(program, "u_texture");
        if (location < 0) {
            fprintf(stderr, "texture input requires an active u_texture sampler uniform\n");
            goto done;
        }
        glActiveTexture(GL_TEXTURE0);
        glGenTextures(1, &sample_texture);
        glBindTexture(GL_TEXTURE_2D, sample_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        mipmap_mode ? (linear_filter ? GL_LINEAR_MIPMAP_NEAREST : GL_NEAREST_MIPMAP_NEAREST) :
                                      (linear_filter ? GL_LINEAR : GL_NEAREST));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear_filter ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            fprintf(stderr, "texture min/mag %s S=0x%x T=0x%x sampler setup failed: 0x%x\n",
                    linear_filter ? "linear" : "nearest", wrap_s, wrap_t, error);
            goto done;
        }
        if (mipmap_mode) {
            for (unsigned level = 0; level < mipmaps.level_count; level++) {
                glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA, mipmaps.widths[level], mipmaps.heights[level],
                             0, GL_RGBA, GL_UNSIGNED_BYTE, mipmaps.texels + mipmaps.offsets[level]);
                error = glGetError();
                if (error != GL_NO_ERROR) {
                    fprintf(stderr, "texture RGBA8 mip level %u upload failed: 0x%x\n", level, error);
                    goto done;
                }
            }
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width, texture.height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, texture.texels);
        }
        glUniform1i(location, 0);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            fprintf(stderr, "texture RGBA8 upload failed: 0x%x\n", error);
            goto done;
        }
    }
    if (readback_path || getenv("MGC_HOST_CLEAR_RGBA")) {
        glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
        if (depth_test) {
            if (clear_depth)
                clear_depth(1.0);
            else
                glClearDepthf(1);
        }
        glClear(GL_COLOR_BUFFER_BIT | (depth_test ? GL_DEPTH_BUFFER_BIT : 0));
    }
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(*vertices), vertices, GL_STATIC_DRAW);
    if (instancing) {
        glGenBuffers(2, instance_buffers);
        for (unsigned stream = 0; stream < 2; stream++) {
            glBindBuffer(GL_ARRAY_BUFFER, instance_buffers[stream]);
            glBufferData(GL_ARRAY_BUFFER, instances.records[stream] * instances.components[stream] * sizeof(GLfloat),
                         instances.values[stream], GL_STATIC_DRAW);
        }
    }
    GLint attributes = 0;
    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attributes);
    for (GLint index = 0; index < attributes; index++) {
        char name[256] = {0};
        GLsizei length = 0;
        GLint size = 0;
        GLenum type = 0;
        glGetActiveAttrib(program, index, sizeof(name), &length, &size, &type, name);
        GLint location = glGetAttribLocation(program, name);
        if (instancing && ((!clear_depth && !strcmp(name, "gl_InstanceIDEXT")) ||
            (clear_depth && (!strcmp(name, "gl_InstanceIDARB") || !strcmp(name, "gl_InstanceID"))))) {
            if (size != 1 || type != GL_INT || location != -1 || length >= (GLsizei)sizeof(name) - 1) {
                fprintf(stderr, "unsupported instancing system value: %s type=0x%x location=%d\n", name, type, location);
                goto done;
            }
            continue;
        }
        if (size != 1 || location < 0 || length >= (GLsizei)sizeof(name) - 1 ||
            (type != GL_FLOAT && type != GL_FLOAT_VEC2 &&
             type != GL_FLOAT_VEC3 && type != GL_FLOAT_VEC4)) {
            fprintf(stderr, "unsupported draw attribute: %s\n", name);
            goto done;
        }
        size_t offset;
        GLint components;
        if (!strcmp(name, "a_instance_offset") || !strcmp(name, "a_instance_color")) {
            unsigned stream = !strcmp(name, "a_instance_color");
            if (!instancing || (type != GL_FLOAT_VEC3 && type != GL_FLOAT_VEC4) ||
                instances.components[stream] != (unsigned)(type == GL_FLOAT_VEC4 ? 4 : 3)) {
                fprintf(stderr, "instance attribute requires matching explicit vec3/vec4 records: %s\n", name);
                goto done;
            }
            glBindBuffer(GL_ARRAY_BUFFER, instance_buffers[stream]);
            glEnableVertexAttribArray(location);
            glVertexAttribPointer(location, instances.components[stream], GL_FLOAT, GL_FALSE,
                                  instances.components[stream] * sizeof(GLfloat), NULL);
            attribute_divisor(location, instances.divisors[stream]);
            continue;
        }
        if (!strcmp(name, "a_position")) {
            offset = offsetof(struct triangle_vertex, position);
            components = 4;
        } else if (!strcmp(name, "a_color")) {
            offset = offsetof(struct triangle_vertex, color);
            components = 3;
        } else if (!strcmp(name, "a_normal")) {
            offset = offsetof(struct triangle_vertex, normal);
            components = 3;
        } else {
            fprintf(stderr, "no triangle data for draw attribute: %s\n", name);
            goto done;
        }
        glEnableVertexAttribArray(location);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(location, components, GL_FLOAT, GL_FALSE,
                              sizeof(struct triangle_vertex), (const void *)offset);
        if (instancing)
            attribute_divisor(location, 0);
    }
    for (unsigned iteration = 0; iteration < repeats; iteration++) {
        if (instancing)
            draw_instanced(GL_TRIANGLES, 0, vertex_count, instances.count);
        else
            glDrawArrays(GL_TRIANGLES, 0, vertex_count);
        glFinish();
    }
    if (readback_path) {
        GLubyte pixels[128 * 128 * 4];
        size_t pixel_size = width * height * 4;
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            fprintf(stderr, "%s readback failed: 0x%x\n", native_readback ? "native GL" : "reference", error);
            goto done;
        }
        int descriptor = open(readback_path, O_WRONLY | O_CREAT | O_TRUNC |
                              O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0600);
        struct stat storage;
        if (descriptor < 0 || fstat(descriptor, &storage) || !S_ISREG(storage.st_mode)) {
            fprintf(stderr, "%s output must be an ordinary file: %s\n",
                    native_readback ? "native GL" : "reference", readback_path);
            if (descriptor >= 0)
                close(descriptor);
            goto done;
        }
        FILE *output = fdopen(descriptor, "wb");
        if (!output) {
            perror(readback_path);
            close(descriptor);
            goto done;
        }
        size_t written = fwrite(pixels, 1, pixel_size, output);
        int closed = fclose(output);
        if (written != pixel_size || closed) {
            fprintf(stderr, "could not write %s pixels: %s\n",
                    native_readback ? "native GL" : "reference", readback_path);
            goto done;
        }
        fprintf(stderr, "wrote %ux%u bottom-left RGBA8 %s pixels\n", width, height,
                native_readback ? "native GL readback" : "reference");
    } else {
        fprintf(stderr, "submitted %u GL triangle-list vertices; capture only, no rendering or presentation\n",
                vertex_count);
    }
    success = 1;

done:
    glDeleteBuffers(2, instance_buffers);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vertexbuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &framebuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glDeleteRenderbuffers(1, &depthbuffer);
    glDeleteRenderbuffers(1, &color_renderbuffer);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &sample_texture);
    glDeleteTextures(1, &colorbuffer);
    free(mesh_vertices);
    return success;
}

int main(int argc, char **argv) {
    int draw_mode = argc == 4 && !strcmp(argv[3], "--draw");
    const char *reference_path = argc == 5 && !strcmp(argv[3], "--reference") ? argv[4] : NULL;
    const char *native_path = argc == 5 && !strcmp(argv[3], "--host-execute") ? argv[4] : NULL;
    if (argc != 3 && !draw_mode && !reference_path && !native_path) {
        fprintf(stderr, "usage: %s <vert.glsl> <frag.glsl> [--draw | --reference <raw.rgba8> | --host-execute <raw.rgba8>]\n", argv[0]);
        return 1;
    }
    const char *profile = getenv("MGC_HOST_GLSL_PROFILE");
    const char *texture_version = getenv("MGC_HOST_TEXTURE");
    int desktop_profile = profile && !strcmp(profile, "desktop-120");
    int mipmap_version = texture_version && (!strcmp(texture_version, "9") || !strcmp(texture_version, "10"));
    if ((profile && !desktop_profile) || desktop_profile != mipmap_version) {
        fprintf(stderr, "MGC_HOST_GLSL_PROFILE must be explicit desktop-120 only with MGC_HOST_TEXTURE 9 or 10; no implicit profile fallback\n");
        return 1;
    }
    const char *profile_target = getenv("MGC_COMPILE_TARGET");
    if (desktop_profile && (!profile_target || strcmp(profile_target, "legacy-mgcsim-v1"))) {
        fprintf(stderr, "desktop-120 mipmaps require the explicit legacy-mgcsim-v1 compiler target, not hardware\n");
        return 1;
    }
    unsigned width = 16;
    unsigned height = 16;
    unsigned repeats = 1;
    if (native_path || reference_path) {
        if (!draw_repeats(&repeats))
            return 1;
    } else if (unsetenv("MGC_HOST_DRAW_REPEATS")) {
        perror("clearing inherited MGC_HOST_DRAW_REPEATS");
        return 1;
    }
    if (native_path) {
        const char *executor_socket = getenv("MGC_HOST_EXECUTOR_SOCKET");
        const char *depth_mode = getenv("MGC_HOST_DEPTH_TEST");
        const char *compile_target = getenv("MGC_COMPILE_TARGET");
        if (!executor_socket || !executor_socket[0]) {
            fprintf(stderr, "--host-execute requires explicit MGC_HOST_EXECUTOR_SOCKET\n");
            return 1;
        }
        if (compile_target && strcmp(compile_target, "legacy-mgcsim-v1")) {
            fprintf(stderr, "--host-execute supports the legacy compiler target only, not hardware\n");
            return 1;
        }
        if (depth_mode && strcmp(depth_mode, "0") && strcmp(depth_mode, "1")) {
            fprintf(stderr, "--host-execute depth state must be explicit 0 or 1 (LESS/write)\n");
            return 1;
        }
        if (!framebuffer_extent("MGC_HOST_WIDTH", &width) ||
            !framebuffer_extent("MGC_HOST_HEIGHT", &height))
            return 1;
    } else if (unsetenv("MGC_HOST_EXECUTOR_SOCKET")) {
        perror("clearing inherited MGC_HOST_EXECUTOR_SOCKET");
        return 1;
    }
    const char *blend_mode = getenv("MGC_HOST_BLEND");
    const char *blend_modes[] = {"disabled", "source-over", "additive", "subtract", "reverse-subtract",
                                "minimum", "maximum"};
    unsigned blend = 0;
    if (blend_mode) {
        for (blend = 0; blend < sizeof(blend_modes) / sizeof(blend_modes[0]); blend++) {
            if (!strcmp(blend_mode, blend_modes[blend]))
                break;
        }
        if (blend == sizeof(blend_modes) / sizeof(blend_modes[0])) {
            fprintf(stderr, "unsupported MGC_HOST_BLEND (expected disabled/source-over/additive/subtract/reverse-subtract/minimum/maximum): %s\n", blend_mode);
            return 1;
        }
    }
    GLfloat clear_color[4];
    if (!clear_rgba(clear_color))
        return 1;

    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL) ||
        !eglBindAPI(desktop_profile ? EGL_OPENGL_API : EGL_OPENGL_ES_API)) {
        fprintf(stderr, "EGL initialization failed: 0x%x\n", eglGetError());
        return 1;
    }

    EGLint cfg_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, desktop_profile ? EGL_OPENGL_BIT : EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE,
    };
    EGLConfig cfg;
    EGLint num_cfg;
    if (!eglChooseConfig(dpy, cfg_attribs, &cfg, 1, &num_cfg) || num_cfg != 1) {
        fprintf(stderr, "EGL configuration failed: 0x%x\n", eglGetError());
        eglTerminate(dpy);
        return 1;
    }

    EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLint desktop_attribs[] = { EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, desktop_profile ? desktop_attribs : ctx_attribs);
    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "EGL context creation failed: 0x%x\n", eglGetError());
        if (ctx != EGL_NO_CONTEXT)
            eglDestroyContext(dpy, ctx);
        eglTerminate(dpy);
        return 1;
    }

    fprintf(stderr, "GL_RENDERER: %s\n", (const char *)glGetString(GL_RENDERER));
    desktop_clear_depth_proc clear_depth = NULL;
    if (desktop_profile) {
        const char *version = (const char *)glGetString(GL_VERSION);
        const char *language = (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
        clear_depth = (desktop_clear_depth_proc)eglGetProcAddress("glClearDepth");
        desktop_clamp_color_proc clamp_color = (desktop_clamp_color_proc)eglGetProcAddress("glClampColorARB");
        fprintf(stderr, "MGC_HOST_GLSL_PROFILE: desktop-120; GL_VERSION: %s; GL_SHADING_LANGUAGE_VERSION: %s\n",
                version ? version : "unavailable", language ? language : "unavailable");
        if (!version || strstr(version, "OpenGL ES") || !language || !clear_depth || !clamp_color ||
            !extension_advertised((const char *)glGetString(GL_EXTENSIONS), "GL_ARB_shader_texture_lod") ||
            !extension_advertised((const char *)glGetString(GL_EXTENSIONS), "GL_ARB_color_buffer_float")) {
            fprintf(stderr, "MGC_MIPMAP_EXTENSION_UNAVAILABLE: desktop GLSL120 requires actual desktop OpenGL, glClearDepth, GL_ARB_shader_texture_lod and explicit unclamped vertex-color control\n");
            eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroyContext(dpy, ctx);
            eglTerminate(dpy);
            return 1;
        }
        clamp_color(0x891A, GL_FALSE);
        if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr, "desktop mipmaps could not disable fixed-function vertex-color clamp\n");
            eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroyContext(dpy, ctx);
            eglTerminate(dpy);
            return 1;
        }
    }
    if (reference_path && strcmp((const char *)glGetString(GL_RENDERER), "softpipe")) {
        fprintf(stderr, "--reference requires the softpipe CPU renderer\n");
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(dpy, ctx);
        eglTerminate(dpy);
        return 1;
    }
    if (native_path && strcmp((const char *)glGetString(GL_RENDERER), "mgc")) {
        fprintf(stderr, "--host-execute requires the mgc CPU host-executor renderer, not softpipe or hardware\n");
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(dpy, ctx);
        eglTerminate(dpy);
        return 1;
    }

    char *vsrc = read_file(argv[1]);
    char *fsrc = read_file(argv[2]);
    if (desktop_profile && (!desktop_120_source(vsrc) || !desktop_120_source(fsrc))) {
        fprintf(stderr, "desktop-120 mipmaps require explicit #version 120 in both GLSL sources\n");
        free(vsrc);
        free(fsrc);
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(dpy, ctx);
        eglTerminate(dpy);
        return 1;
    }
    GLuint vs = compile(GL_VERTEX_SHADER, vsrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsrc);
    free(vsrc);
    free(fsrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    if (draw_mode || reference_path || native_path) {
        glBindAttribLocation(prog, 0, "a_position");
        glBindAttribLocation(prog, 1, "a_color");
        glBindAttribLocation(prog, 2, "a_normal");
        if (getenv("MGC_HOST_INSTANCING")) {
            glBindAttribLocation(prog, 3, "a_instance_offset");
            glBindAttribLocation(prog, 4, "a_instance_color");
        }
    }
    glLinkProgram(prog);

    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    char log[4096] = {0};
    glGetProgramInfoLog(prog, sizeof(log), NULL, log);
    if (log[0])
        fprintf(stderr, "%s\n", log);

    if (linked && (draw_mode || reference_path || native_path) &&
        !draw_triangle(prog, native_path ? native_path : reference_path, blend,
                       native_path != NULL, width, height, repeats, clear_color, clear_depth))
        linked = 0;

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL error: 0x%x\n", error);
        linked = 0;
    }
    if (linked && native_path)
        printf("{\"native_gl_readback\":true,\"fpga_execution\":false,\"width\":%u,\"height\":%u,"
               "\"format\":\"R8G8B8A8_UNORM\",\"pixel_origin\":\"bottom-left\",\"repeats\":%u}\n",
               width, height, repeats);
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);

    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglTerminate(dpy);
    return linked ? 0 : 1;
}
