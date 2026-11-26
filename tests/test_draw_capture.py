"""Bounded C transport checks and freshly linked, actual GL draw captures."""

import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import struct
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
DRIVER = ROOT / "mesa/src/gallium/drivers/mgc"
SOURCE = r'''
#include <assert.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include "compiler/glsl_types.h"
#include "util/ralloc.h"
#include "util/u_inlines.h"
#include "mgc_compiler.h"
#include "mgc_context.h"
#include "mgc_public.h"
#include "mgc_resource.h"
#include "mgc_screen.h"

static void
bind_transport_shader(struct pipe_context *pipe, unsigned stage, bool hardware)
{
   struct mgc_shader_state *state = calloc(1, sizeof(*state));
   assert(state);
   pipe_reference_init(&state->reference, 1);
   struct mgc_shader *shader = rzalloc(NULL, struct mgc_shader);
   assert(shader);
   state->shader = shader;
   shader->abi = hardware ? mgc_default_hardware_compile_options() : mgc_default_compile_options();
   shader->abi_version = hardware ? MGC_HARDWARE_ABI_VERSION : MGC_ABI_VERSION;
   shader->program_format = hardware ? "fp16-hardware-v1" : "legacy-mgcsim-v1";
   shader->asm_text = ralloc_asprintf(shader, "; mgc %s backend, ABI=%u, stage=%s\nstop_core\n",
      hardware ? "fp16-hardware-v1" : "legacy-simulator", shader->abi_version,
      stage ? "fragment" : "vertex");
   shader->layout_json = ralloc_asprintf(shader,
      "{\"version\":1,\"program_format\":\"%s\",\"shader_abi_version\":%u,"
      "\"stage\":\"%s\",\"address_units\":\"%s\","
      "\"bases\":{\"input\":%u,\"uniform\":%u,\"vertex\":%u,\"fragment\":%u},"
      "\"inputs\":[],\"outputs\":[],\"uniforms\":[],\"supported\":true,\"errors\":[]}",
      shader->program_format, shader->abi_version, stage ? "fragment" : "vertex",
      hardware ? "bytes" : "words", shader->abi.input_base, shader->abi.uniform_base,
      shader->abi.vertex_output_base, shader->abi.fragment_output_base);
   assert(shader->asm_text && shader->layout_json);
   if (stage) {
      pipe->bind_fs_state(pipe, state);
      pipe->delete_fs_state(pipe, state);
   } else {
      pipe->bind_vs_state(pipe, state);
      pipe->delete_vs_state(pipe, state);
   }
}

struct consumer {
   struct pipe_context *pipe;
   unsigned count;
   bool reenter;
};

static bool
consume(void *data, const struct mgc_draw_artifact *artifact)
{
   struct consumer *consumer = data;
   consumer->count++;
   assert(artifact->layout[0] == mgc_context(consumer->pipe)->shaders[0]->shader->layout_json);
   assert(artifact->layout[1] == mgc_context(consumer->pipe)->shaders[1]->shader->layout_json);
   if (consumer->reenter)
      consumer->pipe->draw_vbo(consumer->pipe, artifact->info, 0, NULL, artifact->draw, 1);
   return true;
}

static void
replace_layout(struct mgc_shader *shader, const char *text)
{
   shader->layout_json = ralloc_strdup(shader, text);
   assert(shader->layout_json);
}

int
main(int argc, char **argv)
{
   assert(argc == 2);
   const char *test = argv[1];
   glsl_type_singleton_init_or_ref();
   struct pipe_screen *screen = mgc_create_screen(NULL);
   assert(screen);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   assert(pipe);
   bool hardware = !strcmp(test, "hardware");
   bind_transport_shader(pipe, 0, hardware);
   bind_transport_shader(pipe, 1, hardware);
   struct mgc_context *ctx = mgc_context(pipe);
   struct mgc_shader *vertex_shader = ctx->shaders[0]->shader;
   struct mgc_shader *fragment_shader = ctx->shaders[1]->shader;
   unsigned bytes = !strcmp(test, "buffer_limit") ? MGC_HOST_CAPTURE_MAX_BUFFER_BYTES + 5 : 40;
   struct pipe_resource *resource = pipe_buffer_create(screen, PIPE_BIND_VERTEX_BUFFER,
                                                       PIPE_USAGE_DEFAULT, bytes);
   assert(resource);
   const float vertices[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
   memcpy((char *)mgc_resource(resource)->data + 4, vertices, sizeof(vertices));
   struct pipe_vertex_buffer buffers[MGC_RUNTIME_VERTEX_BUFFERS] = {{.buffer_offset = 4,
                                                                    .buffer.resource = resource}};
   unsigned buffer_count = !strcmp(test, "data_limit") ? 5 : 1;
   if (buffer_count > 1) {
      pipe_resource_reference(&buffers[0].buffer.resource, NULL);
      for (unsigned index = 0; index < buffer_count; index++) {
         buffers[index].buffer_offset = 0;
         buffers[index].buffer.resource = pipe_buffer_create(screen, PIPE_BIND_VERTEX_BUFFER,
            PIPE_USAGE_DEFAULT, MGC_HOST_CAPTURE_MAX_BUFFER_BYTES);
         assert(buffers[index].buffer.resource);
      }
   }
   pipe->set_vertex_buffers(pipe, buffer_count, buffers);
   struct pipe_vertex_element elements[MGC_RUNTIME_VERTEX_BUFFERS] = {{
      .src_format = PIPE_FORMAT_R32G32B32_FLOAT, .src_stride = 12, .vertex_buffer_index = 0,
   }};
   unsigned element_count = buffer_count;
   for (unsigned index = 1; index < element_count; index++) {
      elements[index] = elements[0];
      elements[index].vertex_buffer_index = index;
   }
   if (!strcmp(test, "format"))
      elements[0].src_format = PIPE_FORMAT_R8G8B8A8_UNORM;
   if (!strcmp(test, "divisor"))
      elements[0].instance_divisor = 1;
   if (!strcmp(test, "stride")) {
      elements[1] = elements[0];
      elements[1].src_stride = 8;
      element_count = 2;
   }
   if (!strcmp(test, "sparse"))
      element_count = 0;
   void *element_state = pipe->create_vertex_elements_state(pipe, element_count, elements);
   assert(element_state);
   pipe->bind_vertex_elements_state(pipe, element_state);
   pipe->delete_vertex_elements_state(pipe, element_state);
   uint32_t constants[] = {0x3f800000, 0x3f000000, 0x3e800000, 0x3f800000};
   struct pipe_constant_buffer constant = {.user_buffer = constants, .buffer_size = sizeof(constants)};
   pipe->set_constant_buffer(pipe, PIPE_SHADER_VERTEX, 0, false, &constant);
   constants[0] = 0;
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
   struct pipe_draw_start_count_bias draws[] = {{0, 3, 0}, {0, 3, 0}};
   unsigned draw_count = !strcmp(test, "batch") ? 2 : 1;
   info.increment_draw_id = draw_count > 1;
   if (!strcmp(test, "indexed1") || !strcmp(test, "indexed2") || !strcmp(test, "indexed4")) {
      info.index_size = test[7] - '0';
      struct pipe_resource *indices = pipe_buffer_create(screen, PIPE_BIND_INDEX_BUFFER,
         PIPE_USAGE_DEFAULT, 4 * info.index_size);
      assert(indices);
      for (unsigned index = 0; index < 4; index++) {
         uint32_t value = index ? index : 99;
         memcpy((char *)mgc_resource(indices)->data + index * info.index_size, &value, info.index_size);
      }
      info.index.resource = indices;
      info.take_index_buffer_ownership = true;
      draws[0].start = 1;
      draws[0].index_bias = -1;
   }
   if (!strcmp(test, "range"))
      draws[0].count = 4;
   if (!strcmp(test, "instances"))
      info.instance_count = 2;
   if (!strcmp(test, "start_instance"))
      info.start_instance = 1;
   if (!strcmp(test, "restart"))
      info.primitive_restart = true;
   if (!strcmp(test, "mode"))
      info.mode = MESA_PRIM_LINES;
   if (!strcmp(test, "empty"))
      draws[0].count = 0;
   if (!strcmp(test, "missing_layout"))
      vertex_shader->layout_json = NULL;
   if (!strcmp(test, "missing_assembly"))
      vertex_shader->asm_text = NULL;
   if (!strcmp(test, "missing_format"))
      vertex_shader->program_format = NULL;
   if (!strcmp(test, "abi_mismatch"))
      fragment_shader->abi_version = 2;
   if (!strcmp(test, "tag_mismatch"))
      fragment_shader->program_format = "fp16-hardware-v1";
   if (!strcmp(test, "target_mismatch"))
      fragment_shader->abi.target = MGC_TARGET_FP16_HARDWARE_V1;
   if (!strcmp(test, "assembly_tag"))
      vertex_shader->asm_text = ralloc_strdup(vertex_shader, "stop_core\n");
   if (!strcmp(test, "base_mismatch"))
      vertex_shader->abi.input_base++;
   if (!strcmp(test, "layout_limit") || !strcmp(test, "assembly_limit")) {
      char *large = ralloc_array(vertex_shader, char, MGC_HOST_CAPTURE_MAX_TEXT_BYTES + 2);
      assert(large);
      memset(large, ' ', MGC_HOST_CAPTURE_MAX_TEXT_BYTES + 1);
      large[MGC_HOST_CAPTURE_MAX_TEXT_BYTES + 1] = 0;
      if (!strcmp(test, "layout_limit"))
         vertex_shader->layout_json = large;
      else
         vertex_shader->asm_text = large;
   }
   if (!strcmp(test, "layout_array"))
      replace_layout(vertex_shader, "[]");
   if (!strcmp(test, "invalid_json"))
      replace_layout(vertex_shader, "{\"version\":1,}");
   if (!strcmp(test, "extra_json"))
      vertex_shader->layout_json = ralloc_asprintf(vertex_shader, "%s{}", vertex_shader->layout_json);
   if (!strcmp(test, "duplicate_version"))
      vertex_shader->layout_json = ralloc_asprintf(vertex_shader, "{\"version\":1,%s",
                                                   vertex_shader->layout_json + 1);
   if (!strcmp(test, "unsupported_layout")) {
      char *supported = strstr(vertex_shader->layout_json, "\"supported\":true");
      assert(supported);
      memcpy(supported + strlen("\"supported\":"), "null", 4);
   }
   if (!strcmp(test, "bad_utf8"))
      replace_layout(vertex_shader, "{\"name\":\"\xc0\x80\"}");
   if (!strcmp(test, "bad_surrogate"))
      replace_layout(vertex_shader, "{\"name\":\"\\ud800\"}");
   if (!strcmp(test, "depth_limit")) {
      char nested[256];
      unsigned cursor = 0;
      memcpy(nested, "{\"nested\":", 10);
      cursor += 10;
      for (unsigned depth = 0; depth < 40; depth++)
         nested[cursor++] = '[';
      nested[cursor++] = '0';
      for (unsigned depth = 0; depth < 40; depth++)
         nested[cursor++] = ']';
      nested[cursor++] = '}';
      nested[cursor] = 0;
      replace_layout(vertex_shader, nested);
   }
   struct pipe_transfer *transfer = NULL;
   if (!strcmp(test, "mapped")) {
      struct pipe_box box = {.width = 4, .height = 1, .depth = 1};
      assert(pipe->buffer_map(pipe, ctx->vertex_buffers[0].buffer.resource,
                             0, PIPE_MAP_WRITE, &box, &transfer));
   }
   struct consumer consumer = {.pipe = pipe, .reenter = !strcmp(test, "reentry")};
   if (!strcmp(test, "custom") || !strcmp(test, "reentry")) {
      mgc_context_set_artifact_callback(pipe, consume, &consumer);
      mgc_init_draw_capture(pipe);
   }
   if (!strcmp(test, "disabled"))
      mgc_context_set_artifact_callback(pipe, NULL, NULL);
   if (!strcmp(test, "write_failure")) {
      signal(SIGXFSZ, SIG_IGN);
      struct rlimit limit = {.rlim_cur = 32, .rlim_max = 32};
      assert(!setrlimit(RLIMIT_FSIZE, &limit));
   }
   pipe->draw_vbo(pipe, &info, 7, NULL, draws, draw_count);
   if (transfer)
      pipe->buffer_unmap(pipe, transfer);
   if (!strcmp(test, "immutable")) {
      memset(mgc_resource(ctx->vertex_buffers[0].buffer.resource)->data, 0xee, bytes);
      memset(mgc_resource(ctx->constants[0][0].buffer)->data, 0xee, sizeof(constants));
   }
   struct mgc_runtime_status status;
   mgc_context_get_runtime_status(pipe, &status);
   printf("{\"accepted\":%" PRIu64 ",\"rejected\":%" PRIu64 ",\"sequence\":%" PRIu64
          ",\"error\":%d,\"callbacks\":%u}\n", status.accepted_draws, status.rejected_draws,
          status.sequence, status.error, consumer.count);
   pipe->destroy(pipe);
   assert(mgc_screen(screen)->allocated_bytes == 0);
   screen->destroy(screen);
   glsl_type_singleton_decref();
   return 0;
}
'''


def checked(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, timeout=120, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def fresh_binaries(tmp_path_factory):
    if not (BUILD / "compile_commands.json").is_file():
        pytest.skip("configure Mesa with the mgc Gallium driver first")
    environment = dict(os.environ)
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    ninja = shutil.which("ninja", path=environment["PATH"])
    if not ninja:
        pytest.skip("ninja is needed for a fresh Mesa build")
    libraries = list((BUILD / "src/gallium/targets/dri").glob("libgallium-*.so"))
    assert len(libraries) == 1
    checked([ninja, "-C", str(BUILD), str(libraries[0].relative_to(BUILD)),
             "src/gallium/drivers/mgc/mgc_test_arith"], env=environment)
    temporary = tmp_path_factory.mktemp("fresh-draw-capture")
    entry = next(item for item in json.loads((BUILD / "compile_commands.json").read_text())
                 if item["file"].endswith("mgc_test_arith.c"))
    arguments = shlex.split(entry["command"])
    flags = []
    cursor = 1
    while cursor < len(arguments):
        argument = arguments[cursor]
        if argument in ("-o", "-c", "-MF", "-MQ", "-MT"):
            cursor += 2
        elif argument == "-MD":
            cursor += 1
        else:
            flags.append(argument)
            cursor += 1
    source = temporary / "capture.c"
    source.write_text(SOURCE)
    object_file = temporary / "capture.o"
    sanitized = environment.get("MGC_RUNTIME_SANITIZE") == "1"
    if sanitized:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    checked([arguments[0], *flags, "-I" + str(DRIVER), "-o", str(object_file),
             "-c", str(source)], cwd=BUILD)
    commands = checked([ninja, "-C", str(BUILD), "-t", "commands",
                        "src/gallium/drivers/mgc/mgc_test_arith"], env=environment)
    link = shlex.split(commands.stdout.strip().splitlines()[-1])
    helper = temporary / "capture"
    link[link.index("-o") + 1] = str(helper)
    original = "src/gallium/drivers/mgc/mgc_test_arith.p/mgc_test_arith.c.o"
    link[link.index(original)] = str(object_file)
    if sanitized:
        driver_objects = []
        for filename in ("mgc_context.c", "mgc_resource.c", "mgc_screen.c", "mgc_state.c",
                         "mgc_compiler.c", "mgc_vector_compiler.c", "mgc_host_executor.c"):
            output = temporary / (filename + ".o")
            checked([arguments[0], *flags, "-o", str(output), "-c", str(DRIVER / filename)], cwd=BUILD)
            driver_objects.append(str(output))
        position = link.index("src/gallium/drivers/mgc/libmgc.a")
        link[position:position + 1] = driver_objects
        link += ["-fsanitize=address,undefined"]
    checked(link, cwd=BUILD)
    environment["PKG_CONFIG_PATH"] = str(BUILD / "meson-uninstalled")
    pkg_flags = checked(["pkg-config", "--cflags", "--libs", "egl", "glesv2"], env=environment)
    harness = temporary / "mgc_harness"
    checked([arguments[0], "-std=c11", "-Wall", "-Wextra", "-Werror",
             str(ROOT / "tools/mgc_harness.c"), "-o", str(harness),
             *shlex.split(pkg_flags.stdout)], env=environment)
    return helper, harness


def transport(fresh_binaries, directory, case="basic", enabled=True):
    environment = dict(os.environ)
    environment.pop("MGC_HOST_CAPTURE_DIR", None)
    environment.pop("MGC_HOST_MESH", None)
    environment.pop("MGC_COMPILE_TARGET", None)
    environment.pop("MGC_ASM_DUMP", None)
    if enabled:
        environment["MGC_HOST_CAPTURE_DIR"] = str(directory)
    result = checked([str(fresh_binaries[0]), case], env=environment)
    return json.loads(result.stdout)


def captures(directory):
    paths = sorted(directory.glob("*.json"))
    for path in paths:
        assert stat.S_ISREG(path.lstat().st_mode)
        assert path.stat().st_size <= 12 * 1024 * 1024
    assert not list(directory.glob(".mgc-*.tmp"))
    return [json.loads(path.read_text()) for path in paths]


def test_transport_snapshot_offsets_constants(fresh_binaries, tmp_path):
    status = transport(fresh_binaries, tmp_path, "immutable")
    assert status["accepted"] == 1 and not status["rejected"] and not status["error"]
    artifact, = captures(tmp_path)
    assert artifact["schema"] == "mgc-host-draw-v1"
    assert artifact["program_format"] == "legacy-mgcsim-v1"
    assert artifact["shader_abi_version"] == 1
    assert artifact["draw"] == {
        "mode": "triangles", "start": 0, "count": 3, "index_size": 0, "index_bias": 0,
        "instance_count": 1, "start_instance": 0, "primitive_restart": False, "restart_index": 0,
    }
    assert artifact["vertex_buffers"][0]["stride"] == 12
    assert bytes.fromhex(artifact["vertex_buffers"][0]["data_hex"]) == struct.pack("=9f", *range(1, 10))
    assert artifact["vertex_elements"] == [
        {"src_offset": 0, "vertex_buffer_index": 0, "instance_divisor": 0, "format": "R32G32B32_FLOAT"},
    ]
    assert bytes.fromhex(artifact["constants"]["vs"][0]) == struct.pack("=4f", 1, .5, .25, 1)
    assert artifact["constants"]["vs"][1:] == [None] * 7
    assert artifact["constants"]["fs"] == [None] * 8
    assert artifact["indices"] is None
    assert all(isinstance(shader["layout"], dict) for shader in artifact["shaders"].values())


@pytest.mark.parametrize("index_size", [1, 2, 4])
def test_transport_index_sizes_start_and_negative_bias(fresh_binaries, tmp_path, index_size):
    status = transport(fresh_binaries, tmp_path, f"indexed{index_size}")
    assert status["accepted"] == 1 and not status["error"]
    artifact, = captures(tmp_path)
    assert artifact["draw"]["start"] == 1
    assert artifact["draw"]["index_bias"] == -1
    assert artifact["draw"]["index_size"] == index_size
    assert bytes.fromhex(artifact["indices"]) == struct.pack("=" + {1: "B", 2: "H", 4: "I"}[index_size] * 4,
                                                            99, 1, 2, 3)


def test_transport_batch_sequences(fresh_binaries, tmp_path):
    status = transport(fresh_binaries, tmp_path, "batch")
    assert status["accepted"] == 2 and not status["error"]
    artifacts = captures(tmp_path)
    assert [artifact["drawid"] for artifact in artifacts] == [7, 8]
    assert [artifact["sequence"] for artifact in artifacts] == [1, 2]


def test_transport_explicit_hardware_capture(fresh_binaries, tmp_path):
    status = transport(fresh_binaries, tmp_path, "hardware")
    assert status["accepted"] == 1 and not status["error"]
    artifact, = captures(tmp_path)
    assert artifact["program_format"] == "fp16-hardware-v1"
    assert artifact["shader_abi_version"] == 2
    assert all(shader["layout"]["address_units"] == "bytes" for shader in artifact["shaders"].values())


@pytest.mark.parametrize("case", [
    "buffer_limit", "data_limit", "layout_limit", "assembly_limit", "format", "stride", "sparse", "divisor",
    "missing_layout", "missing_assembly", "missing_format", "abi_mismatch", "tag_mismatch",
    "target_mismatch", "assembly_tag", "base_mismatch", "layout_array", "invalid_json", "extra_json",
    "duplicate_version", "unsupported_layout", "bad_utf8", "bad_surrogate", "depth_limit", "mapped",
    "range", "instances", "start_instance", "restart", "mode", "write_failure",
])
def test_transport_rejects_without_partial_files(fresh_binaries, tmp_path, case):
    status = transport(fresh_binaries, tmp_path, case)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"]
    assert captures(tmp_path) == []


@pytest.mark.parametrize("case,enabled", [("basic", False), ("disabled", True)])
def test_no_implicit_capture_or_null_override(fresh_binaries, tmp_path, case, enabled):
    status = transport(fresh_binaries, tmp_path, case, enabled)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"]
    assert captures(tmp_path) == []


def test_explicit_callback_overrides_capture(fresh_binaries, tmp_path):
    status = transport(fresh_binaries, tmp_path, "custom")
    assert status["accepted"] == 1 and status["callbacks"] == 1 and not status["error"]
    assert captures(tmp_path) == []


def test_callback_reentry_is_rejected(fresh_binaries, tmp_path):
    status = transport(fresh_binaries, tmp_path, "reentry")
    assert status["accepted"] == 1 and status["callbacks"] == 1
    assert status["rejected"] == 1 and status["error"]
    assert captures(tmp_path) == []


def test_empty_draw_creates_no_capture(fresh_binaries, tmp_path):
    status = transport(fresh_binaries, tmp_path, "empty")
    assert not status["accepted"] and not status["rejected"] and not status["error"]
    assert captures(tmp_path) == []


@pytest.mark.parametrize("kind", ["file", "symlink", "fifo"])
def test_atomic_publish_never_clobbers(fresh_binaries, tmp_path, kind):
    final = tmp_path / "draw-1-7.json"
    sentinel = b"existing content must survive\n"
    if kind == "file":
        final.write_bytes(sentinel)
    elif kind == "symlink":
        target = tmp_path / "target"
        target.write_bytes(sentinel)
        final.symlink_to(target)
    else:
        os.mkfifo(final)
    status = transport(fresh_binaries, tmp_path)
    assert not status["accepted"] and status["rejected"] == 1
    assert not list(tmp_path.glob(".mgc-*.tmp"))
    if kind == "fifo":
        assert stat.S_ISFIFO(final.lstat().st_mode)
    else:
        assert final.read_bytes() == sentinel
        assert final.is_symlink() == (kind == "symlink")


@pytest.mark.parametrize("kind", ["missing", "file", "symlink"])
def test_capture_requires_existing_real_directory(fresh_binaries, tmp_path, kind):
    destination = tmp_path / "destination"
    if kind == "file":
        destination.write_text("not a directory")
    elif kind == "symlink":
        destination.symlink_to(tmp_path, target_is_directory=True)
    status = transport(fresh_binaries, destination)
    assert not status["accepted"] and status["rejected"] == 1 and status["error"]
    assert not list(tmp_path.glob("draw-*.json"))


def gl_environment(dump, capture=None, driver="mgc", target="legacy-mgcsim-v1"):
    environment = dict(os.environ)
    environment.pop("MGC_HOST_MESH", None)
    environment.update({
        "GALLIUM_DRIVER": driver, "EGL_PLATFORM": "surfaceless", "LIBGL_ALWAYS_SOFTWARE": "1",
        "MESA_SHADER_CACHE_DISABLE": "1", "MESA_LOADER_DRIVER_OVERRIDE": "swrast",
        "LIBGL_DRIVERS_PATH": str(BUILD / "src/gallium/targets/dri"),
        "LD_LIBRARY_PATH": os.pathsep.join(str(BUILD / suffix) for suffix in
                                          ("src/egl", "src/mapi/es2api", "src/gallium/targets/dri"))
        + os.pathsep + environment.get("LD_LIBRARY_PATH", ""),
        "MGC_ASM_DUMP": str(dump), "MGC_COMPILE_TARGET": target,
    })
    environment.pop("MGC_HOST_CAPTURE_DIR", None)
    if capture is not None:
        environment["MGC_HOST_CAPTURE_DIR"] = str(capture)
    return environment


@pytest.mark.parametrize("shader", ["01_flat_color", "02_varying"])
def test_fresh_default_harness_compiles_without_drawing(fresh_binaries, tmp_path, shader):
    dump = tmp_path / "shader-dump"
    capture = tmp_path / "draw-capture"
    dump.mkdir()
    capture.mkdir()
    source = ROOT / "shaders" / shader
    result = checked([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl")],
                     env=gl_environment(dump, capture))
    assert "GL_RENDERER: mgc" in result.stderr
    assert not captures(capture)
    for stage in ("vs", "fs"):
        assert (dump / f"{stage}.asm").is_file()
        assert json.loads((dump / f"{stage}.layout.json").read_text())["supported"]


@pytest.mark.parametrize("shader,target,abi", [
    (shader, "legacy-mgcsim-v1", 1)
    for shader in ("01_flat_color", "02_varying", "03_uniform", "12_runtime_arithmetic")
] + [(shader, "fp16-hardware-v1", 2) for shader in ("01_flat_color", "02_varying")])
def test_fresh_gl_draw_captures_real_layout_and_fp32_data(fresh_binaries, tmp_path, shader, target, abi):
    dump = tmp_path / "shader-dump"
    capture = tmp_path / "draw-capture"
    dump.mkdir()
    capture.mkdir()
    source = ROOT / "shaders" / shader
    result = checked([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"), "--draw"],
                     env=gl_environment(dump, capture, target=target))
    assert "GL_RENDERER: mgc" in result.stderr
    artifact, = captures(capture)
    assert artifact["program_format"] == target and artifact["shader_abi_version"] == abi
    assert artifact["draw"]["mode"] == "triangles" and artifact["draw"]["count"] == 3
    assert artifact["draw"]["index_size"] == 0 and artifact["draw"]["index_bias"] == 0
    assert artifact["draw"]["restart_index"] == 0
    for stage in ("vs", "fs"):
        assert artifact["shaders"][stage]["assembly"] == (dump / f"{stage}.asm").read_text()
        assert artifact["shaders"][stage]["layout"] == json.loads((dump / f"{stage}.layout.json").read_text())
        assert artifact["shaders"][stage]["layout"]["supported"]
    positions = [(-.75, -.75, 0, 1), (.75, -.75, 0, 1), (0, .75, 0, 1)]
    for element in artifact["vertex_elements"]:
        buffer = artifact["vertex_buffers"][element["vertex_buffer_index"]]
        assert buffer["stride"] == 40
        data = bytes.fromhex(buffer["data_hex"])
        components = element["format"].count("32")
        for index in range(3):
            actual = struct.unpack_from("=" + "f" * components, data,
                                        element["src_offset"] + index * buffer["stride"])
            assert actual == (positions[index] if components == 4 else (1, .5, .25))
    if shader == "12_runtime_arithmetic":
        stages = [stage for stage in ("vs", "fs") if artifact["shaders"][stage]["layout"]["uniforms"]]
        assert len(stages) == 1
        data = bytes.fromhex(artifact["constants"][stages[0]][0])
        assert struct.unpack_from("=f", data, 0)[0] == .25
        assert struct.unpack_from("=f", data, 16)[0] == 2
    if shader == "03_uniform":
        data = bytes.fromhex(artifact["constants"]["vs"][0])
        assert struct.unpack_from("=16f", data, 0) == (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        assert struct.unpack_from("=4f", data, 64) == (.5, .25, 1, .5)


@pytest.mark.parametrize("shader,expected", [
    ("01_flat_color", (255, 0, 0, 255)), ("02_varying", (255, 128, 64, 255)),
    ("03_uniform", (128, 32, 64, 128)), ("12_runtime_arithmetic", (128, 0, 64, 255)),
])
def test_softpipe_reference_uses_same_triangle_and_uniforms(fresh_binaries, tmp_path, shader, expected):
    source = ROOT / "shaders" / shader
    output = tmp_path / "reference.rgba8"
    result = checked([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"),
                      "--reference", str(output)], env=gl_environment(tmp_path, driver="softpipe"))
    assert "GL_RENDERER: softpipe" in result.stderr
    pixels = output.read_bytes()
    assert len(pixels) == 16 * 16 * 4
    assert pixels[:4] == bytes(4)
    center = (7 * 16 + 8) * 4
    assert all(abs(actual - value) <= 1 for actual, value in zip(pixels[center:center + 4], expected))
    assert pixels[(3 * 16 + 8) * 4 + 3] > 0
    assert pixels[(13 * 16 + 8) * 4 + 3] == 0


@pytest.mark.parametrize("shader", ["03_uniform", "12_runtime_arithmetic"])
def test_hardware_unsupported_arithmetic_does_not_emit_capture(fresh_binaries, tmp_path, shader):
    source = ROOT / "shaders" / shader
    dump = tmp_path / "shader-dump"
    capture = tmp_path / "draw-capture"
    dump.mkdir()
    capture.mkdir()
    result = subprocess.run([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"),
                             "--draw"], env=gl_environment(dump, capture, target="fp16-hardware-v1"),
                            text=True, capture_output=True, timeout=60)
    assert result.returncode != 0 and "unsupported NIR instruction" in result.stderr
    assert not captures(capture)


def mesh_bytes(positions):
    return b"".join(struct.pack("<7f", *position, 0, 0, 1) for position in positions)


def test_mesh_positions_and_normals_are_actually_bound(fresh_binaries, tmp_path):
    positions = [(-.5, -.5, 0, 1), (.5, -.5, 0, 1), (.5, .5, 0, 1),
                 (-.5, -.5, 0, 1), (.5, .5, 0, 1), (-.5, .5, 0, 1)]
    mesh = tmp_path / "mesh.bin"
    mesh.write_bytes(mesh_bytes(positions))
    source = ROOT / "shaders/07_lighting_phong"
    capture = tmp_path / "capture"
    capture.mkdir()
    environment = gl_environment(tmp_path, capture)
    environment["MGC_HOST_MESH"] = str(mesh)
    checked([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"), "--draw"],
            env=environment)
    artifact, = captures(capture)
    assert artifact["draw"]["count"] == 6
    assert len(artifact["vertex_elements"]) == 2
    for element in artifact["vertex_elements"]:
        buffer = artifact["vertex_buffers"][element["vertex_buffer_index"]]
        data = bytes.fromhex(buffer["data_hex"])
        assert buffer["stride"] == 40
        components = element["format"].count("32")
        for index in range(6):
            actual = struct.unpack_from("<" + "f" * components, data,
                                        index * buffer["stride"] + element["src_offset"])
            assert actual == (positions[index] if components == 4 else (0, 0, 1))
    environment = gl_environment(tmp_path, driver="softpipe")
    environment["MGC_HOST_MESH"] = str(mesh)
    output = tmp_path / "mesh.rgba8"
    checked([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"),
             "--reference", str(output)], env=environment)
    pixels = output.read_bytes()
    assert len(pixels) == 1024
    assert sum(pixels[index] != 0 for index in range(3, len(pixels), 4)) == 64


@pytest.mark.parametrize("kind", [
    "empty", "short", "count", "large", "nan", "infinity", "zero_normal", "symlink", "fifo",
])
def test_mesh_rejects_invalid_or_nonordinary_input(fresh_binaries, tmp_path, kind):
    mesh = tmp_path / "mesh.bin"
    positions = [(-.75, -.75, 0, 1), (.75, -.75, 0, 1), (0, .75, 0, 1)]
    data = mesh_bytes(positions)
    if kind == "empty":
        data = b""
    elif kind == "short":
        data = data[:-1]
    elif kind == "count":
        data = data[:28]
    elif kind == "large":
        data *= 1366
    elif kind == "nan":
        data = struct.pack("<f", float("nan")) + data[4:]
    elif kind == "infinity":
        data = data[:16] + struct.pack("<f", float("inf")) + data[20:]
    elif kind == "zero_normal":
        data = data[:16] + bytes(12) + data[28:]
    if kind == "symlink":
        source_mesh = tmp_path / "source.bin"
        source_mesh.write_bytes(data)
        mesh.symlink_to(source_mesh)
    elif kind == "fifo":
        os.mkfifo(mesh)
    else:
        mesh.write_bytes(data)
    source = ROOT / "shaders/01_flat_color"
    capture = tmp_path / "capture"
    capture.mkdir()
    environment = gl_environment(tmp_path, capture)
    environment["MGC_HOST_MESH"] = str(mesh)
    result = subprocess.run([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"),
                             "--draw"], env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0 and "mesh" in result.stderr
    assert not captures(capture)


def test_mesh_last_valid_triangle_list_size(fresh_binaries, tmp_path):
    mesh = tmp_path / "mesh.bin"
    mesh.write_bytes(mesh_bytes([(-.75, -.75, 0, 1), (.75, -.75, 0, 1), (0, .75, 0, 1)]) * 1365)
    capture = tmp_path / "capture"
    capture.mkdir()
    source = ROOT / "shaders/01_flat_color"
    environment = gl_environment(tmp_path, capture)
    environment["MGC_HOST_MESH"] = str(mesh)
    checked([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"), "--draw"],
            env=environment)
    artifact, = captures(capture)
    assert artifact["draw"]["count"] == 4095


def test_reference_rejects_mgc_without_claiming_readback(fresh_binaries, tmp_path):
    source = ROOT / "shaders/01_flat_color"
    output = tmp_path / "reference.rgba8"
    result = subprocess.run([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"),
                             "--reference", str(output)], env=gl_environment(tmp_path),
                            text=True, capture_output=True, timeout=60)
    assert result.returncode != 0 and "requires the softpipe CPU renderer" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("shader", ["01_flat_color", "02_varying", "03_uniform", "12_runtime_arithmetic"])
def test_fresh_captured_draw_matches_real_softpipe_pixels(fresh_binaries, tmp_path, shader):
    from tools.host_render import render_artifact

    source = ROOT / "shaders" / shader
    capture = tmp_path / "capture"
    capture.mkdir()
    checked([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"), "--draw"],
            env=gl_environment(tmp_path, capture))
    artifact, = captures(capture)
    image, coverage, report = render_artifact(artifact, width=16, height=16)
    output = tmp_path / "reference.rgba8"
    checked([str(fresh_binaries[1]), str(source / "vert.glsl"), str(source / "frag.glsl"),
             "--reference", str(output)], env=gl_environment(tmp_path, driver="softpipe"))
    bottom_left = output.read_bytes()
    reference = b"".join(bottom_left[row * 64:(row + 1) * 64] for row in range(15, -1, -1))
    actual = image.tobytes()
    assert len(actual) == len(reference) == 1024
    assert max(abs(first - second) for first, second in zip(actual, reference)) <= 8
    assert all((actual[index] != 0) == (reference[index] != 0) for index in range(3, 1024, 4))
    assert coverage.max() == 1 and report["fpga_execution"] is False
