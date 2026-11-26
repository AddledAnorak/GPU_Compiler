"""Direct C sampler, view, binding, and capture guards; no rendered-output claims."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
DRIVER = ROOT / "mesa/src/gallium/drivers/mgc"
SOURCE = ROOT / "tests/fixtures/texture-capture/guard.c"
SAMPLER_REJECTIONS = (
    "min_linear", "mag_linear", "mip_nearest", "mip_linear", "wrap_s", "wrap_t",
    "unnormalized", "compare", "anisotropy", "lod_bias", "lod_bias_nan", "min_lod",
    "min_lod_nan", "max_lod_negative", "max_lod_nan", "max_lod_infinite",
    "reduction", "seamless_cube", "integer_border",
)
RESOURCE_REJECTIONS = (
    "view_format", "view_level", "view_last_level", "view_layer", "view_last_layer",
    "view_target", "view_from_buffer", "view_swizzle", "resource_format", "mip_chain",
    "array", "rectangle", "width_large", "height_large", "width_zero", "height_zero",
    "samples", "storage_samples", "depth", "mapped", "feedback_color", "feedback_depth",
    "storage_stride", "storage_layer_stride", "storage_offset", "storage_size",
    "storage_large_stride", "storage_width", "storage_height", "storage_layers", "storage_null",
)
BINDING_REJECTIONS = (
    "missing_view", "view_null_entry", "missing_state", "state_null_entry",
    "extra_fs_view", "extra_vs_view", "extra_fs_state", "extra_vs_state",
)
TRANSACTIONAL_BINDINGS = (
    "bind_state_stage", "bind_state_start", "bind_state_count", "create_state_null",
    "bind_view_stage", "bind_view_start", "bind_view_count", "bind_view_trailing",
    "bind_view_transaction",
)
VALID_TEXTURES = (
    "valid", "valid_small", "valid_npot", "valid_copy", "valid_snapshot",
    "valid_empty_binds", "valid_trailing_unbind", "valid_anisotropy_one",
    "valid_max_lod_zero", "valid_padded", "valid_status", "valid_discard",
)
DISCARD_CONTRACT_REJECTIONS = (
    "discard_unversioned", "discard_status_option_zero", "discard_status_option_two",
    "discard_status_missing", "discard_status_version", "discard_status_offset",
    "discard_status_pending", "discard_status_discarded", "discard_status_committed",
    "discard_status_duplicate", "discard_status_unknown", "discard_status_missing_field",
    "discard_status_noninteger",
)
FRAGMENT_STATUS = {
    "version": 1, "word_offset": 65535, "pending": 0, "discarded": 1, "committed": 2,
}
HOST_TEXTURE = {
    "version": 1, "texture": 0, "sampler": 0, "storage_block": 15,
    "max_width": 16, "max_height": 16, "filter": "nearest", "wrap": "clamp-to-edge",
}
SAMPLER = {
    "normalized_coords": True, "min_filter": "nearest", "mag_filter": "nearest",
    "mip_filter": "none", "wrap_s": "clamp-to-edge", "wrap_t": "clamp-to-edge",
}


def checked(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, timeout=120, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def build_texture_guard(tmp_path_factory, source=SOURCE):
    if not (BUILD / "compile_commands.json").is_file():
        pytest.skip("configure and build Mesa with the mgc Gallium driver first")
    environment = dict(os.environ)
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    ninja = shutil.which("ninja", path=environment["PATH"])
    if not ninja:
        pytest.skip("ninja is required to inspect Mesa's existing link command")
    commands = json.loads((BUILD / "compile_commands.json").read_text())
    entry = next((item for item in commands if item["file"].endswith("/mgc_context.c")), None)
    if entry is None:
        pytest.skip("Mesa was not configured with mgc")
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
    sanitized = environment.get("MGC_RUNTIME_SANITIZE") == "1"
    if sanitized:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    temporary = tmp_path_factory.mktemp("texture-capture-guards")
    object_file = temporary / "guard.o"
    checked([arguments[0], *flags, "-I" + str(DRIVER), "-o", str(object_file),
             "-c", str(source)], cwd=BUILD)
    link_lines = checked([ninja, "-C", str(BUILD), "-t", "commands",
                          "src/gallium/drivers/mgc/mgc_test_arith"], env=environment)
    link = shlex.split(link_lines.strip().splitlines()[-1])
    helper = temporary / "guard"
    link[link.index("-o") + 1] = str(helper)
    original = "src/gallium/drivers/mgc/mgc_test_arith.p/mgc_test_arith.c.o"
    link[link.index(original)] = str(object_file)
    driver_objects = []
    for filename in ("mgc_context.c", "mgc_resource.c", "mgc_screen.c", "mgc_state.c", "mgc_compiler.c", "mgc_vector_compiler.c", "mgc_host_executor.c"):
        output = temporary / (filename + ".o")
        checked([arguments[0], *flags, "-o", str(output), "-c", str(DRIVER / filename)], cwd=BUILD)
        driver_objects.append(str(output))
    position = link.index("src/gallium/drivers/mgc/libmgc.a")
    link[position:position + 1] = driver_objects
    if sanitized:
        link += ["-fsanitize=address,undefined"]
    checked(link, cwd=BUILD)
    return helper


@pytest.fixture(scope="module")
def texture_guard(tmp_path_factory):
    return build_texture_guard(tmp_path_factory)


def run_guard(helper, directory, case):
    environment = dict(os.environ)
    for key in ("MGC_HOST_TEXTURE", "MGC_FRAGMENT_STATUS", "MGC_COMPILE_TARGET",
                "MGC_HOST_MESH", "MGC_ASM_DUMP", "MGC_HOST_EXECUTOR_SOCKET"):
        environment.pop(key, None)
    environment["MGC_HOST_CAPTURE_DIR"] = str(directory)
    result = subprocess.run([str(helper), case], env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def read_capture(directory):
    files = list(directory.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json" and files[0].is_file()
    assert files[0].stat().st_size < 12 * 1024 * 1024
    return json.loads(files[0].read_text())


def assert_texture_capture(artifact, case, *, host_texture=HOST_TEXTURE, sampler=SAMPLER):
    assert artifact["schema"] == "mgc-host-draw-v2"
    assert artifact["program_format"] == "legacy-mgcsim-v1"
    assert artifact["shader_abi_version"] == 1
    assert artifact["draw"]["count"] == 3 and artifact["drawid"] == 7
    assert artifact["shaders"]["vs"]["layout"]["version"] == 1
    assert "host_texture" not in artifact["shaders"]["vs"]["layout"]
    assert artifact["shaders"]["fs"]["layout"]["version"] == 2
    assert artifact["shaders"]["fs"]["layout"]["host_texture"] == host_texture
    fragment_layout = artifact["shaders"]["fs"]["layout"]
    assert fragment_layout["uses_discard"] == (case == "valid_discard")
    if case in ("valid_status", "valid_discard"):
        assert fragment_layout["fragment_status"] == FRAGMENT_STATUS
    else:
        assert "fragment_status" not in fragment_layout
    assert "fragment_status" not in artifact["shaders"]["vs"]["layout"]
    texture, = artifact["textures"]
    width, height = {"valid_small": (1, 1), "valid_npot": (3, 5),
                     "valid_padded": (3, 2)}.get(case, (16, 16))
    assert texture["width"] == width and texture["height"] == height
    assert texture["format"] == "R8G8B8A8_UNORM"
    assert texture["sampler"] == sampler
    texels = bytes.fromhex(texture["data_hex"])
    assert len(texels) == width * height * 4 <= 1024
    if case == "valid_padded":
        indices = [offset for row in range(height) for offset in range(4 + row * 64, 4 + row * 64 + width * 4)]
    else:
        indices = range(width * height * 4)
    assert texels == bytes((index * 17 + 3) & 255 for index in indices)


@pytest.mark.parametrize("case", VALID_TEXTURES + TRANSACTIONAL_BINDINGS)
def test_direct_c_valid_capture_and_transactional_bindings(texture_guard, tmp_path, case):
    status = run_guard(texture_guard, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0
    assert status["error"] == 0 and status["callbacks"] == 0
    assert bool(status["operation_error"]) == (case in TRANSACTIONAL_BINDINGS)
    assert_texture_capture(read_capture(tmp_path), case)


@pytest.mark.parametrize("case", SAMPLER_REJECTIONS + RESOURCE_REJECTIONS + BINDING_REJECTIONS + DISCARD_CONTRACT_REJECTIONS + (
    "custom_texture", "custom_texture_option_zero", "callback_disabled",
))
def test_direct_c_rejection_before_capture_or_custom_callback(texture_guard, tmp_path, case):
    status = run_guard(texture_guard, tmp_path, case)
    assert status["accepted"] == 0 and status["rejected"] == 1
    assert status["error"] != 0 and status["operation_error"] == 0
    assert status["callbacks"] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ("legacy_default", "legacy_opt_in", "legacy_custom", "legacy_custom_opt_in"))
def test_untextured_v1_and_public_callback_abi_unchanged(texture_guard, tmp_path, case):
    status = run_guard(texture_guard, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    custom = "custom" in case
    assert status["callbacks"] == int(custom)
    if custom:
        assert list(tmp_path.iterdir()) == []
    else:
        artifact = read_capture(tmp_path)
        assert artifact["schema"] == "mgc-host-draw-v1"
        assert artifact["shader_abi_version"] == 1
        assert "textures" not in artifact
        for shader in artifact["shaders"].values():
            assert shader["layout"]["version"] == 1
            assert "host_texture" not in shader["layout"]
