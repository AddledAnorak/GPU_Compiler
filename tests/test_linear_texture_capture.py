"""Bounded C linear capture guards, not output sampling or a hardware contract."""

import pytest

from test_texture_capture import (
    BINDING_REJECTIONS,
    DISCARD_CONTRACT_REJECTIONS,
    HOST_TEXTURE,
    RESOURCE_REJECTIONS,
    ROOT,
    SAMPLER,
    SAMPLER_REJECTIONS,
    SOURCE,
    TRANSACTIONAL_BINDINGS,
    VALID_TEXTURES,
    assert_texture_capture,
    build_texture_guard,
    read_capture,
    run_guard,
    texture_guard,
)


LINEAR_HOST_TEXTURE = dict(HOST_TEXTURE, version=2, filter="linear")
LINEAR_SAMPLER = dict(SAMPLER, min_filter="linear", mag_filter="linear")
LINEAR_SAMPLER_REJECTIONS = tuple(
    case for case in SAMPLER_REJECTIONS if case not in ("min_linear", "mag_linear")
) + ("min_nearest", "mag_nearest", "both_nearest", "min_unknown", "mag_unknown")
LAYOUT_REJECTIONS = (
    "abi_zero", "abi_nearest", "abi_unknown", "abi_vs_unknown", "uses_texture_false",
    "nearest_abi_linear", "nearest_linear_sampler", "nearest_hardware", "linear_hardware",
    "legacy_unknown_opt_in", "layout_version_one", "layout_version_three",
    "layout_host_version_one", "layout_host_version_three", "layout_host_version_float",
    "layout_filter_nearest", "layout_filter_case", "layout_filter_unknown",
    "layout_wrap_repeat", "layout_texture_slot", "layout_sampler_slot", "layout_block",
    "layout_max_width", "layout_max_height", "layout_duplicate", "layout_unknown",
    "layout_missing_field", "layout_missing_texture", "layout_stage", "layout_shader_abi",
)


@pytest.fixture(scope="module")
def linear_texture_guard(tmp_path_factory):
    hooks = ROOT / "tests/fixtures/linear-texture-capture/hooks.c"
    replacements = (
        ('if (opt_in) assert(!setenv("MGC_HOST_TEXTURE", "1", 1));',
         'if (opt_in) assert(!setenv("MGC_HOST_TEXTURE", '
         '!strncmp(name, "nearest_", 8) ? "1" : "2", 1));'),
        (".min_img_filter = PIPE_TEX_FILTER_NEAREST,",
         ".min_img_filter = PIPE_TEX_FILTER_LINEAR,"),
        (".mag_img_filter = PIPE_TEX_FILTER_NEAREST,",
         ".mag_img_filter = PIPE_TEX_FILTER_LINEAR,"),
        ('known |= change_discard_contract(name, ctx->shaders[1]->shader);',
         'known |= change_discard_contract(name, ctx->shaders[1]->shader);\n'
         '   known |= change_linear_contract(name, ctx);'),
        ('known |= change_sampler(name, &sampler);',
         'known |= change_sampler(name, &sampler);\n'
         '      known |= change_linear_sampler(name, &sampler);'),
        ('((struct pipe_sampler_state *)state)->min_img_filter = PIPE_TEX_FILTER_LINEAR;',
         '((struct pipe_sampler_state *)state)->min_img_filter = PIPE_TEX_FILTER_NEAREST;'),
        ('int\nmain(int argc, char **argv)',
         f'#include "{hooks}"\n\nstatic int\ncapture_guard_main(int argc, char **argv)'),
    )
    contents = SOURCE.read_text()
    for original, replacement in replacements:
        assert contents.count(original) == 1, original
        contents = contents.replace(original, replacement, 1)
    source = tmp_path_factory.mktemp("linear-texture-capture-source") / "guard.c"
    source.write_text(contents)
    return build_texture_guard(tmp_path_factory, source)


@pytest.mark.parametrize("case", VALID_TEXTURES + TRANSACTIONAL_BINDINGS)
def test_direct_c_linear_capture_and_transactional_bindings(linear_texture_guard, tmp_path, case):
    status = run_guard(linear_texture_guard, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["callbacks"] == 0
    assert (status["operation_error"] != 0) == (case in TRANSACTIONAL_BINDINGS)
    assert_texture_capture(read_capture(tmp_path), case,
                           host_texture=LINEAR_HOST_TEXTURE, sampler=LINEAR_SAMPLER)


@pytest.mark.parametrize("case", LINEAR_SAMPLER_REJECTIONS + RESOURCE_REJECTIONS +
                         BINDING_REJECTIONS + DISCARD_CONTRACT_REJECTIONS +
                         LAYOUT_REJECTIONS + (
                             "custom_texture", "custom_texture_option_zero", "callback_disabled",
                         ))
def test_direct_c_linear_rejects_before_capture_or_callback(linear_texture_guard, tmp_path, case):
    status = run_guard(linear_texture_guard, tmp_path, case)
    assert status["accepted"] == 0 and status["rejected"] == 1
    assert status["error"] != 0 and status["operation_error"] == 0
    assert status["callbacks"] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ("legacy_default", "legacy_opt_in", "legacy_custom",
                                 "legacy_custom_opt_in"))
def test_untextured_linear_option_preserves_public_callback_abi(linear_texture_guard, tmp_path, case):
    status = run_guard(linear_texture_guard, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    custom = "custom" in case
    assert status["callbacks"] == int(custom)
    if custom:
        assert list(tmp_path.iterdir()) == []
    else:
        artifact = read_capture(tmp_path)
        assert artifact["schema"] == "mgc-host-draw-v1"
        assert artifact["shader_abi_version"] == 1 and "textures" not in artifact
        for shader in artifact["shaders"].values():
            assert shader["layout"]["version"] == 1 and "host_texture" not in shader["layout"]


def test_linear_payload_shape_matches_nearest(texture_guard, linear_texture_guard, tmp_path):
    artifacts = []
    for helper, name in ((texture_guard, "nearest"), (linear_texture_guard, "linear")):
        directory = tmp_path / name
        directory.mkdir()
        assert run_guard(helper, directory, "valid")["accepted"] == 1
        artifacts.append(read_capture(directory))
    nearest, linear = artifacts
    assert nearest.keys() == linear.keys()
    for stage in ("vs", "fs"):
        assert nearest["shaders"][stage].keys() == linear["shaders"][stage].keys()
        assert nearest["shaders"][stage]["layout"].keys() == linear["shaders"][stage]["layout"].keys()
    assert nearest["shaders"]["fs"]["layout"]["host_texture"].keys() == LINEAR_HOST_TEXTURE.keys()
    assert nearest["textures"][0].keys() == linear["textures"][0].keys()
    assert nearest["textures"][0]["sampler"].keys() == linear["textures"][0]["sampler"].keys()
    assert dict(linear["textures"][0], sampler=SAMPLER) == nearest["textures"][0]


@pytest.mark.parametrize("case", ("compile_default", "compile_zero", "compile_unknown",
                                 "compile_hardware_nearest", "compile_hardware_linear"))
def test_default_unknown_and_hardware_options_still_reject_texture(linear_texture_guard, tmp_path, case):
    status = run_guard(linear_texture_guard, tmp_path, case)
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert status["accepted"] == 0 and status["rejected"] == 0 and status["callbacks"] == 0
    assert list(tmp_path.iterdir()) == []
