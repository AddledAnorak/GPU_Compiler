import copy
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tools.host_texture import HOST_TEXTURE_LAYOUT, SAMPLER_STATE, TEXTURE_BASE, pack_texture
from tools.texture_acceptance import quad_mesh, texel_oracle, texture_pattern, validate
from tools.host_render import ShaderExecutor, pack_constants, render_artifact
from tools.mgcdriver import Driver, ProgramFormat
from tools import run_render
from tools.run_render import capture_draw, softpipe_reference
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout


FIXTURE = ROOT / "tests/fixtures/host-texture"
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))


def texture_layout():
    return {"version": 2, "supported": True, "errors": [],
            "program_format": ProgramFormat.LEGACY.value, "shader_abi_version": 1,
            "address_units": "words", "stage": "fragment",
            "bases": {"input": 0x40000, "uniform": 0x80000, "vertex": 0xC0000, "fragment": 0x400000},
            "inputs": [], "outputs": [], "uniforms": [], "host_texture": dict(HOST_TEXTURE_LAYOUT)}


def captured_texture(width=2, height=2, contents=None):
    contents = bytes(range(width * height * 4)) if contents is None else contents
    return {"width": width, "height": height, "format": "R8G8B8A8_UNORM",
            "data_hex": contents.hex(), "sampler": dict(SAMPLER_STATE)}


@pytest.mark.parametrize("width,height", [(1, 1), (1, 16), (16, 1), (2, 2), (4, 4), (16, 16)])
def test_payload_dimensions_header_row_order_and_fp16_low_words(width, height):
    contents = bytes(index % 256 for index in range(width * height * 4))
    source = texture_layout()
    textures = [captured_texture(width, height, contents)]
    original = copy.deepcopy((source, textures))
    payload = pack_texture(SimpleNamespace(source=source), textures)
    assert (source, textures) == original
    assert sorted(payload) == list(range(TEXTURE_BASE, TEXTURE_BASE + 4 + len(contents)))
    assert payload[TEXTURE_BASE] == struct.unpack("<H", struct.pack("<e", width))[0]
    assert payload[TEXTURE_BASE + 1] == struct.unpack("<H", struct.pack("<e", height))[0]
    assert payload[TEXTURE_BASE + 2] == payload[TEXTURE_BASE + 3] == 0
    for index, channel in enumerate(contents):
        assert payload[TEXTURE_BASE + 4 + index] == struct.unpack("<H", struct.pack("<e", channel / 255))[0]
    assert all(type(value) is int and 0 <= value <= 0xFFFF for value in payload.values())


@pytest.mark.parametrize("textures", [None, {}, (), [None], [], [captured_texture(), captured_texture()]])
def test_payload_requires_exactly_one_explicit_texture(textures):
    with pytest.raises(ValueError):
        pack_texture(texture_layout(), textures)


@pytest.mark.parametrize("field,value", [("width", 0), ("width", 17), ("width", True),
                                        ("width", 2.0), ("width", "2"), ("width", None),
                                        ("height", -1), ("height", 17), ("height", False),
                                        ("height", 2.0), ("height", 1 << 32),
                                        ("format", "R8G8B8A8_SRGB"), ("format", "B8G8R8A8_UNORM"),
                                        ("format", "R32G32B32A32_FLOAT"), ("format", "RGBA8"),
                                        ("format", None), ("format", 1),
                                        ("data_hex", ""), ("data_hex", "00"),
                                        ("data_hex", "00" * 17), ("data_hex", "zz" * 16),
                                        ("data_hex", "00 " * 16), ("data_hex", bytes(16)),
                                        ("data_hex", None)])
def test_payload_rejects_bad_dimensions_formats_and_exact_bytes(field, value):
    texture = captured_texture()
    texture[field] = value
    with pytest.raises(ValueError):
        pack_texture(texture_layout(), [texture])


@pytest.mark.parametrize("field,value", [("normalized_coords", False), ("normalized_coords", 1),
                                        ("min_filter", "linear"), ("mag_filter", "linear"),
                                        ("mip_filter", "nearest"), ("mip_filter", "linear"),
                                        ("wrap_s", "repeat"), ("wrap_t", "mirrored-repeat"),
                                        ("wrap_t", "clamp-to-border")])
def test_payload_rejects_unsupported_sampler_state(field, value):
    texture = captured_texture()
    texture["sampler"][field] = value
    with pytest.raises(ValueError, match="sampler"):
        pack_texture(texture_layout(), [texture])


@pytest.mark.parametrize("field", ["width", "height", "format", "data_hex", "sampler"])
def test_payload_rejects_missing_texture_fields(field):
    texture = captured_texture()
    del texture[field]
    with pytest.raises(ValueError):
        pack_texture(texture_layout(), [texture])


@pytest.mark.parametrize("extra", ["row_stride", "depth", "levels", "level", "target", "texture", "storage_block"])
def test_payload_rejects_extra_texture_fields(extra):
    texture = captured_texture()
    texture[extra] = 1
    with pytest.raises(ValueError):
        pack_texture(texture_layout(), [texture])


@pytest.mark.parametrize("sampler", [None, [], {}, {**SAMPLER_STATE, "compare_mode": "none"},
                                    {key: value for key, value in SAMPLER_STATE.items() if key != "wrap_s"}])
def test_payload_rejects_inexact_sampler_objects(sampler):
    texture = captured_texture()
    texture["sampler"] = sampler
    with pytest.raises(ValueError):
        pack_texture(texture_layout(), [texture])


@pytest.mark.parametrize("field,value", [("version", 0), ("version", 2), ("version", True),
                                        ("texture", 1), ("texture", False), ("sampler", 1),
                                        ("storage_block", 14), ("storage_block", 15.0),
                                        ("max_width", 32), ("max_height", 8),
                                        ("filter", "linear"), ("wrap", "repeat")])
def test_payload_rejects_metadata_versions_types_and_aliases(field, value):
    source = texture_layout()
    source["host_texture"][field] = value
    with pytest.raises(ValueError):
        pack_texture(source, [captured_texture()])


@pytest.mark.parametrize("contract", [None, [], {}, {**HOST_TEXTURE_LAYOUT, "extra": 0}])
def test_payload_requires_exact_metadata_dictionary(contract):
    source = texture_layout()
    source["host_texture"] = contract
    with pytest.raises(ValueError):
        pack_texture(source, [captured_texture()])


@pytest.mark.parametrize("field,value", [("version", 1), ("version", 3), ("version", True),
                                        ("version", 2.0), ("supported", False),
                                        ("program_format", ProgramFormat.HARDWARE.value),
                                        ("shader_abi_version", 2), ("shader_abi_version", True),
                                        ("address_units", "bytes"), ("stage", "vertex")])
def test_payload_rejects_wrong_layout_format_version_stage_and_abi(field, value):
    source = texture_layout()
    source[field] = value
    with pytest.raises(ValueError):
        pack_texture(source, [captured_texture()])


def test_payload_v1_without_texture_remains_empty_and_default_texture_rejects():
    source = texture_layout()
    source["version"] = 1
    del source["host_texture"]
    assert pack_texture(source, []) == {}
    with pytest.raises(ValueError, match="actual host texture"):
        pack_texture(source, [captured_texture()])
    source["version"] = 2
    with pytest.raises(ValueError):
        pack_texture(source, [])


@pytest.mark.parametrize("offset", [TEXTURE_BASE, TEXTURE_BASE + 4, 65535])
def test_payload_rejects_ordinary_uniform_aliases_to_block15(offset):
    source = texture_layout()
    source["uniforms"] = [{"word_offset": offset, "type": "float", "bit_size": 32}]
    with pytest.raises(ValueError, match="block 15"):
        pack_texture(source, [captured_texture()])


def test_reflection_ordinary_constant_block15_is_guarded_even_without_reflected_access():
    source = ShaderLayout(texture_layout())
    for encoded in ("00000000", ""):
        with pytest.raises(ValueError, match="texture|reserved|block"):
            pack_constants(source, [None] * 15 + [encoded])
    assert pack_constants(source, [None] * 16) == {}


def test_reflection_ordinary_lower_block_constants_and_texture_payload_are_disjoint():
    source = texture_layout()
    source["uniforms"] = [{"word_offset": 14 * 4096, "type": "float", "bit_size": 32}]
    reflected = ShaderLayout(source)
    constants = pack_constants(reflected, [None] * 14 + [struct.pack("<f", 0.5).hex()])
    texels = pack_texture(reflected, [captured_texture()])
    assert constants == {14 * 4096: 0x3800}
    assert set(constants).isdisjoint(texels)


@pytest.mark.parametrize("options", [{"host_texture": 1}, {"host_texture": "1"},
                                    {"host_texture": None}, {"host_texture": True},
                                    {"texture": Path("texture.bin")},
                                    {"host_texture": True, "texture": Path("texture.bin"),
                                     "target": ProgramFormat.HARDWARE}])
def test_helper_capture_fails_closed_before_invocation(monkeypatch, tmp_path, options):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid texture options must not invoke GL")
    monkeypatch.setattr(run_render.subprocess, "run", unexpected)
    with pytest.raises(ValueError):
        capture_draw(FIXTURE, BUILD, tmp_path, **options)


@pytest.mark.parametrize("enabled", [False, True])
def test_helper_capture_clears_inherited_texture_state_and_preserves_modes(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv("MGC_HOST_TEXTURE", "1")
    monkeypatch.setenv("MGC_HOST_TEXTURE_DATA", "/untrusted/inherited.bin")
    texture = tmp_path / "texture.bin"
    mesh = tmp_path / "vertices.bin"
    def fake_run(command, *, env, **kwargs):
        assert command[-1] == "--draw" and len(command) == 4
        assert env["MGC_HOST_MESH"] == str(mesh.resolve())
        assert env["MGC_HOST_DEPTH_TEST"] == "1"
        assert "MGC_FRAGMENT_STATUS" not in env
        if enabled:
            assert env["MGC_HOST_TEXTURE"] == "1"
            assert env["MGC_HOST_TEXTURE_DATA"] == str(texture.absolute())
        else:
            assert "MGC_HOST_TEXTURE" not in env and "MGC_HOST_TEXTURE_DATA" not in env
        (Path(env["MGC_HOST_CAPTURE_DIR"]) / "draw.json").write_text('{"schema":"helper-test"}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    assert capture_draw(FIXTURE, BUILD, tmp_path / "output", mesh=mesh, depth_test=True,
                        fragment_status=False, host_texture=enabled,
                        texture=texture if enabled else None) == {"schema": "helper-test"}


@pytest.mark.parametrize("supplied", [False, True])
def test_helper_softpipe_upload_uses_only_independent_data_flag(monkeypatch, tmp_path, supplied):
    monkeypatch.setenv("MGC_HOST_TEXTURE", "1")
    monkeypatch.setenv("MGC_HOST_TEXTURE_DATA", "/untrusted/inherited.bin")
    texture = tmp_path / "texture.bin"
    def fake_run(command, *, env, **kwargs):
        assert command[-2] == "--reference" and len(command) == 5
        assert env["GALLIUM_DRIVER"] == "softpipe"
        assert "MGC_HOST_TEXTURE" not in env and "MGC_HOST_CAPTURE_DIR" not in env
        assert "MGC_FRAGMENT_STATUS" not in env and "MGC_ASM_DUMP" not in env
        if supplied:
            assert env["MGC_HOST_TEXTURE_DATA"] == str(texture.absolute())
        else:
            assert "MGC_HOST_TEXTURE_DATA" not in env
        Path(command[-1]).write_bytes(bytes(16 * 16 * 4))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(run_render.subprocess, "run", fake_run)
    assert softpipe_reference(FIXTURE, BUILD, tmp_path / "output",
                              texture=texture if supplied else None).shape == (16, 16, 4)


def test_helper_cli_forwards_explicit_texture_to_capture_and_reference(monkeypatch, tmp_path):
    texture = tmp_path / "texture.bin"
    calls = []
    def fake_capture(*args, **kwargs):
        calls.append("capture")
        assert kwargs["host_texture"] is True and kwargs["texture"] == texture
        return {}
    def fake_reference(*args, **kwargs):
        calls.append("reference")
        assert kwargs["texture"] == texture
        return np.zeros((16, 16, 4), dtype=np.uint8)
    monkeypatch.setattr(run_render, "capture_draw", fake_capture)
    monkeypatch.setattr(run_render, "softpipe_reference", fake_reference)
    monkeypatch.setattr(run_render, "render_artifact", lambda *args, **kwargs:
                        (np.zeros((16, 16, 4), dtype=np.uint8), np.ones((16, 16), dtype=np.uint32), {}))
    assert run_render.main([str(FIXTURE), "--host-texture", "--texture", str(texture),
                            "--compare-softpipe", "--output", str(tmp_path / "output")]) == 0
    assert calls == ["capture", "reference"]


@pytest.mark.parametrize("size,phase", [(1, "identity"), (2, "identity"), (4, "identity"),
                                      (16, "identity"), (2, "clamp"), (2, "phase")])
def test_oracle_pixel_centers_avoid_texel_boundaries_and_keep_zero_alpha(size, phase):
    texels = texture_pattern(size)
    expected = texel_oracle(texels, phase)
    assert expected.shape == (16, 16, 4) and expected.dtype == np.uint8
    assert np.any(expected[:, :, 3] == 0)
    if size == 16:
        assert np.array_equal(expected, texels[::-1])
    if size == 2 and phase == "identity":
        assert np.all(expected[:8, :8] == texels[1, 0])
        assert np.all(expected[8:, 8:] == texels[0, 1])
    assert len(quad_mesh()) == 6 * 28


@pytest.fixture(scope="module")
def fresh_texture_runs(tmp_path_factory):
    output = tmp_path_factory.mktemp("fresh-host-texture")
    submissions = []
    original = Driver.submit
    def actual_submit(driver, program, *args, **kwargs):
        submissions.append(program.format)
        return original(driver, program, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Driver, "submit", actual_submit)
        summary = validate(output, BUILD)
    return output, summary, submissions


@pytest.mark.parametrize("case", ["checker-4x4", "pattern-2x2", "zero-alpha-1x1",
                                 "centers-16x16", "clamp-2x2", "phase-2x2"])
def test_fresh_capture_texture_actual_driver_softpipe_and_analytic_oracles(fresh_texture_runs, case):
    output, summary, submissions = fresh_texture_runs
    report = next(result for result in summary["cases"] if result["case"] == case)
    artifact = json.loads((output / case / "draw.json").read_text())
    assert artifact["schema"] == "mgc-host-draw-v2"
    assert artifact["shaders"]["fs"]["layout"]["host_texture"] == HOST_TEXTURE_LAYOUT
    assert artifact["shaders"]["fs"]["layout"]["version"] == 2
    assert artifact["shaders"]["vs"]["layout"]["version"] == 1
    assert report["softpipe_comparison"]["max_channel_error"] == 0
    assert report["fragments"] == report["committed_fragments"] == 256
    assert report["host_steps"] > 256 and report["fpga_execution"] is False
    assert len(submissions) == 6 * (6 + 256)
    assert all(target == ProgramFormat.LEGACY for target in submissions)


@pytest.mark.parametrize("horizontal,vertical", [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75),
                                                (0.75, 0.75), (-2, -1), (2, 3),
                                                (0, 0.75), (1, 0.25)])
def test_fresh_compiled_legacy_sample_at_centers_and_clamp(fresh_texture_runs, horizontal, vertical):
    output, _, _ = fresh_texture_runs
    artifact = json.loads((output / "pattern-2x2/draw.json").read_text())
    shader = artifact["shaders"]["fs"]
    layout = ShaderLayout(shader["layout"])
    inputs = {entry["word_offset"]: struct.unpack("<H", struct.pack("<e", (horizontal, vertical)[entry["component"]]))[0]
              for entry in layout.entries("inputs")}
    executor = ShaderExecutor(shader["assembly"], layout, texture=artifact["textures"])
    try:
        result = executor.invoke_fragment(inputs)
        assert executor.invocations == 1 and executor.steps > 1
    finally:
        executor.close()
    texels = texture_pattern(2)
    column = min(max(int(np.floor(horizontal * 2)), 0), 1)
    row = min(max(int(np.floor(vertical * 2)), 0), 1)
    for entry in layout.entries("outputs"):
        expected = struct.unpack("<H", struct.pack("<e", int(texels[row, column, entry["component"]]) / 255))[0]
        assert result.outputs[entry["word_offset"]] == expected


@pytest.mark.parametrize("offset", [TEXTURE_BASE, TEXTURE_BASE + 4, 65535])
def test_fresh_executor_rejects_caller_constants_in_reserved_texture_block(fresh_texture_runs, offset):
    output, _, _ = fresh_texture_runs
    artifact = json.loads((output / "pattern-2x2/draw.json").read_text())
    shader = artifact["shaders"]["fs"]
    with pytest.raises(ValueError, match="constants|reserved|texture"):
        ShaderExecutor(shader["assembly"], ShaderLayout(shader["layout"]),
                       {offset: 0}, texture=artifact["textures"])


def test_fresh_compiled_load_reads_changed_texels_without_shadow_shader_output(fresh_texture_runs):
    output, _, _ = fresh_texture_runs
    artifact = json.loads((output / "pattern-2x2/draw.json").read_text())
    original_assembly = artifact["shaders"]["fs"]["assembly"]
    texels = texture_pattern(2).copy()
    texels[:, :, :3] = 255 - texels[:, :, :3]
    artifact["textures"][0]["data_hex"] = texels.tobytes().hex()
    image, coverage, report = render_artifact(artifact)
    assert artifact["shaders"]["fs"]["assembly"] == original_assembly
    assert np.array_equal(image, texel_oracle(texels)) and np.all(coverage == 1)
    assert report["fragments"] == 256 and report["host_steps"] > 256


@pytest.mark.parametrize("mutation", ["v1-textures", "missing-textures", "no-texture-metadata",
                                     "hardware", "bad-format", "bad-state", "ubo15"])
def test_fresh_artifact_rejects_schema_format_state_and_ordinary_ubo_collisions(fresh_texture_runs, mutation):
    output, _, _ = fresh_texture_runs
    artifact = json.loads((output / "pattern-2x2/draw.json").read_text())
    if mutation == "v1-textures":
        artifact["schema"] = "mgc-host-draw-v1"
    elif mutation == "missing-textures":
        del artifact["textures"]
    elif mutation == "no-texture-metadata":
        del artifact["shaders"]["fs"]["layout"]["host_texture"]
        artifact["shaders"]["fs"]["layout"]["version"] = 1
    elif mutation == "hardware":
        artifact["program_format"] = ProgramFormat.HARDWARE.value
    elif mutation == "bad-format":
        artifact["textures"][0]["format"] = "R8G8B8A8_SRGB"
    elif mutation == "bad-state":
        artifact["textures"][0]["sampler"]["min_filter"] = "linear"
    elif mutation == "ubo15":
        artifact["constants"]["fs"] = [None] * 15 + ["00000000"]
    with pytest.raises(ValueError):
        render_artifact(artifact)


def test_fresh_default_and_hardware_texture_compilation_reject(tmp_path):
    for target in (ProgramFormat.LEGACY, ProgramFormat.HARDWARE):
        with pytest.raises(ValueError, match="fresh Gallium"):
            capture_draw(FIXTURE, BUILD, tmp_path / target.value, target)


def test_fresh_plain_v1_capture_clears_inherited_texture_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("MGC_HOST_TEXTURE", "1")
    monkeypatch.setenv("MGC_HOST_TEXTURE_DATA", "/untrusted/inherited.bin")
    artifact = capture_draw(ROOT / "tests/fixtures/host-gradient", BUILD, tmp_path)
    assert artifact["schema"] == "mgc-host-draw-v1" and "textures" not in artifact
    assert all(shader["layout"]["version"] == 1 and "host_texture" not in shader["layout"]
               for shader in artifact["shaders"].values())
    image, coverage, report = render_artifact(artifact)
    assert image.shape == (16, 16, 4) and np.any(coverage)
    assert report["fpga_execution"] is False
    artifact["textures"] = [captured_texture()]
    with pytest.raises(ValueError):
        render_artifact(artifact)
    artifact["schema"] = "mgc-host-draw-v2"
    with pytest.raises(ValueError):
        render_artifact(artifact)


@pytest.mark.parametrize("contents", [b"", struct.pack("<II", 0, 1) + bytes(4),
                                     struct.pack("<II", 17, 1) + bytes(68),
                                     struct.pack("<II", 1, 1) + bytes(3),
                                     struct.pack("<II", 1, 1) + bytes(5),
                                     struct.pack(">II", 1, 1) + bytes(4),
                                     struct.pack("<II", 16, 16) + bytes(1025)])
def test_fresh_harness_rejects_malformed_texture_input_files(tmp_path, contents):
    texture = tmp_path / "texture.bin"
    texture.write_bytes(contents)
    with pytest.raises(ValueError, match="softpipe reference failed"):
        softpipe_reference(FIXTURE, BUILD, tmp_path / "output", texture=texture)


def test_fresh_harness_rejects_symlink_texture_file(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(struct.pack("<II", 1, 1) + bytes(4))
    texture = tmp_path / "texture.bin"
    texture.symlink_to(target)
    with pytest.raises(ValueError, match="softpipe reference failed"):
        softpipe_reference(FIXTURE, BUILD, tmp_path / "output", texture=texture)
