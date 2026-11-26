"""Actual legacy texture loads combined with explicit discard and bounded CFG."""

import copy
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from tools.host_render import FragmentStatus, ShaderExecutor, pack_float, render_artifact
from tools.host_texture import HOST_TEXTURE_LAYOUT, SAMPLER_STATE, TEXTURE_BASE
from tools.mgcdriver import Driver, ExecutionTimeoutError as DriverExecutionTimeoutError, ProgramFormat
from tools.mgcdriver.backends import _LegacyMemory
from tools.run_render import capture_draw
from tools.run_shader import ROOT, mesa_environment
from tools.shader_layout import ShaderLayout
from tools.texture_acceptance import quad_mesh, texture_pattern
from tools.texture_flow_acceptance import CASES, FIXTURE, STATUS, flow_oracle, main, validate


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
VARIANTS = ("late", "early", "runtime-loop")


def shader_directory(variant):
    return FIXTURE if variant == "late" else FIXTURE / variant


def fragment_inputs(layout, horizontal, vertical=0.25):
    return {entry["word_offset"]: pack_float((horizontal, vertical)[entry["component"]])
            for entry in layout.entries("inputs")}


def fragment_outputs(layout, texel):
    return {entry["word_offset"]: pack_float(int(texel[entry["component"]]) / 255)
            for entry in layout.entries("outputs")}


def captured_artifact(fresh_texture_flow, case):
    output, _, _ = fresh_texture_flow
    return json.loads((output / case / "draw.json").read_text())


@pytest.mark.parametrize("size", (4, 2))
def test_flow_oracle_coverage_is_not_inferred_from_rgba_or_alpha(size):
    texels = texture_pattern(size)
    expected, coverage = flow_oracle(texels)
    assert expected.shape == (16, 16, 4) and expected.dtype == np.uint8
    assert coverage.dtype == np.uint32
    assert np.all(expected[:, :8] == 0) and np.all(coverage[:, :8] == 0)
    assert np.all(coverage[:, 8:] == 1) and coverage.sum() == 128
    assert np.any((expected[:, :, 3] == 0) & (coverage == 1))
    if size == 2:
        assert np.all(expected[:8, 8:] == 0) and np.all(coverage[:8, 8:] == 1)
        assert np.all(expected[8:, 8:] == [0, 255, 0, 0])
    assert np.array_equal(texels, texture_pattern(size))


def test_command_exposes_output_and_existing_build_without_building(capsys):
    with pytest.raises(SystemExit) as exit_status:
        main(["--help"])
    assert exit_status.value.code == 0
    help_text = capsys.readouterr().out
    assert "--output" in help_text and "--build" in help_text


@pytest.fixture(scope="module")
def fresh_texture_flow(tmp_path_factory):
    output = tmp_path_factory.mktemp("fresh-texture-flow")
    submissions = []
    original_submit = Driver.submit

    def actual_submit(driver, program, *args, **kwargs):
        submissions.append(program.format)
        return original_submit(driver, program, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Driver, "submit", actual_submit)
        summary = validate(output, BUILD)
    return output, summary, submissions


@pytest.mark.parametrize("case,variant,size", CASES)
def test_fresh_same_input_softpipe_and_independent_texel_coverage_oracles(fresh_texture_flow, case, variant, size):
    output, summary, submissions = fresh_texture_flow
    directory = output / case
    artifact = captured_artifact(fresh_texture_flow, case)
    texels = texture_pattern(size)
    expected, expected_coverage = flow_oracle(texels)
    image = np.frombuffer((directory / "host.rgba8").read_bytes(), dtype=np.uint8).reshape(16, 16, 4)
    reference = np.frombuffer((directory / "softpipe.rgba8").read_bytes(), dtype=np.uint8).reshape(16, 16, 4)[::-1]
    coverage = np.frombuffer((directory / "coverage.u32").read_bytes(), dtype="<u4").reshape(16, 16)
    assert np.array_equal(image, expected) and np.array_equal(reference, expected)
    assert np.array_equal(coverage, expected_coverage)
    assert (directory / "expected.rgba8").read_bytes() == expected.tobytes()
    assert (directory / "texture.bin").read_bytes() == struct.pack("<II", size, size) + texels.tobytes()
    assert (directory / "vertices.bin").read_bytes() == quad_mesh()
    assert len(quad_mesh()) == 6 * 7 * 4
    assert artifact["textures"] == [{"width": size, "height": size, "format": "R8G8B8A8_UNORM",
                                     "data_hex": texels.tobytes().hex(), "sampler": dict(SAMPLER_STATE)}]
    report = next(result for result in summary["cases"] if result["case"] == case)
    assert report["fixture"] == variant and report["texture_dimensions"] == [size, size]
    assert report["vertices"] == 6 and report["fragments"] == 256
    assert report["committed_fragments"] == report["discarded_fragments"] == 128
    assert report["covered_pixels"] == 128 and report["max_overdraw"] == 1
    assert report["host_steps"] > 256 and report["fpga_execution"] is False
    assert report["softpipe_comparison"]["passed"] is True
    assert report["softpipe_comparison"]["max_channel_error"] == 0
    assert report["softpipe_comparison"]["mismatch_pixels"] == 0
    assert report["softpipe_comparison"]["channel_tolerance"] == 0
    assert "never RGBA/alpha sentinel inference" in report["coverage_oracle"]
    assert json.loads((directory / "render.json").read_text()) == {key: value for key, value in report.items() if key != "case"}
    assert json.loads((output / "summary.json").read_text()) == summary
    for filename in ("capture.log", "softpipe.log", "host.png", "softpipe.png"):
        assert (directory / filename).is_file()
    assert len(submissions) == len(CASES) * (6 + 256)
    assert all(target == ProgramFormat.LEGACY for target in submissions)


@pytest.mark.parametrize("case,variant,size", CASES)
def test_actual_header_layout_v2_and_explicit_status_v1_remain_legacy(fresh_texture_flow, case, variant, size):
    artifact = captured_artifact(fresh_texture_flow, case)
    assert artifact["schema"] == "mgc-host-draw-v2"
    assert artifact["program_format"] == ProgramFormat.LEGACY.value
    assert artifact["shader_abi_version"] == 1
    for stage, version in (("vs", 1), ("fs", 2)):
        shader = artifact["shaders"][stage]
        layout = ShaderLayout(shader["layout"])
        assert shader["assembly"].startswith("; mgc legacy-simulator backend, ABI=1,")
        assert layout.target == ProgramFormat.LEGACY and layout.source["version"] == version
        assert layout.source["shader_abi_version"] == 1 and layout.source["address_units"] == "words"
        assert layout.host_texture == (HOST_TEXTURE_LAYOUT if stage == "fs" else None)
        assert layout.fragment_status == (STATUS if stage == "fs" else None)
    fragment = artifact["shaders"]["fs"]
    assert fragment["layout"]["uses_discard"] is True
    assert all(entry["word_offset"] != STATUS["word_offset"] for entry in fragment["layout"]["outputs"])
    if variant == "runtime-loop":
        assert "mgc_block_" in fragment["assembly"]
        assert (shader_directory(variant) / "frag.glsl").read_text().count("texture2D(") == 1


@pytest.mark.parametrize("case,variant,size", CASES)
def test_actual_discard_returns_empty_outputs_and_alpha_zero_commits(fresh_texture_flow, case, variant, size):
    artifact = captured_artifact(fresh_texture_flow, case)
    shader = artifact["shaders"]["fs"]
    layout = ShaderLayout(shader["layout"])
    texels = texture_pattern(size)
    texel = texels[int(0.25 * size), int(0.75 * size)]
    assert texel[3] == 0 and np.any(texel[:3] != 0)
    executor = ShaderExecutor(shader["assembly"], layout, texture=artifact["textures"])
    try:
        for horizontal in (0.75, 0.25, 0.75, 0.25):
            result = executor.invoke_fragment(fragment_inputs(layout, horizontal))
            if horizontal < 0.5:
                assert result.status is FragmentStatus.DISCARDED and result.outputs == {}
            else:
                assert result.status is FragmentStatus.COMMITTED
                assert result.outputs == fragment_outputs(layout, texel)
        assert executor.invocations == 4 and executor.steps > 4
    finally:
        executor.close()


@pytest.mark.parametrize("variant", VARIANTS)
def test_one_changed_texel_is_loaded_from_actual_managed_bytearray_between_invocations(fresh_texture_flow, monkeypatch, variant):
    artifact = captured_artifact(fresh_texture_flow, f"{variant}-4x4")
    shader = artifact["shaders"]["fs"]
    layout = ShaderLayout(shader["layout"])
    original_textures = copy.deepcopy(artifact["textures"])
    executor = ShaderExecutor(shader["assembly"], layout, texture=artifact["textures"])
    uniform = executor.buffers["uniform"]
    texel_offset = TEXTURE_BASE + 4 + (1 * 4 + 3) * 4
    addresses = {uniform.address // 4 + texel_offset + component for component in range(4)}
    reads = []
    original_load = _LegacyMemory.load

    def actual_load(memory, word_address):
        value = original_load(memory, word_address)
        if word_address in addresses:
            reads.append((word_address, value))
        return value

    monkeypatch.setattr(_LegacyMemory, "load", actual_load)
    try:
        assert isinstance(uniform._data, bytearray)
        words = executor.program.words
        inputs = fragment_inputs(layout, 0.75)
        original_texel = texture_pattern(4)[1, 3]
        first = executor.invoke_fragment(inputs)
        assert first.status is FragmentStatus.COMMITTED
        assert first.outputs == fragment_outputs(layout, original_texel)
        assert {address for address, _ in reads} == addresses and len(reads) == 4
        changed_texel = np.array([91, 23, 211, 0], dtype=np.uint8)
        encoded = struct.pack("<4I", *(pack_float(int(channel) / 255) for channel in changed_texel))
        before = bytes(uniform._data)
        uniform.upload(encoded, offset=texel_offset * 4)
        after = bytes(uniform._data)
        assert after[:texel_offset * 4] == before[:texel_offset * 4]
        assert after[texel_offset * 4:texel_offset * 4 + 16] == encoded
        assert after[texel_offset * 4 + 16:] == before[texel_offset * 4 + 16:]
        reads.clear()
        second = executor.invoke_fragment(inputs)
        assert second.status is FragmentStatus.COMMITTED
        assert second.outputs == fragment_outputs(layout, changed_texel) and second.outputs != first.outputs
        assert dict(reads) == {uniform.address // 4 + texel_offset + component: pack_float(int(channel) / 255)
                               for component, channel in enumerate(changed_texel)}
        assert len(reads) == 4
        assert executor.program.words == words and artifact["textures"] == original_textures
        discarded = executor.invoke_fragment(fragment_inputs(layout, 0.25))
        assert discarded.status is FragmentStatus.DISCARDED and discarded.outputs == {}
    finally:
        executor.close()


def test_one_static_runtime_loop_sample_executes_multiple_actual_cpu_loads(fresh_texture_flow, monkeypatch):
    artifact = captured_artifact(fresh_texture_flow, "runtime-loop-4x4")
    shader = artifact["shaders"]["fs"]
    layout = ShaderLayout(shader["layout"])
    assert "mgc_block_" in shader["assembly"]
    assert (FIXTURE / "runtime-loop/frag.glsl").read_text().count("texture2D(") == 1
    executor = ShaderExecutor(shader["assembly"], layout, texture=artifact["textures"])
    base = executor.buffers["uniform"].address // 4 + TEXTURE_BASE + 4
    reads = []
    original_load = _LegacyMemory.load

    def actual_load(memory, word_address):
        if base <= word_address < base + 4 * 4 * 4:
            reads.append(word_address - base)
        return original_load(memory, word_address)

    monkeypatch.setattr(_LegacyMemory, "load", actual_load)
    try:
        for vertical, iterations in ((0.25, 1), (0.75, 3)):
            reads.clear()
            result = executor.invoke_fragment(fragment_inputs(layout, 0.75, vertical))
            row = int(vertical * 4)
            assert result.status is FragmentStatus.COMMITTED
            assert result.outputs == fragment_outputs(layout, texture_pattern(4)[row, 3])
            assert sorted(reads) == sorted([(row * 4 + 3) * 4 + component
                                            for component in range(4)] * iterations)
        assert executor.invocations == 2
    finally:
        executor.close()


@pytest.mark.parametrize("horizontal", (0.25, 0.75))
def test_runtime_loop_budget_failure_is_driver_timeout_not_discard(fresh_texture_flow, monkeypatch, horizontal):
    artifact = captured_artifact(fresh_texture_flow, "runtime-loop-4x4")
    shader = artifact["shaders"]["fs"]
    layout = ShaderLayout(shader["layout"])
    executor = ShaderExecutor(shader["assembly"], layout, texture=artifact["textures"], max_steps=1)

    def forbidden_download(*args, **kwargs):
        pytest.fail("failed managed execution must not read fragment status or outputs")

    monkeypatch.setattr(executor.buffers["fragment"], "download", forbidden_download)
    try:
        with pytest.raises(DriverExecutionTimeoutError, match="max_steps"):
            executor.invoke_fragment(fragment_inputs(layout, horizontal))
        assert executor.invocations == 0 and executor.steps == 0
    finally:
        executor.close()


@pytest.mark.parametrize("budget", ({"max_steps": 1}, {"max_fragments": 1}))
def test_texture_flow_draw_budgets_fail_closed(fresh_texture_flow, budget):
    with pytest.raises(ValueError, match="budget"):
        render_artifact(captured_artifact(fresh_texture_flow, "runtime-loop-4x4"), **budget)


@pytest.mark.parametrize("mutation", ("wire-v1", "unknown-schema", "layout-v1", "missing-resource",
                                      "short-resource", "wrong-format", "linear-sampler", "missing-status",
                                      "status-v2", "hardware-target", "wrong-header", "status-output-alias", "ubo15"))
def test_actual_texture_discard_artifact_rejects_invalid_schema_resource_and_status(fresh_texture_flow, mutation):
    artifact = captured_artifact(fresh_texture_flow, "runtime-loop-4x4")
    shader = artifact["shaders"]["fs"]
    if mutation == "wire-v1":
        artifact["schema"] = "mgc-host-draw-v1"
    elif mutation == "unknown-schema":
        artifact["schema"] = "mgc-host-draw-v3"
    elif mutation == "layout-v1":
        shader["layout"]["version"] = 1
    elif mutation == "missing-resource":
        del artifact["textures"]
    elif mutation == "short-resource":
        artifact["textures"][0]["data_hex"] = artifact["textures"][0]["data_hex"][:-2]
    elif mutation == "wrong-format":
        artifact["textures"][0]["format"] = "R8G8B8A8_SRGB"
    elif mutation == "linear-sampler":
        artifact["textures"][0]["sampler"]["min_filter"] = "linear"
    elif mutation == "missing-status":
        del shader["layout"]["fragment_status"]
    elif mutation == "status-v2":
        shader["layout"]["fragment_status"]["version"] = 2
    elif mutation == "hardware-target":
        artifact["program_format"] = ProgramFormat.HARDWARE.value
    elif mutation == "wrong-header":
        shader["assembly"] = shader["assembly"].replace("ABI=1,", "ABI=2,", 1)
    elif mutation == "status-output-alias":
        shader["layout"]["outputs"][0]["word_offset"] = STATUS["word_offset"]
    elif mutation == "ubo15":
        artifact["constants"]["fs"] = [None] * 15 + ["00000000"]
    with pytest.raises(ValueError):
        render_artifact(artifact)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("target,host_texture,fragment_status", (
    (ProgramFormat.LEGACY, None, True), (ProgramFormat.LEGACY, False, True),
    (ProgramFormat.LEGACY, True, False), (ProgramFormat.HARDWARE, True, True),
))
def test_fresh_compile_requires_texture_and_status_opt_in_and_legacy_target(tmp_path, variant, target, host_texture, fragment_status):
    shader = shader_directory(variant)
    dump = tmp_path / "assembly"
    dump.mkdir()
    environment = mesa_environment(BUILD, dump, target)
    if host_texture is not None:
        environment["MGC_HOST_TEXTURE"] = "1" if host_texture else "0"
    if fragment_status:
        environment["MGC_FRAGMENT_STATUS"] = "1"
    completed = subprocess.run([str(ROOT / "tools/mgc_harness"), str(shader / "vert.glsl"),
                                str(shader / "frag.glsl")], env=environment,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode != 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ("status-disabled", "default-texture-disabled", "hardware", "hardware-opt-in"))
def test_fresh_capture_rejects_status_opt_out_and_default_or_hardware_texture(tmp_path, variant, mode):
    mesh = tmp_path / "vertices.bin"
    mesh.write_bytes(quad_mesh())
    texture = tmp_path / "texture.bin"
    texture.write_bytes(struct.pack("<II", 4, 4) + texture_pattern(4).tobytes())
    options = {"mesh": mesh}
    if mode in ("status-disabled", "hardware-opt-in"):
        options.update(host_texture=True, texture=texture)
    if mode == "status-disabled":
        options["fragment_status"] = False
    if mode.startswith("hardware"):
        options["target"] = ProgramFormat.HARDWARE
    output = tmp_path / "capture"
    with pytest.raises(ValueError, match="fresh Gallium|unsupported on hardware"):
        capture_draw(shader_directory(variant), BUILD, output, **options)
    assert not (output / "draw.json").exists()
