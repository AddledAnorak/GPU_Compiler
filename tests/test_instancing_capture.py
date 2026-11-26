"""Direct C ABI3 capture and real NIR instance-ID lowering; no native-render claims."""

import json
import os
from pathlib import Path
import struct
import subprocess

import pytest

from test_texture_capture import build_texture_guard, read_capture
from tools.mgcasm import ISA, assemble
from tools.mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from tools.mgcsim import CPU


SOURCE = Path(__file__).parent / "fixtures/instancing-capture/guard.c"
HOST_INSTANCING = {
    "version": 1, "max_instances": 16, "max_vertices": 4096,
    "indexing": "base-instance-plus-floor-local-instance-over-divisor",
    "shader_abi_version": 3, "instance_id_word": 65535,
}
SHADER_INSTANCING = {
    "version": 1, "max_instances": 16, "instance_id_word": 65535,
    "instance_id": "local-zero-based", "uses_instance_id": False,
}
VALID = (
    "valid", "single", "single_divisor", "base", "floor", "per_vertex", "mixed_divisors",
    "max_instances", "max_budget", "max_base", "max_divisor", "offset", "env_after_creation",
    "indexed_1", "indexed_2", "indexed_4", "indexed_divisor", "indexed_per_vertex",
    "indexed_per_vertex_base", "indexed_positive_bias", "indexed_start", "indexed_ownership",
)
INVALID = (
    "large_instances", "zero_instances", "zero_vertices", "large_budget", "large_base",
    "base_overflow", "large_divisor", "divisor_unknown", "dual_slot", "restart", "lines",
    "short_vertex", "short_instance", "short_base", "short_floor", "batch_invalid",
    "indexed_short", "indexed_negative_bias", "indexed_short_ownership", "vertex_screen",
    "vertex_target", "vertex_storage", "vertex_null", "vertex_mapped", "vertex_offset",
    "hardware_vertex", "hardware_fragment", "unknown_version", "unsupported_reflection",
    "custom", "custom_single", "callback_disabled",
)


@pytest.fixture(scope="module")
def instancing_guard(tmp_path_factory):
    return build_texture_guard(tmp_path_factory, SOURCE)


def run_instancing_guard(helper, directory, case, option="1", texture=None, target=None):
    environment = dict(os.environ)
    for key in ("MGC_HOST_INSTANCING", "MGC_HOST_TEXTURE", "MGC_HOST_TEXTURE_WRAP_S",
                "MGC_HOST_TEXTURE_WRAP_T", "MGC_FRAGMENT_STATUS", "MGC_COMPILE_TARGET",
                "MGC_HOST_MESH", "MGC_ASM_DUMP", "MGC_HOST_EXECUTOR_SOCKET"):
        environment.pop(key, None)
    if option is not None:
        environment["MGC_HOST_INSTANCING"] = option
    if texture:
        environment["MGC_HOST_TEXTURE"] = str(texture)
        if texture in (7, 8):
            environment["MGC_HOST_TEXTURE_WRAP_S"] = "repeat"
            environment["MGC_HOST_TEXTURE_WRAP_T"] = "mirrored-repeat"
    if target:
        environment["MGC_COMPILE_TARGET"] = target
    environment["MGC_HOST_CAPTURE_DIR"] = str(directory)
    result = subprocess.run([str(helper), case], env=environment, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("case", VALID)
def test_opt_in_capture_actual_counts_base_divisors_and_owned_buffers(instancing_guard, tmp_path, case):
    status = run_instancing_guard(instancing_guard, tmp_path, case)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["callbacks"] == 0 and status["init_error"] == 0
    artifact = read_capture(tmp_path)
    assert artifact["schema"] == "mgc-host-draw-v3" and artifact["host_instancing"] == HOST_INSTANCING
    assert artifact["shader_abi_version"] == 3 and artifact["program_format"] == "legacy-mgcsim-v1"
    assert artifact["drawid"] == 7 and artifact["draw"]["count"] * artifact["draw"]["instance_count"] <= 4096
    for shader in artifact["shaders"].values():
        assert shader["layout"]["version"] == 3
        assert shader["layout"]["shader_abi_version"] == 3
        assert shader["layout"]["host_instancing"] == SHADER_INSTANCING
        assert shader["assembly"].startswith("; mgc legacy-simulator backend, ABI=3,")
    draw = artifact["draw"]
    assert draw["instance_count"] == {"single": 1, "single_divisor": 1, "floor": 5,
                                      "max_instances": 16, "max_budget": 16}.get(case, 4)
    assert draw["start_instance"] == {"base": 2, "floor": 2, "max_base": 1048576,
                                      "indexed_per_vertex_base": 2}.get(case, 0)
    assert draw["count"] == (256 if case == "max_budget" else 3)
    per_vertex = {"single", "per_vertex", "mixed_divisors", "max_budget", "indexed_per_vertex",
                  "indexed_per_vertex_base", "indexed_positive_bias"}
    expected_divisor = 0 if case in per_vertex else {"floor": 3, "max_divisor": 1048576}.get(case, 1)
    assert artifact["vertex_elements"][0]["instance_divisor"] == expected_divisor
    if case == "mixed_divisors":
        assert artifact["vertex_elements"][1]["instance_divisor"] == 2
    for element in artifact["vertex_elements"]:
        buffer = artifact["vertex_buffers"][element["vertex_buffer_index"]]
        payload = bytes.fromhex(buffer["data_hex"])
        divisor = element["instance_divisor"]
        for local_instance in range(draw["instance_count"]):
            row = draw["start_instance"] + local_instance // divisor if divisor else draw["count"] - 1
            offset = element["src_offset"] + row * buffer["stride"]
            assert offset + 16 <= len(payload)
            actual, = struct.unpack_from("<f", payload, offset)
            assert actual == (row if buffer["stride"] else 0)
    if case.startswith("indexed"):
        assert draw["index_size"] in (1, 2, 4) and artifact["indices"]


@pytest.mark.parametrize("case", INVALID)
def test_guards_reject_without_callbacks_files_or_sequence(instancing_guard, tmp_path, case):
    status = run_instancing_guard(instancing_guard, tmp_path, case)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []
    if case != "unsupported_reflection":
        assert status["sequence"] == 0


@pytest.mark.parametrize("option", (None, "0"))
@pytest.mark.parametrize("case", ("valid", "base", "single_divisor", "mixed_divisors"))
def test_default_still_rejects_instancing(instancing_guard, tmp_path, option, case):
    status = run_instancing_guard(instancing_guard, tmp_path, case, option)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("option", (None, "0"))
@pytest.mark.parametrize("texture", (None, 1, 7, 8))
def test_default_keeps_v1_v2_without_metadata(instancing_guard, tmp_path, option, texture):
    status = run_instancing_guard(instancing_guard, tmp_path, "single", option, texture)
    assert status["accepted"] == 1 and status["error"] == 0
    artifact = read_capture(tmp_path)
    assert artifact["schema"] == ("mgc-host-draw-v2" if texture else "mgc-host-draw-v1")
    assert "host_instancing" not in artifact


@pytest.mark.parametrize("texture", (1, 7, 8))
def test_instancing_and_texture_uses_v3(instancing_guard, tmp_path, texture):
    status = run_instancing_guard(instancing_guard, tmp_path, "valid", texture=texture)
    assert status["accepted"] == 1 and status["error"] == 0
    artifact = read_capture(tmp_path)
    assert artifact["schema"] == "mgc-host-draw-v3" and artifact["host_instancing"] == HOST_INSTANCING
    assert artifact["shader_abi_version"] == 3
    assert artifact["shaders"]["fs"]["layout"]["version"] == 3
    assert artifact["shaders"]["fs"]["layout"]["host_texture"]["version"] == texture
    assert len(artifact["textures"]) == 1


@pytest.mark.parametrize("option", ("", "2", "01", "+1", "-1", "1.0", " 1", "1 ", "true", "unknown"))
def test_malformed_opt_in_remains_invalid_after_clear_error(instancing_guard, tmp_path, option):
    status = run_instancing_guard(instancing_guard, tmp_path, "malformed_env", option)
    assert status["init_error"] != 0 and status["error"] != 0 and status["rejected"] == 1
    assert status["accepted"] == status["callbacks"] == status["sequence"] == 0
    assert list(tmp_path.iterdir()) == []


def test_hardware_capture_only_still_rejects_instancing(instancing_guard, tmp_path):
    status = run_instancing_guard(instancing_guard, tmp_path, "single", target="fp16-hardware-v1")
    assert status["accepted"] == 0 and status["compile_rejected"] == 1 and status["error"] != 0
    assert status["sequence"] == 0 and list(tmp_path.iterdir()) == []


def test_default_public_callback_abi_is_preserved(instancing_guard, tmp_path):
    status = run_instancing_guard(instancing_guard, tmp_path, "custom_single", option=None)
    assert status["accepted"] == 1 and status["callbacks"] == 1 and status["error"] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("option", (None, "0"))
def test_default_generic_callback_still_accepts_bounded_instancing(instancing_guard, tmp_path, option):
    status = run_instancing_guard(instancing_guard, tmp_path, "custom", option=option)
    assert status["accepted"] == 1 and status["rejected"] == 0 and status["error"] == 0
    assert status["callbacks"] == 1 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ("instance_id", "instance_id_base", "instance_id_max_instances",
                                  "instance_id_indexed_divisor"))
def test_genuine_instance_id_is_local_raw_reserved_input(instancing_guard, tmp_path, case):
    status = run_instancing_guard(instancing_guard, tmp_path, case)
    assert status["accepted"] == 1 and status["error"] == 0
    artifact = read_capture(tmp_path)
    assert artifact["shader_abi_version"] == 3 and artifact["host_instancing"] == HOST_INSTANCING
    assert artifact["shaders"]["vs"]["layout"]["host_instancing"] == dict(SHADER_INSTANCING, uses_instance_id=True)
    assert artifact["shaders"]["fs"]["layout"]["host_instancing"] == SHADER_INSTANCING
    words = assemble(artifact["shaders"]["vs"]["assembly"], ISA())
    for local_id in range(16):
        cpu = CPU(isa=ISA(), max_cycles=100000)
        for component, value in enumerate((2, 0, 0, 1)):
            cpu.mem.store(0x10000 + component, float_to_fp16_bits(value))
        cpu.mem.store(0x10000 + 65535, local_id)
        cpu.run(words)
        assert cpu.halted
        assert fp16_bits_to_float(cpu.mem.load(0x30000) & 65535) == 2 + local_id
    assert artifact["draw"]["start_instance"] == (2 if case == "instance_id_base" else 0)


@pytest.mark.parametrize("option", (None, "0"))
def test_instance_id_builtin_without_opt_in_is_rejected(instancing_guard, tmp_path, option):
    status = run_instancing_guard(instancing_guard, tmp_path, "instance_id", option)
    assert status["compile_rejected"] == 1 and status["error"] != 0
    assert status["accepted"] == status["callbacks"] == status["sequence"] == 0
    assert list(tmp_path.iterdir()) == []


METADATA_FIELDS = ("version", "max_instances", "instance_id_word", "instance_id", "uses_instance_id")
METADATA_CASES = (
    "object_missing", "object_null", "object_duplicate", "unknown", "layout1", "layout2", "layout4",
    "version_0", "version_2", "version_1.0", 'version_"1"', "max_instances_15", "max_instances_17",
    "instance_id_word_65534", "instance_id_word_65536", "instance_id_word_65535.0",
    'instance_id_"base-plus-local"', 'instance_id_"unknown"', "instance_id_null",
    "uses_instance_id_true", 'uses_instance_id_"false"', "uses_instance_id_0", "uses_instance_id_null",
) + tuple(f"{operation}_{field}" for operation in ("missing", "duplicate") for field in METADATA_FIELDS)


@pytest.mark.parametrize("stage", ("vs", "fs"))
@pytest.mark.parametrize("case", METADATA_CASES)
def test_exact_shader_instancing_metadata_is_required(instancing_guard, tmp_path, stage, case):
    status = run_instancing_guard(instancing_guard, tmp_path, f"metadata_{stage}_{case}")
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["sequence"] == status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage", ("vs", "fs"))
@pytest.mark.parametrize("case", ("option_zero", "option_two", "abi1", "abi2", "id_flag"))
def test_both_stage_options_abi_and_actual_id_flags_must_agree(instancing_guard, tmp_path, stage, case):
    status = run_instancing_guard(instancing_guard, tmp_path, f"shader_{stage}_{case}")
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == status["sequence"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ("instance_id_shader_vs_id_flag", "instance_id_metadata_vs_uses_instance_id_false"))
def test_genuine_id_use_cannot_be_hidden(instancing_guard, tmp_path, case):
    status = run_instancing_guard(instancing_guard, tmp_path, case)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage", ("vs", "fs"))
def test_abi1_assembly_header_cannot_enter_abi3_capture(instancing_guard, tmp_path, stage):
    status = run_instancing_guard(instancing_guard, tmp_path, f"header_{stage}_abi1")
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("texture", (None, 1, 7))
@pytest.mark.parametrize("case", ("default_metadata_object", "default_layout3", "default_header3",
                                  "shader_vs_option_one", "shader_fs_option_one", "shader_vs_id_flag"))
def test_default_layouts_do_not_silently_ignore_instancing_metadata(instancing_guard, tmp_path, texture, case):
    status = run_instancing_guard(instancing_guard, tmp_path, case, option=None, texture=texture)
    assert status["accepted"] == 0 and status["rejected"] == 1 and status["error"] != 0
    assert status["callbacks"] == 0 and list(tmp_path.iterdir()) == []
