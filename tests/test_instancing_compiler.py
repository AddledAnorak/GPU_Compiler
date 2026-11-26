"""Fresh opt-in ABI 3 InstanceID NIR without rebuilding the shared GL library."""

import json
import os
import re
import subprocess

import pytest

from test_backend_runtime import backend_binary
from tools.mgcasm import assemble, disassemble
from tools.mgcasm.fp16 import float_to_fp16_bits
from tools.mgcsim import CPU
from tools.mgcsim.cpu import MgcSimError


INSTANCE_WORD = 65535
INSTANCE_MODES = ("instance-id", "instance-vector", "instance-float", "instance-select",
                  "instance-select-constant", "instance-branch", "instance-loop", "instance-loop-phi")


def compile_instance(binary, mode, *, option="host-instancing", count=4, environment=None, success=True):
    process_environment = dict(os.environ)
    for variable in ("MGC_TEST_HOST_INSTANCING", "MGC_TEST_WRAP_S", "MGC_TEST_WRAP_T",
                     "MGC_TEST_SOURCE_LOD", "MGC_NIR_DUMP"):
        process_environment.pop(variable, None)
    process_environment["MGC_TEST_LAYOUT"] = "1"
    process_environment.update(environment or {})
    command = [str(binary), mode, str(count)]
    if option is not None:
        command.append(option)
    result = subprocess.run(command, env=process_environment, capture_output=True, text=True, timeout=30)
    if not success:
        assert result.returncode == 1, result.stderr
        assert not result.stdout and "error exposed partial assembly" not in result.stderr
        return result.stderr
    assert result.returncode == 0, result.stderr
    layout = json.loads(next(line.removeprefix("layout=") for line in result.stderr.splitlines()
                             if line.startswith("layout=")))
    statistics = {name: int(value) for name, value in re.findall(r"(\w+)=(\d+)", result.stderr)}
    words = None
    if layout["program_format"] == "legacy-mgcsim-v1":
        words = assemble(result.stdout)
        assert assemble("\n".join(disassemble(words))) == words
    return {"assembly": result.stdout, "layout": layout, "stats": statistics, "words": words,
            "mode": mode}


@pytest.fixture(scope="module", params=INSTANCE_MODES)
def instance_program(backend_binary, request):
    return compile_instance(backend_binary, request.param)


def instance_cpu(program, instance_id, *, inputs=None, max_cycles=10000):
    layout = program["layout"]
    input_base = layout["bases"]["input"]
    output_base = layout["bases"][layout["stage"]]
    cpu = CPU(max_cycles=max_cycles)
    if instance_id is not None:
        cpu.mem.store(input_base + INSTANCE_WORD, instance_id)
    payload = inputs or {}
    assert set(payload) == {entry["word_offset"] for entry in layout["inputs"]}
    for offset, value in payload.items():
        cpu.mem.store(input_base + offset, value)
    original_load, original_store = cpu.mem.load, cpu.mem.store
    reads, writes = [], []
    allowed_reads = {input_base + offset for offset in payload}
    if layout["host_instancing"]["uses_instance_id"]:
        allowed_reads.add(input_base + INSTANCE_WORD)
    offsets = [entry["word_offset"] for entry in layout["outputs"]]
    allowed_writes = {output_base + offset for offset in offsets}

    def checked_load(address):
        assert address in allowed_reads
        reads.append(address)
        return original_load(address)

    def checked_store(address, value):
        assert address in allowed_writes
        writes.append(address)
        original_store(address, value)

    def output_words():
        return {offset: original_load(output_base + offset) for offset in offsets}

    cpu.mem.load, cpu.mem.store = checked_load, checked_store
    return cpu, reads, writes, output_words


def expected_outputs(mode, instance_id):
    if mode == "instance-vector":
        return {component: instance_id + component for component in range(4)}
    if mode == "instance-float":
        return {0: float_to_fp16_bits(instance_id), 1: float_to_fp16_bits(instance_id + 0.5),
                2: float_to_fp16_bits(instance_id >= 8), 3: float_to_fp16_bits(1)}
    if mode == "instance-select":
        return {0: instance_id if instance_id < 8 else 15}
    if mode == "instance-select-constant":
        return {0: 3 if instance_id < 8 else 11}
    if mode == "instance-branch":
        return {0: instance_id if instance_id < 8 else instance_id + 16}
    if mode == "instance-loop":
        return {0: instance_id, 1: instance_id, 2: instance_id}
    if mode == "instance-loop-phi":
        return {0: instance_id + (3 if instance_id < 8 else 11), 1: instance_id, 2: instance_id}
    return {0: instance_id}


def test_exact_opt_in_vertex_header_layout_and_raw_reserved_word(instance_program):
    layout = instance_program["layout"]
    assert instance_program["assembly"].startswith(
        "; mgc legacy-simulator backend, ABI=3, stage=MESA_SHADER_VERTEX\n")
    assert layout["version"] == layout["shader_abi_version"] == 3
    assert layout["program_format"] == "legacy-mgcsim-v1" and layout["address_units"] == "words"
    assert layout["stage"] == "vertex" and layout["supported"] is True and layout["errors"] == []
    assert layout["bases"] == {"input": 0x10000, "uniform": 0x20000, "vertex": 0x30000,
                               "fragment": 0x100000}
    assert layout["host_instancing"] == {
        "version": 1, "max_instances": 16, "instance_id_word": INSTANCE_WORD,
        "instance_id": "local-zero-based", "uses_instance_id": True,
    }
    assert list(layout["host_instancing"]) == ["version", "max_instances", "instance_id_word",
                                               "instance_id", "uses_instance_id"]
    assert layout["inputs"] == [] and layout["uniforms"] == []
    assert all(entry["type"] == ("float" if instance_program["mode"] == "instance-float" else "int")
               and entry["bit_size"] == 32 for entry in layout["outputs"])
    assert set(layout) == {"version", "program_format", "shader_abi_version", "stage", "address_units",
                           "bases", "host_instancing", "inputs", "outputs", "uniforms", "uses_discard",
                           "supported", "errors"}
    assert instance_program["stats"]["abi"] == 3
    assert instance_program["stats"]["host_instancing_version"] == 1
    assert instance_program["stats"]["uses_host_instance_id"] == 1
    assert 0 < instance_program["stats"]["registers"] <= 32
    assert 0 <= instance_program["stats"]["predicates"] <= 8
    assert 0 < instance_program["stats"]["instructions"] < 1000


@pytest.mark.parametrize("instance_id", range(16))
def test_raw_id_conversion_branch_select_and_integer_loop_execution(instance_program, instance_id):
    cpu, reads, writes, output_words = instance_cpu(instance_program, instance_id)
    cpu.run(instance_program["words"])
    assert cpu.halted and cpu.cycles < cpu.max_cycles
    assert output_words() == expected_outputs(instance_program["mode"], instance_id)
    assert reads and set(reads) == {instance_program["layout"]["bases"]["input"] + INSTANCE_WORD}
    assert len(writes) == len(instance_program["layout"]["outputs"])


def test_execution_budget_never_commits_partial_instance_output(instance_program):
    cpu, _, writes, output_words = instance_cpu(instance_program, 15, max_cycles=1)
    with pytest.raises(MgcSimError, match="exceeded max_cycles=1"):
        cpu.run(instance_program["words"])
    assert not cpu.halted and writes == []
    assert all(value == 0 for value in output_words().values())


@pytest.mark.parametrize("mode", ("instance-branch", "instance-loop", "instance-loop-phi"))
def test_real_cfg_paths_and_parallel_integer_phi_copies_are_exercised(backend_binary, mode):
    program = compile_instance(backend_binary, mode)
    assert program["stats"]["blocks"] > 1
    assert "mgc_block_" in program["assembly"] and "mgc_edge_" in program["assembly"]
    assert "icmp_" in program["assembly"]
    if "loop" in mode:
        assert "add " in program["assembly"]


@pytest.mark.parametrize("mode", ("instance-no-id", "instance-fragment-no-id", "instance-unused"))
def test_both_opt_in_stages_have_metadata_even_without_live_instance_id(backend_binary, mode):
    program = compile_instance(backend_binary, mode)
    layout = program["layout"]
    assert layout["version"] == layout["shader_abi_version"] == 3 and layout["supported"] is True
    assert layout["host_instancing"]["uses_instance_id"] is False
    assert program["stats"]["uses_host_instance_id"] == 0
    inputs = {entry["word_offset"]: float_to_fp16_bits(0.625) for entry in layout["inputs"]}
    cpu, reads, _, output_words = instance_cpu(program, None if layout["stage"] == "fragment" else 7,
                                               inputs=inputs)
    cpu.run(program["words"])
    assert cpu.halted and output_words() == {0: float_to_fp16_bits(1 if mode == "instance-unused" else 0.625)}
    assert layout["bases"]["input"] + INSTANCE_WORD not in reads


@pytest.mark.parametrize("mode", ("instance-id", "instance-unused"))
@pytest.mark.parametrize("option", (None, "hardware"))
def test_default_and_hardware_reject_even_dead_instance_intrinsics(backend_binary, mode, option):
    diagnostic = compile_instance(backend_binary, mode, option=option, success=False)
    assert "load_instance_id requires scalar raw32 host instancing version 1 in a legacy vertex shader" in diagnostic


@pytest.mark.parametrize("mode", ("instance-fragment", "instance-compute"))
def test_instance_id_is_vertex_stage_only(backend_binary, mode):
    diagnostic = compile_instance(backend_binary, mode, success=False)
    assert "legacy vertex shader" in diagnostic or "only fragment and vertex shader stages" in diagnostic


@pytest.mark.parametrize("mode", ("instance-id16", "instance-id64", "instance-id-vector"))
def test_instance_id_is_scalar_raw32_only(backend_binary, mode):
    diagnostic = compile_instance(backend_binary, mode, success=False)
    assert "load_instance_id requires scalar raw32" in diagnostic


@pytest.mark.parametrize("mode", ("instance-base", "instance-vertex-id", "instance-draw-id"))
def test_other_builtins_remain_unsupported(backend_binary, mode):
    diagnostic = compile_instance(backend_binary, mode, success=False)
    assert "host instancing does not support builtin" in diagnostic


@pytest.mark.parametrize("option", ("host-instancing-unknown", "hardware-host-instancing-unknown"))
@pytest.mark.parametrize("version", (2, 99, -1, 0xFFFFFFFF))
def test_unknown_instancing_versions_reject_before_assembly(backend_binary, option, version):
    diagnostic = compile_instance(backend_binary, "instance-no-id", option=option, count=version, success=False)
    assert "host instancing version is unsupported for this target" in diagnostic


def test_hardware_rejects_opt_in_even_without_instance_usage(backend_binary):
    diagnostic = compile_instance(backend_binary, "instance-no-id", option="hardware-host-instancing", success=False)
    assert "host instancing version is unsupported for this target" in diagnostic


@pytest.mark.parametrize("mode", ("instance-input-alias", "instance-input-vector-alias",
                                 "instance-input-alias-unused"))
def test_opt_in_reserves_word_even_without_builtin_usage(backend_binary, mode):
    diagnostic = compile_instance(backend_binary, mode, success=False)
    assert "ordinary input aliases the reserved host instance ID word" in diagnostic


def test_nearby_input_word_remains_valid(backend_binary):
    program = compile_instance(backend_binary, "instance-input-edge", count=INSTANCE_WORD - 1)
    assert program["layout"]["supported"] is True
    assert [entry["word_offset"] for entry in program["layout"]["inputs"]] == [INSTANCE_WORD - 1]
    cpu, reads, _, output_words = instance_cpu(program, 7,
                                               inputs={INSTANCE_WORD - 1: float_to_fp16_bits(0.625)})
    cpu.run(program["words"])
    assert cpu.halted and output_words() == {0: float_to_fp16_bits(0.625)}
    assert reads == [program["layout"]["bases"]["input"] + INSTANCE_WORD - 1]


@pytest.mark.parametrize("option", (None, "hardware"))
def test_default_legacy_and_hardware_abis_and_input_region_are_unchanged(backend_binary, option):
    program = compile_instance(backend_binary, "instance-input-edge", option=option, count=INSTANCE_WORD)
    layout = program["layout"]
    assert layout["supported"] is True and layout["version"] == 1
    assert layout["shader_abi_version"] == (2 if option == "hardware" else 1)
    assert layout["address_units"] == ("bytes" if option == "hardware" else "words")
    assert "host_instancing" not in layout and program["stats"]["host_instancing_version"] == 0
    assert [entry["word_offset"] for entry in layout["inputs"]] == [INSTANCE_WORD]


@pytest.mark.parametrize("option", ("host-instancing-custom", "host-instancing-unscheduled"))
@pytest.mark.parametrize("instance_id", (0, 7, 15))
def test_custom_bases_and_unscheduled_loads_use_same_raw_contract(backend_binary, option, instance_id):
    program = compile_instance(backend_binary, "instance-loop-phi", option=option)
    assert program["layout"]["supported"] is True
    cpu, reads, _, output_words = instance_cpu(program, instance_id)
    cpu.run(program["words"])
    assert cpu.halted and output_words() == expected_outputs("instance-loop-phi", instance_id)
    assert set(reads) == {program["layout"]["bases"]["input"] + INSTANCE_WORD}


@pytest.mark.parametrize("option,mode", (("host-texture-mixed", "texture-rgba"),
                                        ("host-texture-linear-mixed", "texture-rgba"),
                                        ("host-texture-mipmap", "texture-lod"),
                                        ("host-texture-linear-mipmap", "texture-lod")))
@pytest.mark.parametrize("wrap_s,wrap_t", (("clamp-to-edge", "repeat"), ("repeat", "mirrored-repeat"),
                                         ("mirrored-repeat", "clamp-to-edge")))
def test_opt_in_fragment_layout_preserves_texture_metadata_verbatim(backend_binary, option, mode,
                                                                  wrap_s, wrap_t):
    environment = {"MGC_TEST_WRAP_S": wrap_s, "MGC_TEST_WRAP_T": wrap_t, "MGC_TEST_SOURCE_LOD": "2"}
    previous = compile_instance(backend_binary, mode, option=option, environment=environment)
    instanced = compile_instance(backend_binary, mode, option=option,
                                 environment={**environment, "MGC_TEST_HOST_INSTANCING": "1"})
    old, new = previous["layout"], instanced["layout"]
    assert old["version"] == 2 and old["shader_abi_version"] == 1
    assert new["version"] == new["shader_abi_version"] == 3 and new["stage"] == "fragment"
    assert new["supported"] is True and new["host_instancing"]["uses_instance_id"] is False
    assert new["host_texture"] == old["host_texture"]
    assert {key: value for key, value in new.items()
            if key not in ("version", "shader_abi_version", "host_instancing")} == {
        key: value for key, value in old.items() if key not in ("version", "shader_abi_version")}
    assert instanced["assembly"].splitlines()[1:] == previous["assembly"].splitlines()[1:]
    assert instanced["words"] == previous["words"]


def test_unknown_new_option_cannot_silently_ignore_instance_usage(backend_binary):
    diagnostic = compile_instance(backend_binary, "instance-id", option="host-instancing-unknown",
                                  count=2, success=False)
    assert "host instancing version is unsupported for this target" in diagnostic
