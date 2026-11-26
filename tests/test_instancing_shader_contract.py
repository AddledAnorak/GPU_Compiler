import copy
import struct

import pytest

from tools.host_instancing import (HOST_INSTANCE_ID_WORD, shader_instancing_contract,
                                   validate_shader_instancing)
from tools.host_render import ShaderExecutor
from tools.shader_layout import ShaderLayout


def instance_layout(stage="vertex", uses_instance_id=True):
    return {"version": 3, "supported": True, "errors": [], "program_format": "legacy-mgcsim-v1",
            "shader_abi_version": 3, "address_units": "words", "stage": stage,
            "host_instancing": shader_instancing_contract(uses_instance_id),
            "bases": {"input": 0x10000, "uniform": 0x20000, "vertex": 0x30000, "fragment": 0x100000},
            "inputs": [], "uniforms": [],
            "outputs": [{"word_offset": 0, "location": 32, "component": 0, "type": "int",
                         "bit_size": 32, "interpolation": "flat"}]}


def instance_copy_program():
    return ("; mgc legacy-simulator backend, ABI=3, stage=MESA_SHADER_VERTEX\n"
            f"movi_i32 {0x10000 + HOST_INSTANCE_ID_WORD}, r0\n"
            "load r0, r1\nwait_load\nmovi_i32 196608, r0\nstore r1, r0\nstop_core\n")


def test_managed_driver_delivers_exact_raw_local_ids_and_resets_each_invocation():
    executor = ShaderExecutor(instance_copy_program(), ShaderLayout(instance_layout()))
    try:
        for instance_id in (15, 0, 9, 1, 0, 15):
            assert executor.invoke({}, instance_id=instance_id) == {0: instance_id}
            assert struct.unpack("<I", executor.buffers["input"].download(
                offset=HOST_INSTANCE_ID_WORD * 4, size=4))[0] == instance_id
        assert executor.invocations == 6
    finally:
        executor.close()


@pytest.mark.parametrize("value", (None, True, -1, 16, 1.0, "1"))
def test_missing_invalid_ids_fail_before_submission(value):
    executor = ShaderExecutor(instance_copy_program(), ShaderLayout(instance_layout()))
    try:
        with pytest.raises(ValueError, match="instance ID"):
            executor.invoke({}, instance_id=value)
        assert executor.invocations == 0
    finally:
        executor.close()


@pytest.mark.parametrize("field,value", (("version", 2), ("version", True), ("max_instances", 17),
                                       ("instance_id_word", 65534), ("instance_id_word", True),
                                       ("instance_id", "base-plus-local"), ("uses_instance_id", 1)))
def test_shader_instancing_metadata_is_exact(field, value):
    source = instance_layout()
    source["host_instancing"][field] = value
    with pytest.raises(ValueError):
        ShaderLayout(source)


@pytest.mark.parametrize("mutation", ("missing", "extra", "old-layout", "old-abi", "hardware", "fragment-id"))
def test_shader_instancing_version_target_and_stage_are_fail_closed(mutation):
    source = instance_layout()
    if mutation == "missing":
        del source["host_instancing"]["instance_id_word"]
    elif mutation == "extra":
        source["host_instancing"]["extra"] = 0
    elif mutation == "old-layout":
        source["version"] = 1
    elif mutation == "old-abi":
        source["shader_abi_version"] = 1
    elif mutation == "hardware":
        source.update(program_format="fp16-hardware-v1", shader_abi_version=2, address_units="bytes")
    else:
        source["stage"] = "fragment"
    with pytest.raises(ValueError):
        ShaderLayout(source)


def test_ordinary_scalar_inputs_cannot_alias_the_reserved_instance_word():
    source = instance_layout()
    source["inputs"] = [dict(source["outputs"][0], word_offset=HOST_INSTANCE_ID_WORD)]
    with pytest.raises(ValueError, match="reserved host instance-ID"):
        ShaderLayout(source)


def test_contract_copies_and_default_metadata_stay_independent():
    source = instance_layout(uses_instance_id=False)
    original = copy.deepcopy(source)
    assert validate_shader_instancing(source) == shader_instancing_contract()
    assert source == original
    other = shader_instancing_contract()
    other["max_instances"] = 1
    assert shader_instancing_contract()["max_instances"] == 16
    assert validate_shader_instancing({"version": 1, "shader_abi_version": 1}) is None


def test_abi1_does_not_silently_accept_an_instance_argument():
    source = instance_layout(uses_instance_id=False)
    source.update(version=1, shader_abi_version=1)
    del source["host_instancing"]
    program = instance_copy_program().replace("ABI=3", "ABI=1")
    executor = ShaderExecutor(program, ShaderLayout(source))
    try:
        with pytest.raises(ValueError, match="explicit host vertex instancing"):
            executor.invoke({}, instance_id=0)
    finally:
        executor.close()
