import copy
import struct

import pytest

from tools.host_instancing import (HOST_INSTANCING_CONTRACT, shader_instancing_contract,
                                   validate_instancing)
from tools.host_render import fetch_vertices, pack_float
from tools.shader_layout import ShaderLayout


def instanced_fetch_fixture(instances=5, base=2, divisor=2, indexed=False, kind="float"):
    scalar_type = {"float": "FLOAT", "int": "SINT", "uint": "UINT"}[kind]
    layout = {"version": 3, "supported": True, "errors": [], "program_format": "legacy-mgcsim-v1",
              "shader_abi_version": 3, "address_units": "words", "stage": "vertex",
              "host_instancing": shader_instancing_contract(),
              "bases": {"input": 0x10000, "uniform": 0x20000, "vertex": 0x30000, "fragment": 0x100000},
              "inputs": [{"word_offset": 0, "location": 0, "component": 0, "type": "float",
                          "bit_size": 32, "interpolation": "smooth"},
                         {"word_offset": 4, "location": 1, "component": 0, "type": kind,
                          "bit_size": 32, "interpolation": "smooth"}],
              "outputs": [], "uniforms": []}
    vertex_data = struct.pack("<6f", 0, 1, 2, 3, 4, 5)
    instance_data = (struct.pack("<16f", *range(16)) if kind == "float" else
                     struct.pack("<16I", *(0x80000000 + value for value in range(16))))
    artifact = {"schema": "mgc-host-draw-v3", "program_format": "legacy-mgcsim-v1",
                "shader_abi_version": 3,
                "host_instancing": dict(HOST_INSTANCING_CONTRACT),
                "draw": {"mode": "triangles", "count": 3, "start": 1, "instance_count": instances,
                         "start_instance": base, "index_size": 1 if indexed else 0, "index_bias": 0,
                         "primitive_restart": False},
                "vertex_buffers": [{"stride": 4, "data_hex": vertex_data.hex()},
                                   {"stride": 4, "data_hex": instance_data.hex()}],
                "vertex_elements": [{"src_offset": 0, "vertex_buffer_index": 0, "instance_divisor": 0,
                                     "format": "R32_FLOAT"},
                                    {"src_offset": 0, "vertex_buffer_index": 1,
                                     "instance_divisor": divisor, "format": "R32_" + scalar_type}],
                "indices": bytes((5, 3, 1, 4)).hex() if indexed else None}
    return artifact, ShaderLayout(layout)


@pytest.mark.parametrize("instances,divisor,base", ((1, 1, 0), (5, 1, 2), (5, 2, 2),
                                                  (5, 3, 1), (16, 1048576, 2)))
@pytest.mark.parametrize("indexed", (False, True))
@pytest.mark.parametrize("kind", ("float", "int", "uint"))
def test_divisors_use_local_instance_before_base_and_never_vertex_index(instances, divisor, base, indexed, kind):
    artifact, layout = instanced_fetch_fixture(instances, base, divisor, indexed, kind)
    original = copy.deepcopy(artifact)
    result = fetch_vertices(artifact, layout)
    order = (3, 1, 4) if indexed else (1, 2, 3)
    expected = []
    for local_instance in range(instances):
        instance_index = base + local_instance // divisor
        value = pack_float(instance_index) if kind == "float" else 0x80000000 + instance_index
        expected.extend({0: pack_float(vertex), 4: value} for vertex in order)
    assert result == expected
    assert artifact == original


@pytest.mark.parametrize("mutation", ("no-contract", "extra-contract", "wrong-version", "bool-version",
                                      "wrong-indexing", "wrong-cap", "hardware", "old-schema", "unknown-schema"))
def test_instancing_contract_is_versioned_exact_and_legacy_only(mutation):
    artifact, layout = instanced_fetch_fixture()
    if mutation == "no-contract":
        del artifact["host_instancing"]
    elif mutation == "extra-contract":
        artifact["host_instancing"]["extra"] = 1
    elif mutation == "wrong-version":
        artifact["host_instancing"]["version"] = 2
    elif mutation == "bool-version":
        artifact["host_instancing"]["version"] = True
    elif mutation == "wrong-indexing":
        artifact["host_instancing"]["indexing"] = "floor-global-instance-over-divisor"
    elif mutation == "wrong-cap":
        artifact["host_instancing"]["max_instances"] = 17
    elif mutation == "hardware":
        artifact["program_format"] = "fp16-hardware-v1"
    else:
        artifact["schema"] = "mgc-host-draw-v1" if mutation == "old-schema" else "mgc-host-draw-v4"
    with pytest.raises(ValueError):
        fetch_vertices(artifact, layout)


@pytest.mark.parametrize("field", ("instance_count", "start_instance", "divisor"))
@pytest.mark.parametrize("value", (True, -1, 1.0, "2", None, 1048577))
def test_instancing_rejects_noninteger_or_out_of_bounds_fields(field, value):
    artifact, layout = instanced_fetch_fixture()
    if field == "divisor":
        artifact["vertex_elements"][1]["instance_divisor"] = value
    else:
        artifact["draw"][field] = value
    with pytest.raises(ValueError):
        fetch_vertices(artifact, layout)


@pytest.mark.parametrize("instances", (0, 17))
def test_instance_count_is_bounded(instances):
    artifact, layout = instanced_fetch_fixture(instances=instances)
    with pytest.raises(ValueError):
        fetch_vertices(artifact, layout)


def test_expanded_vertex_budget_is_checked_before_fetch():
    artifact, layout = instanced_fetch_fixture(instances=16)
    artifact["draw"]["count"] = 258
    with pytest.raises(ValueError, match="expanded vertex budget"):
        fetch_vertices(artifact, layout)


@pytest.mark.parametrize("indexed", (False, True))
@pytest.mark.parametrize("source", (0, 1))
def test_every_instance_attribute_fetch_is_resource_bounded(indexed, source):
    artifact, layout = instanced_fetch_fixture(indexed=indexed)
    artifact["vertex_buffers"][source]["data_hex"] = struct.pack("<f", 0).hex()
    with pytest.raises(ValueError, match="resource span"):
        fetch_vertices(artifact, layout)


def test_zero_stride_instance_buffer_repeats_one_resource_word():
    artifact, layout = instanced_fetch_fixture()
    artifact["vertex_buffers"][1] = {"stride": 0, "data_hex": struct.pack("<f", 0.5).hex()}
    assert {entry[4] for entry in fetch_vertices(artifact, layout)} == {pack_float(0.5)}


def test_old_capture_does_not_silently_enable_instancing():
    artifact, layout = instanced_fetch_fixture(instances=1, base=0, divisor=0)
    artifact["schema"] = "mgc-host-draw-v1"
    artifact["shader_abi_version"] = 1
    del artifact["host_instancing"]
    source = dict(layout.source, version=1, shader_abi_version=1)
    del source["host_instancing"]
    layout = ShaderLayout(source)
    assert validate_instancing(artifact) is False
    assert len(fetch_vertices(artifact, layout)) == 3
    artifact["draw"]["instance_count"] = 2
    with pytest.raises(ValueError, match="single-instance"):
        fetch_vertices(artifact, layout)
