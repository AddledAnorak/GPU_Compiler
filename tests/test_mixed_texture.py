import copy
import itertools

import pytest

from test_linear_texture import layout_source, texture_payload
from tools.host_render import pack_float
from tools.host_texture import (TEXTURE_BASE, TEXTURE_WRAPS, host_texture_contract,
                               host_texture_metadata_contract, pack_texture)
from tools.shader_layout import ShaderLayout


def mixed_layout(version, wrap_s, wrap_t):
    source = layout_source(1)
    source["host_texture"] = host_texture_contract(version, wrap_s=wrap_s, wrap_t=wrap_t)
    return source


def mixed_texture(width, height, version, wrap_s, wrap_t):
    texture = texture_payload(width, height, 2 if version == 8 else 1)
    texture["sampler"].update(wrap_s=wrap_s, wrap_t=wrap_t)
    return texture


@pytest.mark.parametrize("version", (7, 8))
@pytest.mark.parametrize("wrap_s,wrap_t", tuple(itertools.product(TEXTURE_WRAPS, repeat=2)))
@pytest.mark.parametrize("width,height", ((1, 1), (1, 16), (16, 1), (3, 5), (5, 3), (16, 16)))
def test_mixed_contract_and_packing(version, wrap_s, wrap_t, width, height):
    source = mixed_layout(version, wrap_s, wrap_t)
    texture = mixed_texture(width, height, version, wrap_s, wrap_t)
    original = copy.deepcopy((source, texture))
    payload = pack_texture(ShaderLayout(source), [texture])
    assert (source, texture) == original
    assert set(payload) == set(range(TEXTURE_BASE, TEXTURE_BASE + 4 + width * height * 4))
    assert payload[TEXTURE_BASE] == pack_float(width)
    assert payload[TEXTURE_BASE + 1] == pack_float(height)
    assert payload[TEXTURE_BASE + 2] == payload[TEXTURE_BASE + 3] == 0
    assert [payload[TEXTURE_BASE + 4 + offset] for offset in range(width * height * 4)] == [
        pack_float(channel / 255) for channel in bytes.fromhex(texture["data_hex"])]
    contract = host_texture_metadata_contract(source["host_texture"])
    assert "wrap" not in contract
    contract["wrap_s"] = "invalid"
    assert source["host_texture"]["wrap_s"] == wrap_s


@pytest.mark.parametrize("version", (7, 8))
@pytest.mark.parametrize("axis", ("wrap_s", "wrap_t"))
@pytest.mark.parametrize("value", (None, True, 1, "repeat ", "mirror-repeat", [], {}))
def test_mixed_axes_fail_closed(version, axis, value):
    source = mixed_layout(version, "repeat", "clamp-to-edge")
    source["host_texture"][axis] = value
    with pytest.raises(ValueError):
        ShaderLayout(source)
    with pytest.raises(ValueError):
        pack_texture(source, [mixed_texture(2, 2, version, "repeat", "clamp-to-edge")])


@pytest.mark.parametrize("version", (7, 8))
@pytest.mark.parametrize("mutation", ("missing-s", "missing-t", "extra-wrap", "extra-field",
                                      "filter", "unknown-version", "hardware", "abi", "stage"))
def test_mixed_metadata_is_exact_and_legacy_only(version, mutation):
    source = mixed_layout(version, "repeat", "mirrored-repeat")
    metadata = source["host_texture"]
    if mutation.startswith("missing-"):
        del metadata["wrap_" + mutation[-1]]
    elif mutation == "extra-wrap":
        metadata["wrap"] = "repeat"
    elif mutation == "extra-field":
        metadata["extra"] = 0
    elif mutation == "filter":
        metadata["filter"] = "nearest" if version == 8 else "linear"
    elif mutation == "unknown-version":
        metadata["version"] = 11
    elif mutation == "hardware":
        source.update(program_format="fp16-hardware-v1", shader_abi_version=2, address_units="bytes")
    elif mutation == "abi":
        source["shader_abi_version"] = 2
    else:
        source["stage"] = "vertex"
    with pytest.raises(ValueError):
        ShaderLayout(source)
    with pytest.raises(ValueError):
        pack_texture(source, [mixed_texture(2, 2, version, "repeat", "mirrored-repeat")])


@pytest.mark.parametrize("version", range(1, 7))
def test_existing_contracts_remain_paired(version):
    contract = host_texture_contract(version)
    assert "wrap" in contract and "wrap_s" not in contract and "wrap_t" not in contract
    with pytest.raises(ValueError):
        host_texture_contract(version, wrap_s="repeat", wrap_t="repeat")


@pytest.mark.parametrize("version", (7, 8))
@pytest.mark.parametrize("axis", ("wrap_s", "wrap_t"))
def test_mixed_sampler_must_match_shader_axes(version, axis):
    source = mixed_layout(version, "repeat", "mirrored-repeat")
    texture = mixed_texture(3, 5, version, "repeat", "mirrored-repeat")
    texture["sampler"][axis] = "clamp-to-edge"
    with pytest.raises(ValueError, match="sampler"):
        pack_texture(source, [texture])
