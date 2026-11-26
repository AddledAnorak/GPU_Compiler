import copy
import itertools

import pytest

from test_linear_texture import layout_source, texture_payload
from tools.host_render import pack_float
from tools.host_texture import TEXTURE_BASE, TEXTURE_WRAPS, host_texture_contract, pack_texture
from tools.shader_layout import ShaderLayout


def mipmap_fixture(version, width, height, lod, wrap_s="repeat", wrap_t="clamp-to-edge"):
    source = layout_source(1)
    source["host_texture"] = host_texture_contract(version, wrap_s=wrap_s, wrap_t=wrap_t, lod=lod)
    texture = texture_payload(width, height, 2 if version == 10 else 1)
    texture["sampler"].update(wrap_s=wrap_s, wrap_t=wrap_t, mip_filter="nearest")
    texture["mipmaps"] = []
    level = 0
    while width > 1 or height > 1:
        level += 1
        width, height = max(1, width // 2), max(1, height // 2)
        contents = bytes((level * 41 + offset * 13) % 256 for offset in range(width * height * 4))
        texture["mipmaps"].append({"level": level, "width": width, "height": height,
                                   "data_hex": contents.hex()})
    return source, texture


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("width,height", ((1, 1), (1, 16), (16, 1), (3, 5), (5, 3), (16, 16)))
@pytest.mark.parametrize("lod", range(5))
def test_mipmap_packing_selects_checked_shader_lod_not_host_output(version, width, height, lod):
    source, texture = mipmap_fixture(version, width, height, lod)
    original = copy.deepcopy((source, texture))
    levels = [texture, *texture["mipmaps"]]
    selected = levels[min(lod, len(levels) - 1)]
    payload = pack_texture(ShaderLayout(source), [texture])
    assert (source, texture) == original
    assert payload[TEXTURE_BASE] == pack_float(selected["width"])
    assert payload[TEXTURE_BASE + 1] == pack_float(selected["height"])
    assert payload[TEXTURE_BASE + 2] == payload[TEXTURE_BASE + 3] == 0
    contents = bytes.fromhex(selected["data_hex"])
    assert set(payload) == set(range(TEXTURE_BASE, TEXTURE_BASE + 4 + len(contents)))
    assert [payload[TEXTURE_BASE + 4 + offset] for offset in range(len(contents))] == [
        pack_float(channel / 255) for channel in contents]


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("wrap_s,wrap_t", tuple(itertools.product(TEXTURE_WRAPS, repeat=2)))
def test_mipmap_axes_are_explicit_and_independent(version, wrap_s, wrap_t):
    source, texture = mipmap_fixture(version, 5, 3, 1, wrap_s, wrap_t)
    assert pack_texture(source, [texture])
    for axis in ("wrap_s", "wrap_t"):
        corrupt = copy.deepcopy(texture)
        corrupt["sampler"][axis] = "invalid"
        with pytest.raises(ValueError, match="sampler"):
            pack_texture(source, [corrupt])


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("lod", (None, True, -1, 5, 1.0, "1", [], {}))
def test_reflected_lod_is_exact_integer_constant(version, lod):
    with pytest.raises(ValueError, match="integer LOD"):
        host_texture_contract(version, wrap_s="repeat", wrap_t="repeat", lod=lod)


@pytest.mark.parametrize("version", (9, 10))
@pytest.mark.parametrize("mutation", ("no-mipmaps", "missing-level", "extra-level", "wrong-order",
                                      "wrong-size", "bool-size", "bool-level", "extra-field",
                                      "short-bytes", "invalid-bytes", "last-unselected", "base-bytes",
                                      "mip-linear", "mip-none", "resource-extra"))
def test_entire_mipmap_chain_is_validated_even_when_lod_zero(version, mutation):
    source, texture = mipmap_fixture(version, 16, 16, 0)
    levels = texture["mipmaps"]
    if mutation == "no-mipmaps":
        del texture["mipmaps"]
    elif mutation == "missing-level":
        levels.pop()
    elif mutation == "extra-level":
        levels.append(copy.deepcopy(levels[-1]))
    elif mutation == "wrong-order":
        levels.reverse()
    elif mutation == "wrong-size":
        levels[0]["width"] = 7
    elif mutation == "bool-size":
        levels[-1]["width"] = True
    elif mutation == "bool-level":
        levels[0]["level"] = True
    elif mutation == "extra-field":
        levels[0]["extra"] = 0
    elif mutation == "short-bytes":
        levels[0]["data_hex"] = levels[0]["data_hex"][:-2]
    elif mutation == "invalid-bytes":
        levels[0]["data_hex"] = "!" + levels[0]["data_hex"][1:]
    elif mutation == "last-unselected":
        levels[-1]["data_hex"] = ""
    elif mutation == "base-bytes":
        texture["data_hex"] = ""
    elif mutation.startswith("mip-"):
        texture["sampler"]["mip_filter"] = mutation.removeprefix("mip-")
    else:
        texture["extra"] = 0
    with pytest.raises(ValueError):
        pack_texture(source, [texture])


@pytest.mark.parametrize("version", range(1, 9))
def test_older_versions_never_silently_accept_mipmaps(version):
    source = layout_source(1)
    source["host_texture"] = (host_texture_contract(version, wrap_s="repeat", wrap_t="repeat")
                              if version in (7, 8) else host_texture_contract(version))
    _, texture = mipmap_fixture(9, 2, 2, 0)
    with pytest.raises(ValueError):
        pack_texture(source, [texture])
    with pytest.raises(ValueError):
        host_texture_contract(version, lod=0)


@pytest.mark.parametrize("mutation", ("hardware", "max-levels", "mip-filter", "lod-float", "missing-lod"))
def test_mipmap_layout_is_exact_and_never_hardware(mutation):
    source, _ = mipmap_fixture(9, 2, 2, 1)
    if mutation == "hardware":
        source.update(program_format="fp16-hardware-v1", shader_abi_version=2, address_units="bytes")
    elif mutation == "max-levels":
        source["host_texture"]["max_levels"] = 6
    elif mutation == "mip-filter":
        source["host_texture"]["mip_filter"] = "linear"
    elif mutation == "lod-float":
        source["host_texture"]["lod"] = 1.0
    else:
        del source["host_texture"]["lod"]
    with pytest.raises(ValueError):
        ShaderLayout(source)
