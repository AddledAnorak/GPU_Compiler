"""Current C/NIR compiler -> legacy ISA -> strict managed software execution."""

import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tools.mgcasm import assemble, disassemble
from tools.mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from tools.mgcdriver import Driver, ExecutionTimeoutError, Program, ProgramFormat, SimulationBackend
from tools.host_quad import LaneRegion, QuadSimulationBackend
from test_mixed_texture_compiler import fp16_sample


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = pathlib.Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
INPUT, UNIFORM, VERTEX, FRAGMENT, TEXTURE = 0x10000, 0x20000, 0x30000, 0x100000, 0x110000
WORDS = 65536


@pytest.fixture(scope="module")
def host_backend_binary(tmp_path_factory):
    if not (BUILD / "build.ninja").is_file():
        pytest.skip("configured Mesa build required for fresh private compiler objects")
    ninja = shutil.which("ninja") or str(ROOT / ".venv/bin/ninja")
    directory = tmp_path_factory.mktemp("host_features_private")
    target = "src/gallium/drivers/mgc/mgc_test_trig"
    old_test = target + ".p/mgc_test_trig.c.o"
    old_compiler = "src/gallium/drivers/mgc/libmgc.a.p/mgc_compiler.c.o"
    objects = []
    commands = []
    for original, name in [(old_test, "mgc_test_backend"), (old_compiler, "mgc_compiler")]:
        output = directory / (name + ".o")
        objects.append(str(output))
        command = subprocess.check_output([ninja, "-C", str(BUILD), "-t", "commands", original], text=True).splitlines()[-1]
        source = ROOT / "mesa/src/gallium/drivers/mgc" / (name + ".c")
        commands.append([str(output) + token[len(original):] if token.startswith(original) else
                         str(source) if token.endswith("/mgc_test_trig.c") or token.endswith("/mgc_compiler.c") else token
                         for token in shlex.split(command)])
    def compile_object(command):
        result = subprocess.run(command, cwd=BUILD, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(compile_object, commands))
    binary = directory / "mgc_test_backend"
    command = subprocess.check_output([ninja, "-C", str(BUILD), "-t", "commands", target], text=True).splitlines()[-1]
    link = [str(binary) if token == target else objects[0] if token == old_test else token
            for token in shlex.split(command)]
    link.insert(link.index(objects[0]) + 1, objects[1])
    result = subprocess.run(link, cwd=BUILD, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return binary


def compile_host(binary, mode="host-textures", count=1, version=13, option=None, success=True, **environment):
    command = [str(binary), mode, str(count)] + ([option] if option else [])
    result = subprocess.run(command, text=True, capture_output=True, env={**os.environ,
        "MGC_TEST_LAYOUT": "1", "MGC_TEST_HOST_FEATURES": "1", "MGC_TEST_HOST_TEXTURE": str(version),
        **{key: str(value) for key, value in environment.items()}})
    if not success:
        assert result.returncode == 1, result.stderr
        assert not result.stdout
        return result.stderr
    assert result.returncode == 0, result.stderr
    layout = json.loads(re.search(r"^layout=(.*)$", result.stderr, re.M).group(1))
    stats = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", result.stderr)}
    assert layout["supported"], layout["errors"]
    words = assemble(result.stdout)
    assert assemble("\n".join(disassemble(words))) == words
    return {"source": result.stdout, "layout": layout, "stats": stats,
            "program": Program.from_words(words, format=ProgramFormat.LEGACY)}


def words_bytes(values):
    return struct.pack("<" + "I" * len(values), *values)


def pack_chain(width=4, height=4, seed=0):
    payload = [0] * WORDS
    levels = []
    offset = 32
    while True:
        level = len(levels)
        texels = np.array([[[np.float16((seed + level * 13 + row * 5 + column * 3 + channel) / 256)
                            for channel in range(4)] for column in range(width)] for row in range(height)])
        payload[1 + level * 3:4 + level * 3] = [width, height, offset]
        data = [float_to_fp16_bits(float(value)) for value in texels.flat]
        payload[offset:offset + len(data)] = data
        levels.append(texels)
        offset += len(data)
        if width == height == 1:
            break
        width, height = max(width // 2, 1), max(height // 2, 1)
    payload[0] = len(levels)
    return payload[:16384], levels


def managed_run(compiled, values=(), texture=None, raw_inputs=None, max_steps=400000):
    bases = compiled["layout"]["bases"]
    output_base = bases[compiled["layout"]["stage"]]
    backend = SimulationBackend(memory_size=16 << 20)
    with Driver(backend) as driver:
        source = driver.allocate(WORDS * 4, address=bases["input"] * 4, alignment=4)
        output = driver.allocate(WORDS * 4, address=output_base * 4, alignment=4)
        payload = [float_to_fp16_bits(float(value)) for value in values]
        source.upload(words_bytes(payload))
        for offset, value in (raw_inputs or {}).items():
            source.upload(words_bytes([value]), offset=offset * 4)
        buffers = [source, output]
        if texture is not None:
            storage = driver.allocate(WORDS * 4, address=bases["texture"] * 4, alignment=4)
            storage.upload(words_bytes(texture))
            buffers.append(storage)
        driver.submit(compiled["program"], buffers=buffers, max_steps=max_steps).wait(20)
        result = list(struct.unpack("<" + "I" * WORDS, output.download()))
        return result


class ObservedQuadBackend(QuadSimulationBackend):
    def __init__(self, barriers, **options):
        super().__init__(**options)
        self.barriers = barriers

    def _resolve(self):
        if all(self._waiting):
            self.barriers.append(tuple(self._waiting))
        return super()._resolve()


def managed_quad(compiled, lane_values, texture=None, barriers=None, max_steps=600000):
    bases = compiled["layout"]["bases"]
    options = {"input_base": bases["input"], "output_base": bases["fragment"]}
    backend = QuadSimulationBackend(**options) if barriers is None else ObservedQuadBackend(barriers, **options)
    with Driver(backend) as driver:
        inputs, outputs, regions = [], [], []
        for lane, values in enumerate(lane_values):
            inputs.append(driver.allocate(WORDS * 4, address=bases["input"] * 4 + lane * (6 << 20), alignment=4))
            outputs.append(driver.allocate(WORDS * 4, address=bases["fragment"] * 4 + lane * (6 << 20), alignment=4))
            inputs[-1].upload(words_bytes([float_to_fp16_bits(float(value)) for value in values]))
            regions.append([LaneRegion(bases["input"] * 4, inputs[-1]), LaneRegion(bases["fragment"] * 4, outputs[-1], True)])
        buffers = inputs + outputs
        if texture is not None:
            storage = driver.allocate(WORDS * 4, address=bases["texture"] * 4, alignment=4)
            storage.upload(words_bytes(texture))
            buffers.append(storage)
            for mapping in regions:
                mapping.append(LaneRegion(bases["texture"] * 4, storage))
        backend.configure(regions)
        driver.submit(compiled["program"], buffers=buffers, max_steps=max_steps).wait(30)
        return [list(struct.unpack("<" + "I" * WORDS, output.download())) for output in outputs]


def pack_anisotropic_chain(width=32, height=32):
    payload, levels = pack_chain(width, height)
    for level, texels in enumerate(levels):
        rows, columns = np.indices(texels.shape[:2])
        texels[:, :, 0] = ((columns % 8) >= 4) * (1 - level / 8)
        texels[:, :, 1] = ((rows % 8) >= 4) * (1 - level / 8)
        texels[:, :, 2] = (columns + rows + level * 7) / 128
        texels[:, :, 3] = (level + 1) / 8
        offset = payload[3 + level * 3]
        payload[offset:offset + texels.size] = [float_to_fp16_bits(float(value)) for value in texels.flat]
    return payload, levels


def reference_linear(texels, uv, wraps):
    sizes = (texels.shape[1], texels.shape[0])
    lower, upper, weights = [], [], []
    for coordinate, size, wrap in zip(uv, sizes, wraps):
        if wrap == "repeat":
            coordinate %= 1
        elif wrap == "mirrored-repeat":
            coordinate %= 2
            coordinate = min(coordinate, 2 - coordinate)
        else:
            coordinate = min(max(coordinate, 0), 1)
        position = coordinate * size - 0.5
        first = math.floor(position)
        weights.append(position - first)
        if wrap == "repeat":
            lower.append(first % size)
            upper.append((first + 1) % size)
        else:
            lower.append(min(max(first, 0), size - 1))
            upper.append(min(max(first + 1, 0), size - 1))
    bottom = np.asarray(texels[lower[1], lower[0]], dtype=float) * (1 - weights[0])
    bottom += np.asarray(texels[lower[1], upper[0]], dtype=float) * weights[0]
    top = np.asarray(texels[upper[1], lower[0]], dtype=float) * (1 - weights[0])
    top += np.asarray(texels[upper[1], upper[0]], dtype=float) * weights[0]
    return bottom * (1 - weights[1]) + top * weights[1]


def reference_trilinear(levels, uv, lod, wraps):
    lod = min(max(lod, 0), len(levels) - 1)
    lower = math.floor(lod)
    upper = min(lower + 1, len(levels) - 1)
    return (reference_linear(levels[lower], uv, wraps) * (1 - (lod - lower)) +
            reference_linear(levels[upper], uv, wraps) * (lod - lower))


def reference_anisotropic(levels, uv, gradient_x, gradient_y, wraps=("clamp-to-edge", "clamp-to-edge")):
    dimensions = np.array([levels[0].shape[1], levels[0].shape[0]], dtype=float)
    gradients = [np.asarray(gradient_x, dtype=float), np.asarray(gradient_y, dtype=float)]
    squared = [float(np.dot(gradient * dimensions, gradient * dimensions)) for gradient in gradients]
    maximum, minimum = max(squared), min(squared)
    major_uv = gradients[0 if squared[0] >= squared[1] else 1]
    major_length = math.sqrt(max(maximum, 1))
    minor_length = math.sqrt(max(minimum, maximum / 16, 1))
    taps = min(max(math.ceil(major_length / minor_length), 1), 4)
    lod = min(max(math.log2(min(minor_length, 32)), 0), len(levels) - 1)
    result = np.zeros(4)
    for tap in range(taps):
        coordinate = np.asarray(uv) + ((tap + 0.5) / taps - 0.5) * major_uv
        result += reference_trilinear(levels, coordinate, lod, wraps)
    return result / taps, lod, taps


@pytest.mark.parametrize("version", [11, 12, 13])
def test_dynamic_texture_core(host_backend_binary, version):
    compiled = compile_host(host_backend_binary, version=version)
    texture, levels = pack_chain()
    result = managed_run(compiled, [0.5, 0.5, 0.5], texture)
    assert result[65499] == 0 and result[65535] == 2
    actual = [fp16_bits_to_float(value) for value in result[:4]]
    expected = levels[1][1, 1] if version == 11 else (levels[1].mean(axis=(0, 1)) if version == 12 else
               (levels[0][1:3, 1:3].mean(axis=(0, 1)) + levels[1].mean(axis=(0, 1))) / 2)
    np.testing.assert_allclose(actual, expected, atol=0.002)
    assert compiled["layout"]["version"] == compiled["layout"]["shader_abi_version"] == 4
    assert compiled["layout"]["host_texture"]["bindings"] == [{"texture": 0, "sampler": 0}]


def test_raw_vertex_builtins_and_exact_metadata(host_backend_binary):
    compiled = compile_host(host_backend_binary, "host-vertex", version=0)
    values = {65535: 255, 65520: 60001, 65521: 4294967200, 65522: 17}
    result = managed_run(compiled, raw_inputs=values)
    assert result[:4] == list(values.values())
    assert compiled["stats"]["host_builtin_mask"] == 15
    assert compiled["layout"]["host_features"]["builtins"] == [
        {"name": name, "word_offsets": [offset], "type": kind, "bit_size": 32, "storage": "raw32"}
        for name, offset, kind in [("gl_InstanceID", 65535, "int"), ("gl_VertexID", 65520, "int"),
                                   ("gl_BaseInstance", 65521, "uint"), ("gl_DrawID", 65522, "int")]]


@pytest.mark.parametrize("front", [0, 1])
def test_fragment_builtins_preserve_low_fp16_and_raw_bool(host_backend_binary, front):
    compiled = compile_host(host_backend_binary, "host-fragment", version=0)
    coordinates = [7.5, 3.5, 0.25, 2]
    values = {65524 + index: float_to_fp16_bits(value) for index, value in enumerate(coordinates)}
    values[65528] = front
    result = managed_run(compiled, raw_inputs=values)
    assert [fp16_bits_to_float(value) for value in result[:4]] == [value * (1 if front else -1) for value in coordinates]
    assert compiled["stats"]["host_builtin_mask"] == 48


def test_fragcoord_reflects_only_used_component(host_backend_binary):
    compiled = compile_host(host_backend_binary, "host-fragcoord-subset", count=2, version=0)
    assert compiled["layout"]["host_features"]["builtins"] == [
        {"name": "gl_FragCoord", "word_offsets": [65526], "type": "float", "bit_size": 32, "storage": "low_fp16"}]


@pytest.mark.parametrize("kill", [False, True])
def test_fine_quad_finite_differences_not_analytic_and_helpers(host_backend_binary, kill):
    compiled = compile_host(host_backend_binary, "host-derivative-kill" if kill else "host-derivatives", version=0)
    values = [-1, 3, 4, 8] if kill else [1, 3, 4, 8]
    results = managed_quad(compiled, [[value] for value in values])
    squared = [value * value for value in values]
    dx = [squared[1] - squared[0]] * 2 + [squared[3] - squared[2]] * 2
    dy = [squared[2] - squared[0], squared[3] - squared[1]] * 2
    for lane, result in enumerate(results):
        assert [fp16_bits_to_float(value) for value in result[:4]] == [squared[lane], dx[lane], dy[lane], abs(dx[lane]) + abs(dy[lane])]
        assert result[65535] == (1 if kill and lane == 0 else 2)
    assert compiled["layout"]["host_features"]["uses_derivatives"] is True
    assert compiled["layout"]["host_features"]["derivative_mailbox"]["operation_ids"] == {"dx": 1, "dy": 2}


def test_divergent_derivative_barrier_fails_closed(host_backend_binary):
    compiled = compile_host(host_backend_binary, "host-derivative-divergent", version=0)
    with pytest.raises(Exception, match="divergent|barriers|helper"):
        managed_quad(compiled, [[-1], [3], [4], [8]])


@pytest.mark.parametrize("mode", ["host-spill", "host-spill-cfg"])
def test_scalar_spills_roundtrip_in_managed_output(host_backend_binary, mode):
    compiled = compile_host(host_backend_binary, mode, count=96, version=0)
    result = managed_run(compiled, [0.125] * 96)
    assert fp16_bits_to_float(result[0]) == 12
    stats = compiled["stats"]
    assert 0 < stats["spill_words"] <= 4096
    assert stats["spill_stores"] > 0 and stats["spill_loads"] > 0 and stats["registers"] == 32
    assert compiled["layout"]["host_features"]["spill"]["words"] == stats["spill_words"]


@pytest.mark.parametrize("version", [11, 12, 13])
def test_implicit_lod_uses_quad_uv_derivatives(host_backend_binary, version):
    compiled = compile_host(host_backend_binary, version=version, MGC_TEST_TEXTURE_OP="tex")
    texture, levels = pack_chain(width=8, height=8)
    coordinates = [(0.25, 0.25), (0.5, 0.25), (0.25, 0.5), (0.5, 0.5)]
    results = managed_quad(compiled, [[horizontal, vertical] for horizontal, vertical in coordinates], texture)
    texels = levels[1]
    for result, (horizontal, vertical) in zip(results, coordinates):
        if version == 11:
            expected = texels[int(vertical * 4), int(horizontal * 4)]
        else:
            column, row = int(horizontal * 4 - 0.5), int(vertical * 4 - 0.5)
            expected = texels[row:row + 2, column:column + 2].mean(axis=(0, 1))
        np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.003)
        assert result[65535] == 2 and result[65499] == 0
    assert compiled["layout"]["host_texture"]["lod_mode"] == "implicit"
    assert compiled["layout"]["host_features"]["uses_derivatives"] is True


@pytest.mark.parametrize("version", [11, 12, 13])
def test_explicit_gradients_select_runtime_mip_without_mailbox(host_backend_binary, version):
    compiled = compile_host(host_backend_binary, version=version, MGC_TEST_TEXTURE_OP="txd")
    texture, levels = pack_chain(width=8, height=8)
    result = managed_run(compiled, [0.5, 0.5, 0, 0, 0.25, 0, 0, 0.25], texture)
    expected = levels[1][2, 2] if version == 11 else levels[1][1:3, 1:3].mean(axis=(0, 1))
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.003)
    assert result[65535] == 2 and result[65499] == 0
    assert compiled["layout"]["host_texture"]["lod_mode"] == "gradients"
    assert compiled["layout"]["host_features"]["uses_derivatives"] is False


@pytest.mark.parametrize("version", [11, 12, 13])
def test_constant_gradients_share_zero_with_raw_header_offsets(host_backend_binary, version):
    compiled = compile_host(host_backend_binary, version=version, MGC_TEST_TEXTURE_OP="txd",
                            MGC_TEST_CONSTANT_GRADIENTS=1, MGC_TEST_WRAP_S="repeat",
                            MGC_TEST_WRAP_T="mirrored-repeat")
    texture, levels = pack_chain(width=32, height=32)
    for uv in [(0.25, 0.75), (-0.25, 1.25)]:
        result = managed_run(compiled, uv, texture)
        expected = fp16_sample(levels[2], uv, ("repeat", "mirrored-repeat"), version != 11)
        np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.003)
        assert result[65499] == 0 and result[65535] == 2
    assert compiled["layout"]["host_texture"]["lod_mode"] == "gradients"
    assert compiled["layout"]["host_features"]["uses_derivatives"] is False


def test_multiple_sparse_texture_sampler_pairs(host_backend_binary):
    compiled = compile_host(host_backend_binary, count=4, version=13, MGC_TEST_BINDINGS="3:1,0:2,3:1,2:3")
    assert compiled["stats"]["host_texture_bindings"] == 3
    assert compiled["layout"]["host_texture"]["bindings"] == [
        {"texture": 0, "sampler": 2}, {"texture": 2, "sampler": 3}, {"texture": 3, "sampler": 1}]
    payload, chains = [], []
    for binding in range(4):
        chain, levels = pack_chain(seed=binding * 5)
        payload.extend(chain)
        chains.append(levels)
    result = managed_run(compiled, [0.5, 0.5, 0.25], payload)
    assert result[65535] == 2 and result[65499] == 0
    expected = np.zeros(4, dtype=np.float16)
    for sample, binding in enumerate([3, 0, 3, 2]):
        levels = chains[binding]
        lod = np.float16(0.25 + sample * 0.125)
        lower = fp16_sample(levels[0], (0.5, 0.5), ("clamp-to-edge", "clamp-to-edge"), True)
        upper = fp16_sample(levels[1], (0.5, 0.5), ("clamp-to-edge", "clamp-to-edge"), True)
        sampled = (lower + ((upper - lower).astype(np.float16) * lod).astype(np.float16)).astype(np.float16)
        expected = (expected + sampled).astype(np.float16)
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.002)


@pytest.mark.parametrize("wrap_s", ["clamp-to-edge", "repeat", "mirrored-repeat"])
@pytest.mark.parametrize("wrap_t", ["clamp-to-edge", "repeat", "mirrored-repeat"])
@pytest.mark.parametrize("version", [11, 12, 13])
def test_dynamic_wrap_axes_rectangular_npot_and_fp16_sampling(host_backend_binary, version, wrap_s, wrap_t):
    compiled = compile_host(host_backend_binary, version=version, MGC_TEST_WRAP_S=wrap_s, MGC_TEST_WRAP_T=wrap_t)
    texture, levels = pack_chain(width=7, height=3)
    for uv, lod in [((-0.25, 1.25), 0.25), ((1, 0), 0.75), ((0.375, 0.625), 1.5)]:
        result = managed_run(compiled, [*uv, lod], texture)
        clamped = min(lod, len(levels) - 1)
        lower = int(np.floor(clamped if version == 13 else clamped + 0.5))
        expected = fp16_sample(levels[lower], uv, (wrap_s, wrap_t), version != 11)
        if version == 13:
            next_sample = fp16_sample(levels[min(lower + 1, len(levels) - 1)], uv, (wrap_s, wrap_t), True)
            expected = (expected + ((next_sample - expected).astype(np.float16) * np.float16(clamped - lower)).astype(np.float16)).astype(np.float16)
        np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.002)
        assert result[65499] == 0 and result[65535] == 2


@pytest.mark.parametrize("size", [(1, 1), (32, 32), (32, 1), (1, 32)])
def test_full_chain_storage_maximum_dimensions_and_all_levels(host_backend_binary, size):
    compiled = compile_host(host_backend_binary)
    texture, levels = pack_chain(*size)
    assert texture[3] == 32 and texture[0] <= 6
    for lod in range(len(levels)):
        result = managed_run(compiled, [1, 1, lod], texture)
        np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], levels[lod][-1, -1], atol=0.002)
        assert result[65499] == 0


def test_32_dynamic_samples_execute_actual_legacy_instructions(host_backend_binary):
    compiled = compile_host(host_backend_binary, count=32, version=13, MGC_TEST_BINDINGS="0:3,1:2,2:1,3:0")
    payload = []
    for binding in range(4):
        chain, _ = pack_chain(width=1, height=1, seed=binding * 4)
        payload.extend(chain)
    result = managed_run(compiled, [0.5, 0.5, 0], payload)
    expected = np.array([(6 + channel) / 256 * 32 for channel in range(4)], dtype=np.float16)
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.004)
    assert compiled["stats"]["host_texture_samples"] == 32 and result[65535] == 2


@pytest.mark.parametrize("lod", ["nan", "inf", "-inf", "1e20"])
def test_nonfinite_constant_lod_is_compile_rejected(host_backend_binary, lod):
    assert "finite representable" in compile_host(host_backend_binary, "texture-lod", version=13,
                                                  MGC_TEST_SOURCE_LOD=lod, success=False)


@pytest.mark.parametrize("version", [11, 12, 13])
@pytest.mark.parametrize("lod", [-7, 0, 0.499, 0.501, 1.25, 8])
def test_finite_dynamic_lod_clamps_to_uploaded_chain(host_backend_binary, version, lod):
    compiled = compile_host(host_backend_binary, version=version)
    texture, levels = pack_chain(width=1, height=8)
    result = managed_run(compiled, [0.5, 0.5, lod], texture)
    clamped = min(max(float(np.float16(lod)), 0), len(levels) - 1)
    lower = int(np.floor(clamped if version == 13 else clamped + 0.5))
    def center(level):
        texels = levels[level]
        return texels[len(texels) // 2, 0] if version == 11 else texels.mean(axis=(0, 1))
    expected = center(lower)
    if version == 13:
        expected = expected + (center(min(lower + 1, len(levels) - 1)) - expected) * np.float16(clamped - lower)
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.002)


@pytest.mark.parametrize("mode,count", [("host-input-alias", 65504), ("host-input-alias", 65535),
                                        ("host-input-alias", 65498), ("host-output-alias", 57344),
                                        ("host-output-alias", 65535)])
def test_reserved_slots_reject_ordinary_aliases(host_backend_binary, mode, count):
    assert "reserved" in compile_host(host_backend_binary, mode, count, version=0, success=False)


@pytest.mark.parametrize("lod", [float("nan"), float("inf"), float("-inf")])
def test_runtime_nonfinite_lod_fails_closed(host_backend_binary, lod):
    compiled = compile_host(host_backend_binary)
    texture, _ = pack_chain()
    result = managed_run(compiled, [0.25, 0.25, lod], texture)
    assert result[65499] == 1 and result[65535] == 0


@pytest.mark.parametrize("word,value", [(0, 0), (0, 7), (1, 0), (2, 33), (3, 16384), (3, 0)])
def test_invalid_runtime_header_and_binding_bounds_fail_closed(host_backend_binary, word, value):
    compiled = compile_host(host_backend_binary)
    texture, _ = pack_chain()
    texture[word] = value
    result = managed_run(compiled, [0.25, 0.25, 0], texture)
    assert result[65499] == 1 and result[65535] == 0


@pytest.mark.parametrize("environment", [{"MGC_TEST_BINDINGS": "4:0"}, {"MGC_TEST_BINDINGS": "0:4"},
                                        {"MGC_TEST_TEXTURE_BASE": str(INPUT)},
                                        {"MGC_TEST_TEXTURE_BASE": str(0xFFFFFFF0)},
                                        {"MGC_TEST_HOST_FEATURES": "2"}, {"MGC_TEST_HOST_TEXTURE": "15"}])
def test_invalid_options_and_static_bindings_reject(host_backend_binary, environment):
    compile_host(host_backend_binary, success=False, **environment)


def test_static_sample_and_pair_resource_limits(host_backend_binary):
    assert "32 static samples" in compile_host(host_backend_binary, count=33, success=False)
    assert "four unique" in compile_host(host_backend_binary, count=5, MGC_TEST_BINDINGS="0:0,0:1,0:2,0:3,1:0", success=False)
    assert "4096 words" in compile_host(host_backend_binary, "host-spill", count=4200, version=0, success=False)
    assert "4096-value analysis" in compile_host(host_backend_binary, "host-spill-cfg", count=4100, version=0, success=False)


def test_unused_ordinary_reserved_load_still_rejects(host_backend_binary):
    assert "reserved" in compile_host(host_backend_binary, "instance-input-alias-unused", count=1, version=0, success=False)


@pytest.mark.parametrize("version", range(1, 11))
def test_abi4_rejects_old_texture_storage_contract(host_backend_binary, version):
    assert "ABI4 host textures require versions 11..14" in compile_host(host_backend_binary, version=version, success=False)


@pytest.mark.parametrize("version", range(1, 11))
def test_all_old_texture_versions_keep_abi_and_uniform_storage(host_backend_binary, version):
    mode = "texture-lod" if version >= 9 else "texture-rgba"
    compiled = compile_host(host_backend_binary, mode, count=0, version=version, MGC_TEST_HOST_FEATURES=0)
    layout = compiled["layout"]
    assert compiled["stats"]["abi"] == layout["shader_abi_version"] == 1
    assert layout["version"] == 2 and "texture" not in layout["bases"]
    assert "host_features" not in layout and layout["host_texture"]["storage_block"] == 15
    width, height = 2, 2
    values = [np.float16(index / 32) for index in range(width * height * 4)]
    with Driver(SimulationBackend(memory_size=8 << 20)) as driver:
        inputs = driver.allocate(WORDS * 4, alignment=4, address=INPUT * 4)
        outputs = driver.allocate(WORDS * 4, alignment=4, address=FRAGMENT * 4)
        uniforms = driver.allocate(WORDS * 4, alignment=4, address=UNIFORM * 4)
        inputs.upload(words_bytes([float_to_fp16_bits(0.5), float_to_fp16_bits(0.5)]))
        packed = [float_to_fp16_bits(width), float_to_fp16_bits(height), 0, 0] + [float_to_fp16_bits(float(value)) for value in values]
        uniforms.upload(words_bytes(packed), offset=15 * 4096 * 4)
        driver.submit(compiled["program"], buffers=[inputs, outputs, uniforms], max_steps=10000).wait(5)
        actual = [fp16_bits_to_float(value) for value in struct.unpack("<4I", outputs.download(size=16))]
    texels = np.array(values).reshape(height, width, 4)
    linear = version % 2 == 0
    expected = fp16_sample(texels, (0.5, 0.5), ("clamp-to-edge", "clamp-to-edge"), linear)
    np.testing.assert_allclose(actual, expected, atol=0.002)


def test_old_instancing_abi3_and_raw_local_id_remain_unchanged(host_backend_binary):
    compiled = compile_host(host_backend_binary, "instance-id", version=0,
                            MGC_TEST_HOST_FEATURES=0, MGC_TEST_HOST_INSTANCING=1)
    assert compiled["layout"]["version"] == compiled["stats"]["abi"] == 3
    assert compiled["layout"]["host_instancing"]["max_instances"] == 16
    assert "host_features" not in compiled["layout"] and "texture" not in compiled["layout"]["bases"]
    result = managed_run(compiled, raw_inputs={65535: 15})
    assert result[0] == 15


def test_host_option_rejects_hardware_without_changing_abi2(host_backend_binary):
    error = compile_host(host_backend_binary, "hardware-add", version=0, option="hardware", success=False)
    assert "host features version 1 requires" in error
    command = [str(host_backend_binary), "hardware-add", "1", "hardware"]
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ,
                            "MGC_TEST_HOST_FEATURES": "0", "MGC_TEST_HOST_TEXTURE": "0"})
    assert result.returncode == 0 and "ABI=2" in result.stdout
    assert "abi=2" in result.stderr


@pytest.mark.parametrize("mode", ["host-vertex", "host-fragment", "host-derivatives", "host-textures"])
@pytest.mark.parametrize("option", [None, "hardware"])
def test_default_and_hardware_reject_new_features(host_backend_binary, mode, option):
    compile_host(host_backend_binary, mode, version=13 if mode == "host-textures" else 0,
                 option=option, MGC_TEST_HOST_FEATURES=0, success=False)


LEGACY_TEXTURE_FINGERPRINTS = [
    (11, "txl", "8345a325e52c519798519ded7494c2c7741f121646bb3e86aecad373292e6066", "49995c7bdadbe0164c55545b1ce7b7794994eb5ba3f44f59f589f9c5516925bc"),
    (11, "txd", "b77dfdca7fcf9b769995d018f135426b80afd6b1e038798dd79cf102ac9cf101", "f5978519190dc083dccf6e90073b7fc632c8d4ee5832e0c9b9ff5a5d53e02d03"),
    (11, "tex", "9e3f3aa4a03117eb4df53749ee8498675f3c47b9490ff4378c84056e0980a2f4", "5b5fd9bd07cfa704ca3de38b18a9e32512bebdc8a39e784bedba4f2eb438c088"),
    (12, "txl", "948bf16193dd04db6452a4066133dcde7ff37aecaffd06a5152435460482143a", "b1500c8c1115942d91f2212175e72e397fb0c49e87ca90a9f376018ae8bf96b4"),
    (12, "txd", "a1a25814e464addc9592afa5d31b405b0d2c563ab715c9c378603767bf9d05ef", "995e6cf1877ca0f57e614a18444958f260fc764451e69bbcd2fdb0b98332ee34"),
    (12, "tex", "b391c4d357267afa2b251d4002215506950aaac4afd9616e47d56d1f9bae49f7", "419e26c6aa168118179e24e69ce67c7d214f14bc137b88fa55a69ee71f871ff9"),
    (13, "txl", "f5ba9ba138fee194d997c6b143b0fa71c1bea49ed6d1d1df681a28cc276686ab", "cf5f389209add413a459828c3c866869b5776e121074723f076dc2108614da69"),
    (13, "txd", "b4311dedb65a40c50ac94c7c8052d0b9ca54c55a043718407fcc51ad1e61f61a", "5b219f2d71855d8684c8aee3400aa181727e7e5682a18b67c327a26b5bd54769"),
    (13, "tex", "c75ce7d14b3ea1e292931082ccc1976f0dbb2e564e00056a6cf06f767b6bd4dd", "dea79d86cf5ba04a55d48370dfb426b42183e06d93189763c014ebcd0883127d"),
]


@pytest.mark.parametrize("version,operation,source_hash,layout_hash", LEGACY_TEXTURE_FINGERPRINTS)
def test_v11_v13_assembly_and_layout_fingerprints_unchanged(host_backend_binary, version, operation, source_hash, layout_hash):
    compiled = compile_host(host_backend_binary, version=version, MGC_TEST_TEXTURE_OP=operation)
    canonical = json.dumps(compiled["layout"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(compiled["source"].encode()).hexdigest() == source_hash
    assert hashlib.sha256(canonical.encode()).hexdigest() == layout_hash
    assert "max_anisotropy" not in compiled["layout"]["host_texture"]


@pytest.mark.parametrize("operation", ["txl", "txd", "tex"])
def test_v14_exact_metadata_and_existing_feature_schema(host_backend_binary, operation):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP=operation)
    assert compiled["layout"]["host_texture"] == {
        "version": 14, "bindings": [{"texture": 0, "sampler": 0}], "max_bindings": 4,
        "max_samples": 32, "max_size": 32, "max_levels": 6, "per_binding_stride": 16384,
        "header_words": 32, "lod_mode": {"txl": "explicit", "txd": "gradients", "tex": "implicit"}[operation],
        "wrap_s": "clamp-to-edge", "wrap_t": "clamp-to-edge", "filter": "linear", "mip_filter": "linear",
        "max_anisotropy": 4,
    }
    assert type(compiled["layout"]["host_texture"]["max_anisotropy"]) is int
    assert compiled["layout"]["host_features"]["uses_derivatives"] is (operation == "tex")
    assert compiled["layout"]["host_features"]["fault_word_offset"] == 65499


@pytest.mark.parametrize("lod", [-3, 0, 0.5, 1.25, 8])
def test_v14_txl_retains_v13_isotropic_single_tap(host_backend_binary, lod):
    compiled = compile_host(host_backend_binary, version=14)
    legacy = compile_host(host_backend_binary, version=13)
    texture, levels = pack_anisotropic_chain()
    uv = (0.5546875, 0.4453125)
    result = managed_run(compiled, [*uv, lod], texture)
    assert result == managed_run(legacy, [*uv, lod], texture)
    assert compiled["program"].words == legacy["program"].words
    expected = reference_trilinear(levels, uv, lod, ("clamp-to-edge", "clamp-to-edge"))
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.003)
    assert result[65499] == 0 and result[65535] == 2


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
@pytest.mark.parametrize("major,minor,taps", [(0, 0, 1), (0.5, 0.25, 1), (1, 1, 1), (1.5, 1, 2),
                                             (2.5, 1, 3), (3.5, 1, 4), (4, 1, 4), (8, 1, 4), (3, 1.5, 2)])
def test_v14_explicit_gradient_stripes_offcentre_and_tap_counts(host_backend_binary, axis, major, minor, taps):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="txd")
    texture, levels = pack_anisotropic_chain()
    uv = (0.5546875, 0.4453125)
    gradient_x = (major / 32, 0) if axis == "horizontal" else (minor / 32, 0)
    gradient_y = (0, minor / 32) if axis == "horizontal" else (0, major / 32)
    expected, _, actual_taps = reference_anisotropic(levels, uv, gradient_x, gradient_y)
    assert actual_taps == taps
    result = managed_run(compiled, [*uv, 0, 0, *gradient_x, *gradient_y], texture)
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.009)
    assert result[65499] == 0 and result[65535] == 2
    assert compiled["stats"]["host_texture_samples"] == 1


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
@pytest.mark.parametrize("kill", [False, True])
def test_v14_implicit_quad_stripes_and_discarded_helpers(host_backend_binary, axis, kill):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="tex",
                            MGC_TEST_TEXTURE_KILL=1 if kill else "", MGC_TEST_WRAP_S="repeat")
    texture, levels = pack_anisotropic_chain(width=32, height=16)
    origin = np.array([-0.0625 if kill else 0.5546875, 0.4453125])
    gradient_x = np.array([4 / 32 if axis == "horizontal" else 1 / 32, 0])
    gradient_y = np.array([0, 1 / 16 if axis == "horizontal" else 4 / 16])
    coordinates = [origin, origin + gradient_x, origin + gradient_y, origin + gradient_x + gradient_y]
    barriers = []
    results = managed_quad(compiled, coordinates, texture, barriers)
    for coordinate, result in zip(coordinates, results):
        expected, _, _ = reference_anisotropic(levels, coordinate, gradient_x, gradient_y,
                                               ("repeat", "clamp-to-edge"))
        np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.009)
        assert result[65499] == 0
        assert result[65535] == (1 if kill and coordinate[0] < 0 else 2)
    assert len(barriers) == 4
    assert all(len(set(barrier)) == 1 for barrier in barriers)
    assert sorted(barrier[0][1] for barrier in barriers) == [1, 1, 2, 2]


def test_v14_quad_lane_tap_counts_differ_without_extra_barriers(host_backend_binary):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="tex")
    texture, levels = pack_anisotropic_chain()
    origin = np.array([0.5546875, 0.4453125])
    coordinates = [origin, origin + [1.5 / 32, 0], origin + [0, 1 / 32], origin + [3.5 / 32, 1 / 32]]
    horizontal = [coordinates[1] - coordinates[0]] * 2 + [coordinates[3] - coordinates[2]] * 2
    vertical = [coordinates[2] - coordinates[0], coordinates[3] - coordinates[1]] * 2
    barriers = []
    results = managed_quad(compiled, coordinates, texture, barriers)
    tap_counts = []
    for coordinate, gradient_x, gradient_y, result in zip(coordinates, horizontal, vertical, results):
        expected, _, taps = reference_anisotropic(levels, coordinate, gradient_x, gradient_y)
        tap_counts.append(taps)
        np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.009)
        assert result[65499] == 0 and result[65535] == 2
    assert len(set(tap_counts)) > 1
    assert len(barriers) == 4 and all(len(set(barrier)) == 1 for barrier in barriers)


@pytest.mark.parametrize("wrap_s,wrap_t", [("repeat", "mirrored-repeat"), ("mirrored-repeat", "repeat"),
                                           ("clamp-to-edge", "clamp-to-edge")])
def test_v14_each_tap_wraps_independently_on_npot_chain(host_backend_binary, wrap_s, wrap_t):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="txd",
                            MGC_TEST_WRAP_S=wrap_s, MGC_TEST_WRAP_T=wrap_t)
    texture, levels = pack_anisotropic_chain(width=7, height=3)
    uv, gradient_x, gradient_y = (-0.125, 1.25), (0.5, 0), (0, 0.25)
    expected, _, taps = reference_anisotropic(levels, uv, gradient_x, gradient_y, (wrap_s, wrap_t))
    assert taps == 4
    result = managed_run(compiled, [*uv, 0, 0, *gradient_x, *gradient_y], texture)
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.009)
    assert result[65499] == 0 and result[65535] == 2


def test_v14_constant_gradients_share_typed_zero_and_compute_anisotropic_lod(host_backend_binary):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="txd",
                            MGC_TEST_CONSTANT_GRADIENTS=1, MGC_TEST_WRAP_S="repeat",
                            MGC_TEST_WRAP_T="mirrored-repeat")
    texture, levels = pack_anisotropic_chain()
    uv = (-0.25, 1.25)
    expected, lod, taps = reference_anisotropic(levels, uv, (0.125, 0), (0, 0.125),
                                               ("repeat", "mirrored-repeat"))
    assert lod == 2 and taps == 1
    result = managed_run(compiled, uv, texture)
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.003)
    assert result[65499] == 0 and result[65535] == 2


def test_v14_inactive_positions_cannot_fault_on_unused_overflow(host_backend_binary):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="txd")
    texture, levels = pack_anisotropic_chain(width=1, height=1)
    result = managed_run(compiled, [65504, 65504, 0, 0, 100, 0, 0, 100], texture)
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], levels[0][0, 0], atol=0.003)
    assert result[65499] == 0 and result[65535] == 2


@pytest.mark.parametrize("gradient", [float("nan"), float("inf"), float("-inf"), 8, 65504])
def test_v14_nonfinite_or_overflowing_gradient_faults(host_backend_binary, gradient):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="txd")
    texture, _ = pack_anisotropic_chain()
    result = managed_run(compiled, [0.5, 0.5, 0, 0, gradient, 0, 0, 1 / 32], texture)
    assert result[65499] == 1 and result[65535] == 0


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_v14_nonfinite_uv_faults(host_backend_binary, coordinate):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="txd")
    texture, _ = pack_anisotropic_chain()
    result = managed_run(compiled, [coordinate, 0.5, 0, 0, 0.125, 0, 0, 1 / 32], texture)
    assert result[65499] == 1 and result[65535] == 0


@pytest.mark.parametrize("lod", [float("nan"), float("inf"), float("-inf")])
def test_v14_txl_nonfinite_runtime_lod_faults(host_backend_binary, lod):
    compiled = compile_host(host_backend_binary, version=14)
    texture, _ = pack_chain()
    result = managed_run(compiled, [0.5, 0.5, lod], texture)
    assert result[65499] == 1 and result[65535] == 0


@pytest.mark.parametrize("lod", ["nan", "inf", "-inf", "1e20"])
def test_v14_txl_nonfinite_constant_lod_rejects(host_backend_binary, lod):
    assert "finite representable" in compile_host(host_backend_binary, "texture-lod", version=14,
                                                  MGC_TEST_SOURCE_LOD=lod, success=False)


@pytest.mark.parametrize("word,value", [(0, 0), (0, 7), (1, 0), (2, 33), (3, 16384), (3, 16380)])
def test_v14_runtime_header_and_texel_plane_bounds_fault(host_backend_binary, word, value):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="txd")
    texture, _ = pack_chain()
    texture[word] = value
    result = managed_run(compiled, [0.75, 0.75, 0, 0, 0.5, 0, 0, 0.25], texture)
    assert result[65499] == 1 and result[65535] == 0


def test_v14_sparse_multibinding_numerical_accumulation(host_backend_binary):
    compiled = compile_host(host_backend_binary, count=4, version=14, MGC_TEST_TEXTURE_OP="txd",
                            MGC_TEST_BINDINGS="3:1,0:2,3:1,2:3")
    payload, chains = [], []
    for binding in range(4):
        chain, levels = pack_chain(width=8, height=8, seed=binding * 11)
        payload.extend(chain)
        chains.append(levels)
    uv, gradient_x, gradient_y = (0.375, 0.625), (0.5, 0), (0, 0.125)
    expected = sum(reference_anisotropic(chains[binding], uv, gradient_x, gradient_y)[0]
                   for binding in [3, 0, 3, 2])
    result = managed_run(compiled, [*uv, 0, 0, *gradient_x, *gradient_y], payload)
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.003)
    assert compiled["layout"]["host_texture"]["bindings"] == [
        {"texture": 0, "sampler": 2}, {"texture": 2, "sampler": 3}, {"texture": 3, "sampler": 1}]
    assert result[65499] == 0 and result[65535] == 2
    assert 0 < compiled["stats"]["spill_words"] <= 4096
    assert compiled["stats"]["spill_stores"] > 0 and compiled["stats"]["spill_loads"] > 0


def test_v14_32_source_samples_keep_limit_and_execute_existing_instructions(host_backend_binary):
    compiled = compile_host(host_backend_binary, count=32, version=14, MGC_TEST_TEXTURE_OP="txd",
                            MGC_TEST_BINDINGS="0:3,1:2,2:1,3:0")
    payload = []
    for binding in range(4):
        chain, _ = pack_chain(width=1, height=1, seed=binding * 4)
        payload.extend(chain)
    result = managed_run(compiled, [0.5, 0.5, 0, 0, 0, 0, 0, 0], payload)
    expected = [(6 + channel) / 256 * 32 for channel in range(4)]
    np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.004)
    assert compiled["stats"]["host_texture_samples"] == 32
    assert result[65499] == 0 and result[65535] == 2


def test_v14_static_resources_and_budget_reject(host_backend_binary):
    assert "32 static samples" in compile_host(host_backend_binary, version=14, count=33,
                                               MGC_TEST_TEXTURE_OP="txd", success=False)
    assert "four unique" in compile_host(host_backend_binary, version=14, count=5,
                                         MGC_TEST_BINDINGS="0:0,0:1,0:2,0:3,1:0", success=False)
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP="txd")
    texture, _ = pack_anisotropic_chain()
    with pytest.raises(ExecutionTimeoutError, match="max_steps|budget"):
        managed_run(compiled, [0.5, 0.5, 0, 0, 0.125, 0, 0, 1 / 32], texture, max_steps=50)


@pytest.mark.parametrize("operation", ["txd", "tex"])
def test_v14_sampling_after_cfg_preserves_scalar_spills_and_quad_join(host_backend_binary, operation):
    compiled = compile_host(host_backend_binary, version=14, MGC_TEST_TEXTURE_OP=operation,
                            MGC_TEST_TEXTURE_AFTER_IF=1)
    texture, levels = pack_anisotropic_chain(width=8, height=8)
    uv, gradient_x, gradient_y = (0.375, 0.625), (0.25, 0), (0, 0.125)
    if operation == "txd":
        results = [managed_run(compiled, [*uv, 0, 0, *gradient_x, *gradient_y], texture)]
        coordinates = [uv]
    else:
        origin = np.array(uv)
        coordinates = [origin, origin + gradient_x, origin + gradient_y, origin + gradient_x + gradient_y]
        results = managed_quad(compiled, coordinates, texture)
    for coordinate, result in zip(coordinates, results):
        expected, _, _ = reference_anisotropic(levels, coordinate, gradient_x, gradient_y)
        np.testing.assert_allclose([fp16_bits_to_float(value) for value in result[:4]], expected, atol=0.009)
        assert result[65499] == 0 and result[65535] == 2
    assert compiled["stats"]["spill_words"] > 0


@pytest.mark.parametrize("option,features", [(None, 0), ("hardware", 0), ("hardware", 1)])
def test_v14_default_and_hardware_reject_without_matching_software_selection(host_backend_binary, option, features):
    compile_host(host_backend_binary, version=14, option=option, MGC_TEST_HOST_FEATURES=features, success=False)


def test_v14_texture_helpers_include_new_linear_version_only(host_backend_binary):
    result = subprocess.run([str(host_backend_binary), "texture-helpers"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
