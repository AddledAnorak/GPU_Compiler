"""Bounded CPU rasterization of compiler shaders and actual Gallium artifacts."""

from __future__ import annotations

import hashlib
import math
import re
import struct
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from tools.host_blend import blend_rgba8, validate_blend_mode
from tools.host_features import BUILTINS, DRAW_FEATURES, VertexPayload
from tools.host_instancing import (HOST_INSTANCE_ID_WORD, HOST_INSTANCING_CONTRACT,
                                   validate_instancing)
from tools.mgcasm import assemble
from tools.mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from tools.mgcdriver import Driver, HardwareSimulationBackend, Program, ProgramFormat, SimulationBackend
from tools.mgchw import assemble_hardware, disassemble_hardware
from tools.shader_layout import ShaderLayout, bounded_integer, link_varyings


def pack_float(value):
    if not math.isfinite(value):
        raise ValueError("host shader payload must be finite")
    bits = float_to_fp16_bits(value)
    if bits & 0x7C00 == 0x7C00:
        raise ValueError("host shader payload overflows finite FP16")
    return bits


class FragmentStatus(str, Enum):
    COMMITTED = "committed"
    DISCARDED = "discarded"


@dataclass(frozen=True)
class FragmentResult:
    status: FragmentStatus
    outputs: dict


class ShaderExecutor:
    def __init__(self, assembly, layout, uniforms=None, *, max_steps=100000, texture=None,
                 execution_deadline=None):
        self.layout = layout
        self.max_steps = bounded_integer(max_steps, "invocation instruction budget", 1000000, 1)
        if execution_deadline is not None and (type(execution_deadline) not in (int, float) or
                                               not math.isfinite(execution_deadline)):
            raise ValueError("shader execution deadline must be finite")
        self.execution_deadline = execution_deadline
        hardware = layout.target == ProgramFormat.HARDWARE
        profile = layout.target.value if hardware else "legacy-simulator"
        header = f"; mgc {profile} backend, ABI={layout.source['shader_abi_version']},"
        if not isinstance(assembly, str) or not assembly.startswith(header) or len(assembly) > 4 << 20:
            raise ValueError("assembly provenance does not match compiler layout")
        words = assemble_hardware(assembly) if hardware else assemble(assembly)
        if hardware and assemble_hardware("\n".join(disassemble_hardware(words))) != words:
            raise ValueError("hardware program does not round-trip")
        self.program = Program.from_words(words, format=layout.target)
        byte_bases = {name: base * (1 if hardware else 4) for name, base in layout.source["bases"].items()}
        memory_size = max(byte_bases.values()) + 65536 * 4
        if memory_size > 32 << 20:
            raise ValueError("host executor ABI aperture exceeds bounded memory budget")
        self.quad_lanes = []
        derivatives = bool(layout.host_features and layout.host_features["uses_derivatives"])
        if derivatives:
            from tools.host_quad import QuadSimulationBackend

            backend = QuadSimulationBackend(input_base=layout.base("input"), output_base=layout.base("fragment"))
        else:
            backend = HardwareSimulationBackend(memory_size=memory_size) if hardware else SimulationBackend(memory_size=memory_size)
        self.driver = Driver(backend)
        self.buffers = {}
        try:
            self.buffers = {name: self.driver.allocate(65536 * 4, alignment=4, address=base)
                            for name, base in byte_bases.items()}
            payload = dict(uniforms or {})
            if layout.host_texture and layout.host_features:
                from tools.host_texture_features import pack_texture_features

                self._upload("texture", pack_texture_features(layout, texture))
            elif layout.host_texture:
                from tools.host_texture import pack_texture

                if any(bounded_integer(offset, "constant scalar offset") >= 15 * 4096 for offset in payload):
                    raise ValueError("caller constants overlap reserved host texture storage")
                payload.update(pack_texture(layout, texture))
            elif texture is not None and (not layout.host_features or texture != []):
                raise ValueError("texture payload requires an explicit host texture shader contract")
            self._upload("uniform", payload)
            if derivatives:
                from tools.host_quad import LaneRegion

                for lane in range(4):
                    regions = dict(self.buffers)
                    if lane:
                        for domain in ("input", "fragment"):
                            regions[domain] = self.driver.allocate(65536 * 4, alignment=4,
                                                                  address=byte_bases[domain] + lane * (6 << 20))
                    self.quad_lanes.append(regions)
                backend.configure([[LaneRegion(byte_bases[domain], buffer, domain == "fragment")
                                    for domain, buffer in regions.items()] for regions in self.quad_lanes])
        except Exception:
            self.driver.close()
            raise
        self.steps = 0
        self.invocations = 0

    def _upload(self, domain, payload):
        for offset, value in payload.items():
            bounded_integer(offset, "payload scalar offset")
            bounded_integer(value, "payload word", (1 << 32) - 1)
            self.buffers[domain].upload(struct.pack("<I", value), offset=offset * 4)

    def _remaining_timeout(self, maximum):
        if self.execution_deadline is None:
            return maximum
        remaining = self.execution_deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("draw execution deadline exceeded")
        return min(maximum, remaining)

    def _builtin_payload(self, supplied, instance_id):
        if not self.layout.host_features:
            if supplied:
                raise ValueError("builtin values require explicit host features")
            return {}
        supplied = dict(supplied or {})
        if instance_id is not None:
            if "gl_InstanceID" in supplied and supplied["gl_InstanceID"] != instance_id:
                raise ValueError("conflicting local instance IDs")
            supplied["gl_InstanceID"] = instance_id
        if any(name not in BUILTINS for name in supplied):
            raise ValueError("unsupported invocation builtin value")
        result = {}
        for declaration in self.layout.host_features["builtins"]:
            name = declaration["name"]
            if name not in supplied:
                raise ValueError(f"invocation requires reflected builtin {name}")
            values = supplied[name] if name == "gl_FragCoord" else [supplied[name]]
            full_offsets = BUILTINS[name][0]
            if type(values) not in (tuple, list) or len(values) != len(full_offsets):
                raise ValueError("builtin input component count mismatch")
            for offset in declaration["word_offsets"]:
                value = values[full_offsets.index(offset)]
                if declaration["type"] == "float":
                    result[offset] = pack_float(value)
                elif declaration["type"] == "bool":
                    if type(value) is not bool:
                        raise ValueError("front-facing builtin must be boolean")
                    result[offset] = int(value)
                else:
                    result[offset] = bounded_integer(value, "raw builtin value", (1 << 32) - 1)
        return result

    def _read_result(self, buffers):
        stage = self.layout.source["stage"]
        if self.layout.host_features:
            fault = struct.unpack("<I", buffers[stage].download(offset=65499 * 4, size=4))[0]
            if fault:
                raise ValueError(f"compiled host shader reported execution fault {fault}")
        if self.layout.fragment_status:
            contract = self.layout.fragment_status
            signal = struct.unpack("<I", buffers[stage].download(offset=contract["word_offset"] * 4, size=4))[0]
            if signal == contract["discarded"]:
                return FragmentResult(FragmentStatus.DISCARDED, {})
            if signal != contract["committed"]:
                raise ValueError("shader completed without a valid explicit fragment completion signal")
        outputs = {entry["word_offset"]: struct.unpack("<I", buffers[stage].download(
                    offset=entry["word_offset"] * 4, size=4))[0] for entry in self.layout.entries("outputs")}
        return FragmentResult(FragmentStatus.COMMITTED, outputs)

    def _execute(self, inputs, *, instance_id=None, builtins=None):
        if self.quad_lanes:
            raise ValueError("derivative shader requires an explicit four-lane invocation")
        stage = self.layout.source["stage"]
        if self.layout.host_instancing and stage == "vertex":
            if instance_id is not None or not self.layout.host_features:
                bounded_integer(instance_id, "local instance ID", 255 if self.layout.host_features else 15)
        elif instance_id is not None:
            raise ValueError("instance ID requires an explicit host vertex instancing contract")
        self.buffers["input"].upload(bytes(65536 * 4))
        self.buffers[stage].upload(bytes(65536 * 4))
        required = {entry["word_offset"] for entry in self.layout.entries("inputs")}
        if set(inputs) != required:
            raise ValueError("invocation inputs must match reflected scalar accesses exactly")
        self._upload("input", inputs)
        self._upload("input", self._builtin_payload(builtins, instance_id))
        if self.layout.host_instancing and stage == "vertex" and instance_id is not None:
            self._upload("input", {HOST_INSTANCE_ID_WORD: instance_id})
        fence = self.driver.submit(self.program, buffers=tuple(self.buffers.values()),
                                   max_steps=self.max_steps, execution_timeout=self._remaining_timeout(5))
        execution = fence.wait(self._remaining_timeout(10))
        self._remaining_timeout(10)
        self.steps += execution.steps
        self.invocations += 1
        return self._read_result(self.buffers)

    def invoke_quad(self, inputs, builtins):
        if not self.quad_lanes or type(inputs) is not list or type(builtins) is not list or len(inputs) != 4 or len(builtins) != 4:
            raise ValueError("host derivative invocation requires exactly four input and builtin lanes")
        required = {entry["word_offset"] for entry in self.layout.entries("inputs")}
        for regions, payload, supplied in zip(self.quad_lanes, inputs, builtins):
            if set(payload) != required:
                raise ValueError("quad inputs must match reflected scalar accesses exactly")
            regions["input"].upload(bytes(65536 * 4))
            regions["fragment"].upload(bytes(65536 * 4))
            combined = {**payload, **self._builtin_payload(supplied, None)}
            for offset, value in combined.items():
                bounded_integer(offset, "quad input scalar offset")
                bounded_integer(value, "quad input word", (1 << 32) - 1)
                regions["input"].upload(struct.pack("<I", value), offset=offset * 4)
        buffers = tuple({id(buffer): buffer for regions in self.quad_lanes for buffer in regions.values()}.values())
        fence = self.driver.submit(self.program, buffers=buffers, max_steps=self.max_steps * 4,
                                   execution_timeout=self._remaining_timeout(5))
        execution = fence.wait(self._remaining_timeout(10))
        self._remaining_timeout(10)
        self.steps += execution.steps
        self.invocations += 4
        return [self._read_result(regions) for regions in self.quad_lanes]

    def invoke(self, inputs, *, instance_id=None, builtins=None):
        result = self._execute(inputs, instance_id=instance_id, builtins=builtins)
        if result.status == FragmentStatus.DISCARDED:
            raise ValueError("discarded invocation requires the explicit invoke_fragment result")
        return result.outputs

    def invoke_fragment(self, inputs, *, builtins=None):
        if self.layout.source["stage"] != "fragment":
            raise ValueError("fragment invocation requires a fragment shader layout")
        if self.layout.source.get("uses_discard") and not self.layout.fragment_status:
            raise ValueError("discard requires an explicit shader output-commit contract")
        return self._execute(inputs, builtins=builtins)

    def close(self):
        self.driver.close()


def decode_hex(value):
    if not isinstance(value, str) or len(value) > 32 << 20 or len(value) % 2 or not re.fullmatch("[0-9a-fA-F]*", value):
        raise ValueError("invalid or oversized captured byte payload")
    return bytes.fromhex(value)


def pack_constants(layout, buffers):
    if not isinstance(buffers, list) or len(buffers) > 16:
        raise ValueError("captured constants must be a bounded slot list")
    if sum(len(value) for value in buffers if isinstance(value, str)) > 32 << 20:
        raise ValueError("captured constants exceed aggregate resource budget")
    decoded = [decode_hex(value) if value is not None else None for value in buffers]
    if layout.host_texture and len(decoded) > 15 and decoded[15] is not None:
        raise ValueError("ordinary constant slot15 is reserved for host texture storage")
    payload = {}
    for entry in layout.entries("uniforms"):
        offset = entry["word_offset"]
        block, local = divmod(offset, 4096)
        if layout.host_texture and not layout.host_features and block == 15:
            raise ValueError("ordinary constants overlap reserved host texture storage")
        if block >= len(decoded) or decoded[block] is None or local * 4 + 4 > len(decoded[block]):
            raise ValueError("captured constant span does not cover reflected access")
        contents = decoded[block][local * 4:local * 4 + 4]
        payload[offset] = pack_float(struct.unpack("<f", contents)[0]) if entry["type"] == "float" else struct.unpack("<I", contents)[0]
    return payload


def fetch_vertices(artifact, layout):
    draw = artifact["draw"]
    if not isinstance(draw, dict):
        raise ValueError("captured draw description must be an object")
    instancing = validate_instancing(artifact)
    if instancing and (layout.target != ProgramFormat.LEGACY or not layout.host_instancing):
        raise ValueError("host instancing requires legacy vertex execution")
    limits = DRAW_FEATURES if layout.host_features else HOST_INSTANCING_CONTRACT
    instances = bounded_integer(draw.get("instance_count"), "instance count",
                                limits["max_instances"] if instancing else 1048576, 1)
    start_instance = bounded_integer(draw.get("start_instance", 0), "start instance", 1048576)
    restart = draw.get("primitive_restart", False)
    if not isinstance(restart, bool):
        raise ValueError("primitive restart flag must be explicitly boolean")
    if (draw.get("mode") != "triangles" or restart or
            (not instancing and (instances != 1 or start_instance != 0))):
        raise ValueError("host renderer supports direct single-instance triangle lists only")
    count = bounded_integer(draw.get("count"), "draw count", 65536 if layout.host_features else 4096, 3)
    start = bounded_integer(draw.get("start"), "draw start", 1048576)
    if count % 3:
        raise ValueError("triangle-list count must be divisible by three")
    maximum_vertices = limits["max_expanded_vertices"] if layout.host_features else limits["max_vertices"]
    if instances * count > maximum_vertices:
        raise ValueError("draw exceeds aggregate expanded vertex budget")
    index_size = bounded_integer(draw.get("index_size", 0), "index size", 4)
    bias = draw.get("index_bias", 0)
    if isinstance(bias, bool) or not isinstance(bias, int) or not -(1 << 31) <= bias < (1 << 31):
        raise ValueError("invalid index bias")
    if index_size:
        if index_size not in (1, 2, 4):
            raise ValueError("unsupported captured index width")
        indices = decode_hex(artifact.get("indices"))
        if (start + count) * index_size > len(indices):
            raise ValueError("captured index span is too short")
        order = [int.from_bytes(indices[offset * index_size:(offset + 1) * index_size], "little") + bias
                 for offset in range(start, start + count)]
    else:
        if bias:
            raise ValueError("non-indexed draw cannot have index bias")
        order = list(range(start, start + count))
    if any(index < 0 or index > 1048576 for index in order):
        raise ValueError("captured vertex index exceeds budget")
    buffers = artifact.get("vertex_buffers")
    elements = artifact.get("vertex_elements")
    if not isinstance(buffers, list) or len(buffers) > 16 or not isinstance(elements, list) or len(elements) > 32:
        raise ValueError("invalid captured vertex declarations")
    if any(buffer is not None and not isinstance(buffer, dict) for buffer in buffers) or any(not isinstance(element, dict) for element in elements):
        raise ValueError("captured vertex declarations must be objects")
    if sum(len(buffer.get("data_hex", "")) for buffer in buffers if isinstance(buffer, dict)) > 32 << 20:
        raise ValueError("captured vertices exceed aggregate resource budget")
    decoded = [decode_hex(buffer["data_hex"]) if buffer is not None else None for buffer in buffers]
    for element in elements:
        divisor = bounded_integer(element.get("instance_divisor", 0), "instance divisor", 1048576)
        if divisor and not instancing:
            raise ValueError("instanced vertex elements require version3 host instancing")
    result = []
    for local_instance, vertex_index in ((instance, index) for instance in range(instances) for index in order):
        payload = {}
        for entry in layout.entries("inputs"):
            slot, component = divmod(entry["word_offset"], 4)
            if slot >= len(elements):
                raise ValueError("vertex declaration does not cover reflected input")
            element = elements[slot]
            divisor = element.get("instance_divisor", 0)
            buffer_index = bounded_integer(element.get("vertex_buffer_index"), "vertex buffer index", 15)
            if buffer_index >= len(buffers) or decoded[buffer_index] is None:
                raise ValueError("vertex element references missing captured storage")
            format_name = element.get("format")
            if not isinstance(format_name, str):
                raise ValueError("captured attribute format must be explicit text")
            format_name = format_name.removeprefix("PIPE_FORMAT_")
            match = re.fullmatch(r"(R32(?:G32)?(?:B32)?(?:A32)?)_(FLOAT|SINT|UINT)", format_name)
            if not match:
                raise ValueError("only native scalar/vector 32-bit captured attributes are supported")
            channels = match.group(1).count("32")
            kind = {"FLOAT": "float", "SINT": "int", "UINT": "uint"}[match.group(2)]
            if entry["type"] != kind:
                raise ValueError("captured attribute type disagrees with compiler reflection")
            stride = bounded_integer(buffers[buffer_index].get("stride"), "vertex stride", 65536)
            source_offset = bounded_integer(element.get("src_offset"), "attribute byte offset", 1 << 24)
            attribute_index = start_instance + local_instance // divisor if divisor else vertex_index
            address = source_offset + attribute_index * stride
            if address + channels * 4 > len(decoded[buffer_index]):
                raise ValueError("captured attribute fetch exceeds resource span")
            if component >= channels:
                value = 1 if component == 3 else 0
                payload[entry["word_offset"]] = pack_float(value) if kind == "float" else value
            else:
                contents = decoded[buffer_index][address + component * 4:address + component * 4 + 4]
                payload[entry["word_offset"]] = pack_float(struct.unpack("<f", contents)[0]) if kind == "float" else struct.unpack("<I", contents)[0]
        if layout.host_features:
            payload = VertexPayload(payload, {"gl_InstanceID": local_instance, "gl_VertexID": vertex_index,
                                              "gl_BaseInstance": start_instance,
                                              "gl_DrawID": bounded_integer(artifact.get("drawid", 0), "draw ID", (1 << 31) - 1)})
        result.append(payload)
    return result


def edge(first, second, point):
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (point[0] - first[0])


def clip_triangle(vertices):
    polygon = vertices
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        clipped = []
        for index, current in enumerate(polygon):
            previous = polygon[index - 1]
            distance = current["position"][3] + sign * current["position"][axis]
            old_distance = previous["position"][3] + sign * previous["position"][axis]
            if (distance >= 0) != (old_distance >= 0):
                fraction = old_distance / (old_distance - distance)
                clipped.append({"position": previous["position"] + fraction * (current["position"] - previous["position"]),
                                "varyings": {offset: previous["varyings"][offset] + fraction * (current["varyings"][offset] - previous["varyings"][offset])
                                             for offset in current["varyings"]}})
            if distance >= 0:
                clipped.append(current)
        polygon = clipped
        if not polygon:
            break
    return [(polygon[0], polygon[index], polygon[index + 1]) for index in range(1, len(polygon) - 1)]


def raster_samples(triangle, width, height):
    positions = [vertex["position"] for vertex in triangle]
    if any(position[3] <= 0 or not np.all(np.isfinite(position)) for position in positions):
        raise ValueError("clipped vertices require finite positive homogeneous w")
    screen = [np.array([(position[0] / position[3] + 1) * width / 2,
                        (position[1] / position[3] + 1) * height / 2]) for position in positions]
    area = edge(screen[0], screen[1], screen[2])
    if area == 0:
        return
    if area < 0:
        triangle = (triangle[0], triangle[2], triangle[1])
        positions[1], positions[2] = positions[2], positions[1]
        screen[1], screen[2] = screen[2], screen[1]
        area = -area
    minimum = np.maximum(np.floor(np.min(screen, axis=0) - 0.5).astype(int), 0)
    maximum = np.minimum(np.ceil(np.max(screen, axis=0) - 0.5).astype(int), [width - 1, height - 1])
    sides = [(screen[1], screen[2]), (screen[2], screen[0]), (screen[0], screen[1])]
    inclusive = [(second[1] - first[1] < 0 or (second[1] == first[1] and second[0] < first[0]))
                 for first, second in sides]
    inverse_w = np.array([1 / position[3] for position in positions])
    for row in range(minimum[1], maximum[1] + 1):
        for column in range(minimum[0], maximum[0] + 1):
            edges = [edge(first, second, [column + 0.5, row + 0.5]) for first, second in sides]
            if any(value < 0 or (value == 0 and not include) for value, include in zip(edges, inclusive)):
                continue
            barycentric = np.array(edges) / area
            perspective = barycentric * inverse_w
            perspective /= perspective.sum()
            depth = sum(weight * position[2] / position[3] for weight, position in zip(barycentric, positions)) * 0.5 + 0.5
            yield column, row, triangle, barycentric, perspective, depth


def _fragment_payload(layout, ordered, barycentric, perspective, provoking):
    inputs = {}
    for entry in layout.entries("inputs"):
        offset = entry["word_offset"]
        if entry["type"] != "float":
            if not layout.host_features or entry["interpolation"] != "flat":
                raise ValueError("integer varying raster transport is not implemented outside explicit ABI4 flat interpolation")
            inputs[offset] = bounded_integer(provoking["varyings"][offset], "flat integer varying", (1 << 32) - 1)
            continue
        if entry["interpolation"] == "flat":
            value = provoking["varyings"][offset]
        else:
            weights = perspective if entry["interpolation"] == "smooth" else barycentric
            value = sum(weight * point["varyings"][offset] for weight, point in zip(weights, ordered))
        inputs[offset] = pack_float(value)
    return inputs


def _fragment_builtins(column, row, ordered, barycentric, depth, front_facing):
    reciprocal_w = sum(weight / point["position"][3] for weight, point in zip(barycentric, ordered))
    return {"gl_FragCoord": [float(column) + 0.5, float(row) + 0.5, float(depth), float(reciprocal_w)],
            "gl_FrontFacing": bool(front_facing)}


def _render_samples(fragment, layout, triangle, provoking, width, height, max_fragments, deadline):
    if not layout.host_features:
        for sample in raster_samples(triangle, width, height):
            yield (*sample, None, None)
        return
    original_screen = [((point["position"][0] / point["position"][3] + 1) * width / 2,
                        (point["position"][1] / point["position"][3] + 1) * height / 2) for point in triangle]
    front = edge(*original_screen) > 0
    samples = list(raster_samples(triangle, width, height))
    if not fragment.quad_lanes:
        for sample in samples:
            column, row, ordered, barycentric, perspective, depth = sample
            yield (*sample, None, _fragment_builtins(column, row, ordered, barycentric, depth, front))
        return
    groups = {}
    for sample in samples:
        groups.setdefault((sample[0] // 2, sample[1] // 2), []).append(sample)
    for (quad_column, quad_row), active in groups.items():
        if deadline is not None and time.monotonic() >= deadline:
            raise ValueError("draw execution deadline exceeded")
        if fragment.invocations + 4 > max_fragments:
            raise ValueError("draw exceeds aggregate fragment and helper budget")
        ordered = active[0][2]
        positions = [point["position"] for point in ordered]
        screen = [((position[0] / position[3] + 1) * width / 2,
                   (position[1] / position[3] + 1) * height / 2) for position in positions]
        area = edge(*screen)
        payloads, builtins = [], []
        for lane in range(4):
            column, row = quad_column * 2 + lane % 2, quad_row * 2 + lane // 2
            center = (column + 0.5, row + 0.5)
            barycentric = np.array([edge(screen[1], screen[2], center), edge(screen[2], screen[0], center),
                                    edge(screen[0], screen[1], center)]) / area
            perspective = barycentric / np.array([position[3] for position in positions])
            perspective /= perspective.sum()
            depth = sum(weight * position[2] / position[3] for weight, position in zip(barycentric, positions)) * 0.5 + 0.5
            payloads.append(_fragment_payload(layout, ordered, barycentric, perspective, provoking))
            builtins.append(_fragment_builtins(column, row, ordered, barycentric, depth, front))
        results = fragment.invoke_quad(payloads, builtins)
        for sample in active:
            lane = sample[0] % 2 + sample[1] % 2 * 2
            yield (*sample, results[lane], builtins[lane])


def render_artifact(artifact, *, width=16, height=16, depth_test=False, blend="disabled",
                    max_steps=10000000, max_fragments=65536, initial_image=None,
                    execution_timeout=None, initial_depth=None, depth_output=None,
                    render_state=None, initial_depth_stencil=None, depth_stencil_output=None):
    bounded_integer(width, "framebuffer width", 128, 1)
    bounded_integer(height, "framebuffer height", 128, 1)
    bounded_integer(max_steps, "draw instruction budget", 100000000, 1)
    bounded_integer(max_fragments, "draw fragment budget", 65536, 1)
    if not isinstance(depth_test, bool):
        raise ValueError("depth mode must be explicitly boolean")
    validate_blend_mode(blend)
    if initial_image is not None and (not isinstance(initial_image, np.ndarray) or
            initial_image.dtype != np.uint8 or initial_image.shape != (height, width, 4)):
        raise ValueError("initial framebuffer must match bounded RGBA8 geometry")
    for depth_array in (initial_depth, depth_output):
        if depth_array is not None and (not depth_test or not isinstance(depth_array, np.ndarray) or
                depth_array.dtype != np.uint16 or depth_array.shape != (height, width)):
            raise ValueError("native depth requires enabled depth and matching Z16 geometry")
    if depth_output is not None and not depth_output.flags.writeable:
        raise ValueError("native depth output must be writable staging memory")
    quantized_depth = initial_depth is not None or depth_output is not None
    if execution_timeout is not None and (type(execution_timeout) not in (int, float) or
            not math.isfinite(execution_timeout) or not 0 < execution_timeout <= 60):
        raise ValueError("draw execution timeout must be finite and between zero and sixty seconds")
    deadline = None if execution_timeout is None else time.monotonic() + execution_timeout
    if not isinstance(artifact, dict):
        raise ValueError("unsupported Gallium capture schema")
    instancing = validate_instancing(artifact)
    features = artifact.get("schema") == "mgc-host-draw-v4"
    pipeline_state = None
    depth_plane = None
    if features:
        from tools.host_pipeline_state import pack_depth_stencil, validate_state

        pipeline_state = validate_state(artifact.get("render_state") if render_state is None else render_state)
        if render_state is not None and artifact.get("render_state") != pipeline_state.to_dict():
            raise ValueError("execution state differs from captured render state")
        if depth_test or blend != "disabled" or initial_depth is not None or depth_output is not None:
            raise ValueError("ABI4 state cannot be combined with legacy rendering overrides")
        depth_size = {"none": 0, "z16": 2, "z24s8": 4}[pipeline_state.depth_stencil_format]
        expected_size = width * height * depth_size
        if initial_depth_stencil is not None and (type(initial_depth_stencil) not in (bytes, bytearray) or
                                                 len(initial_depth_stencil) != expected_size or not depth_size):
            raise ValueError("initial native depth/stencil must match declared format and geometry")
        if depth_stencil_output is not None and (type(depth_stencil_output) is not bytearray or
                                                len(depth_stencil_output) != expected_size or not depth_size):
            raise ValueError("native depth/stencil output requires writable matching staging")
        if depth_size:
            depth_plane = bytearray(initial_depth_stencil if initial_depth_stencil is not None else
                                    pack_depth_stencil(pipeline_state.depth_stencil_format, 1, 0) * (width * height))
    elif render_state is not None or initial_depth_stencil is not None or depth_stencil_output is not None:
        raise ValueError("general render state requires explicit ABI4 capture")
    shaders = artifact.get("shaders")
    if not isinstance(shaders, dict) or any(not isinstance(shaders.get(stage), dict) for stage in ("vs", "fs")):
        raise ValueError("capture must contain vertex and fragment shader objects")
    vertex_layout = ShaderLayout(shaders["vs"]["layout"])
    fragment_layout = ShaderLayout(shaders["fs"]["layout"])
    if instancing != bool(vertex_layout.host_instancing) or instancing != bool(fragment_layout.host_instancing):
        raise ValueError("capture and both shader instancing contracts must agree")
    if fragment_layout.host_texture:
        if artifact["schema"] not in ("mgc-host-draw-v2", "mgc-host-draw-v3", "mgc-host-draw-v4"):
            raise ValueError("host texture requires version2 captured resources")
    elif artifact["schema"] not in ("mgc-host-draw-v1", "mgc-host-draw-v3", "mgc-host-draw-v4") or artifact.get("textures"):
        raise ValueError("texture capture requires an explicit host texture shader contract")
    if fragment_layout.source.get("uses_discard") and not fragment_layout.fragment_status:
        raise ValueError("discard requires an explicit shader output-commit contract")
    mapping = link_varyings(vertex_layout, fragment_layout)
    if type(artifact.get("shader_abi_version")) is not int or artifact.get("program_format") != vertex_layout.target.value or artifact.get("shader_abi_version") != vertex_layout.source["shader_abi_version"]:
        raise ValueError("capture and shader target/ABI disagree")
    position = {entry["component"]: entry["word_offset"] for entry in vertex_layout.entries("outputs") if entry["location"] == 0}
    if set(position) != {0, 1, 2, 3}:
        raise ValueError("host rasterization requires reflected vec4 position output")
    colors = fragment_layout.entries("outputs")
    if any(entry["location"] not in (2, 4) or entry["type"] != "float" for entry in colors) or {entry["component"] for entry in colors} != {0, 1, 2, 3} or len(colors) != 4:
        raise ValueError("host renderer requires exactly one floating RGBA output")
    vertices = fetch_vertices(artifact, vertex_layout)
    constants = artifact.get("constants", {})
    if not isinstance(constants, dict):
        raise ValueError("captured constants must be a stage-slot object")
    vertex = ShaderExecutor(shaders["vs"]["assembly"], vertex_layout,
                            pack_constants(vertex_layout, constants.get("vs", [])), execution_deadline=deadline)
    fragment = None
    try:
        fragment = ShaderExecutor(shaders["fs"]["assembly"], fragment_layout,
                                  pack_constants(fragment_layout, constants.get("fs", [])),
                                  texture=artifact.get("textures"), execution_deadline=deadline)
        transformed = []
        for vertex_ordinal, inputs in enumerate(vertices):
            if deadline is not None and time.monotonic() >= deadline:
                raise ValueError("draw execution deadline exceeded")
            outputs = (vertex.invoke(inputs, instance_id=vertex_ordinal // artifact["draw"]["count"],
                                     builtins=getattr(inputs, "builtins", None))
                       if instancing else vertex.invoke(inputs))
            coordinates = np.array([fp16_bits_to_float(outputs[position[component]]) for component in range(4)])
            varyings = {offset: (fp16_bits_to_float(outputs[source]) if
                                next(entry for entry in fragment_layout.entries("inputs") if entry["word_offset"] == offset)["type"] == "float"
                                else outputs[source]) for offset, source in mapping.items()}
            if not np.all(np.isfinite(coordinates)) or not all(math.isfinite(value) for value in varyings.values()):
                raise ValueError("vertex shader produced nonfinite position or varying")
            transformed.append({"position": coordinates, "varyings": varyings})
            if vertex.steps > max_steps:
                raise ValueError("draw exceeds aggregate instruction budget")
        image = np.zeros((height, width, 4), dtype=np.uint8) if initial_image is None else initial_image.copy()
        coverage = np.zeros((height, width), dtype=np.uint32)
        depth_buffer = np.ones((height, width), dtype=float) if initial_depth is None else initial_depth.astype(float) / 65535
        clipped_count = 0
        discarded_fragments = 0
        covered_fragment_invocations = 0
        for index in range(0, len(transformed), 3):
            original = transformed[index:index + 3]
            provoking = original[-1]
            for triangle in clip_triangle(original):
                clipped_count += 1
                for column, row, ordered, barycentric, perspective, depth, prepared, builtins in _render_samples(
                        fragment, fragment_layout, triangle, provoking, width, height, max_fragments, deadline):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise ValueError("draw execution deadline exceeded")
                    pixel_row = height - 1 - row
                    if quantized_depth:
                        depth = float(np.rint(np.clip(depth, 0, 1) * 65535)) / 65535
                    if depth_test and depth >= depth_buffer[pixel_row, column]:
                        continue
                    if prepared is None and fragment.invocations >= max_fragments:
                        raise ValueError("draw exceeds aggregate fragment budget")
                    inputs = _fragment_payload(fragment_layout, ordered, barycentric, perspective, provoking)
                    covered_fragment_invocations += 1
                    result = prepared if prepared is not None else fragment.invoke_fragment(inputs, builtins=builtins)
                    if vertex.steps + fragment.steps > max_steps:
                        raise ValueError("draw exceeds aggregate instruction budget")
                    if result.status == FragmentStatus.DISCARDED:
                        discarded_fragments += 1
                        continue
                    outputs = result.outputs
                    values = [0.0] * 4
                    for entry in colors:
                        values[entry["component"]] = fp16_bits_to_float(outputs[entry["word_offset"]])
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError("shader produced nonfinite framebuffer output")
                    if pipeline_state:
                        from tools.host_pipeline_state import commit_fragment, pack_color, unpack_color

                        destination = pack_color(pipeline_state.color_format, [int(value) / 255 for value in image[pixel_row, column]])
                        plane_offset = (pixel_row * width + column) * depth_size
                        attachment = None if depth_plane is None else bytes(depth_plane[plane_offset:plane_offset + depth_size])
                        committed = commit_fragment(pipeline_state, values, destination, source_depth=float(depth),
                                                    destination_depth_stencil=attachment,
                                                    front_facing=builtins["gl_FrontFacing"])
                        if depth_plane is not None:
                            depth_plane[plane_offset:plane_offset + depth_size] = committed.depth_stencil
                        if not committed.passed:
                            continue
                        image[pixel_row, column] = [round(value * 255) for value in unpack_color(pipeline_state.color_format, committed.color)]
                    else:
                        image[pixel_row, column] = blend_rgba8(values, image[pixel_row, column], blend)
                    coverage[pixel_row, column] += 1
                    depth_buffer[pixel_row, column] = depth
                    if vertex.steps + fragment.steps > max_steps:
                        raise ValueError("draw exceeds aggregate instruction budget")
        if deadline is not None and time.monotonic() >= deadline:
            raise ValueError("draw execution deadline exceeded")
        texture_contract = fragment_layout.host_texture
        texture_wrap = (texture_contract.get("wrap") or
                        f"S:{texture_contract['wrap_s']}/T:{texture_contract['wrap_t']}") if texture_contract else None
        report = {"pipeline": "actual Gallium draw artifact -> typed packing -> managed VS -> CPU triangle rasterizer -> managed FS -> RGBA8",
                  "fpga_execution": False, "rtl_timing_validated": False, "rasterization": "bounded CPU only",
                  "program_format": vertex_layout.target.value, "shader_abi_version": vertex_layout.source["shader_abi_version"],
                  "width": width, "height": height, "vertices": len(vertices), "triangles": len(vertices) // 3,
                  "instances": artifact["draw"]["instance_count"],
                  "instancing": "explicit host attribute-divisor v1" if instancing else "disabled",
                  "clipped_triangles": clipped_count, "fragments": fragment.invocations,
                  "discarded_fragments": discarded_fragments,
                  "committed_fragments": fragment.invocations - discarded_fragments,
                  "fragment_completion": "explicit legacy host status v1" if fragment_layout.fragment_status else "no-discard invocation completion",
                  "texture_sampling": (f"bounded legacy {fragment_layout.host_texture['filter']}-"
                                       f"{'clamp' if texture_wrap == 'clamp-to-edge' else texture_wrap} "
                                       f"v{fragment_layout.host_texture['version']}" +
                                       (f" explicit integer LOD {texture_contract['lod']} nearest-mip"
                                        if 'lod' in texture_contract else "")
                                       if fragment_layout.host_texture else "none"),
                  "covered_pixels": int(np.count_nonzero(coverage)), "max_overdraw": int(coverage.max()),
                  "host_steps": vertex.steps + fragment.steps, "rgba_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                  "linkage": "compiler semantic location/component reflection", "pixel_origin": "exported top-left; raster coordinates bottom-left",
                  "coverage_rule": "single sample, pixel centers, half-open top-left edges", "depth": "less" if depth_test else "disabled",
                  "depth_storage": "Z16_UNORM" if quantized_depth else "host float",
                  "blend": blend, "culling": "disabled", "multisampling": False,
                  "limitations": ["not physical GPU or OpenGL conformance", "only captured native 32-bit attributes and typed constants",
                                  "bounded triangle lists and explicit attribute-divisor instancing; no general texture/filter, arbitrary blend state or fragment-builtin support",
                                  "pipeline state is explicitly configured by this host renderer, not inferred from uncaptured Gallium state"]}
        if depth_output is not None:
            depth_output[:] = np.rint(np.clip(depth_buffer, 0, 1) * 65535).astype(np.uint16)
        if pipeline_state is not None:
            report.update({"render_state": pipeline_state.to_dict(),
                           "committed_fragments": int(coverage.sum()),
                           "helper_invocations": fragment.invocations - covered_fragment_invocations,
                           "host_features": artifact["host_features"],
                           "instancing": "explicit ABI4 attribute-divisor v1",
                           "depth": pipeline_state.depth_comparison if pipeline_state.depth_enabled else "disabled",
                           "depth_storage": pipeline_state.depth_stencil_format,
                           "blend": pipeline_state.to_dict()["blend"],
                           "limitations": ["not physical GPU execution or OpenGL conformance",
                                           "bounded CPU triangles, single-sample coverage and explicit typed captured state",
                                           "FP16 shader arithmetic, fine-quad derivatives only at convergent host barriers"]})
        if depth_stencil_output is not None:
            depth_stencil_output[:] = depth_plane
        return image, coverage, report
    finally:
        vertex.close()
        if fragment is not None:
            fragment.close()
