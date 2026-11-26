import copy
import struct
import time

import numpy as np
import pytest

from tools.host_instancing import HOST_INSTANCING_CONTRACT, shader_instancing_contract
from tools.host_render import ShaderExecutor, render_artifact
from tools.mgcdriver import Driver, Fence
from tools.shader_layout import ShaderLayout


BASES = {"input": 0x10000, "uniform": 0x20000, "vertex": 0x30000, "fragment": 0x100000}


def deadline_artifact(instancing=False):
    abi = 3 if instancing else 1
    position = [{"word_offset": component, "location": 0, "component": component,
                 "type": "float", "bit_size": 32, "interpolation": "smooth"} for component in range(4)]
    outputs = [dict(entry, location=2) for entry in position]
    layouts = {}
    for stage in ("vertex", "fragment"):
        layouts[stage] = {"version": 3 if instancing else 1, "supported": True, "errors": [],
                          "program_format": "legacy-mgcsim-v1", "shader_abi_version": abi,
                          "address_units": "words", "stage": stage, "bases": dict(BASES),
                          "inputs": position if stage == "vertex" else [],
                          "outputs": position if stage == "vertex" else outputs, "uniforms": []}
        if instancing:
            layouts[stage]["host_instancing"] = shader_instancing_contract()
    vertex = f"; mgc legacy-simulator backend, ABI={abi}, stage=MESA_SHADER_VERTEX\n"
    fragment = f"; mgc legacy-simulator backend, ABI={abi}, stage=MESA_SHADER_FRAGMENT\n"
    for component in range(4):
        vertex += (f"movi_i32 {BASES['input'] + component}, r0\nload r0, r1\nwait_load\n"
                   f"movi_i32 {BASES['vertex'] + component}, r0\nstore r1, r0\n")
        fragment += (f"movi_i32 {15360 if component in (0, 3) else 0}, r1\n"
                     f"movi_i32 {BASES['fragment'] + component}, r0\nstore r1, r0\n")
    artifact = {"schema": "mgc-host-draw-v3" if instancing else "mgc-host-draw-v1",
                "program_format": "legacy-mgcsim-v1", "shader_abi_version": abi,
                "draw": {"mode": "triangles", "start": 0, "count": 3, "index_size": 0, "index_bias": 0,
                         "instance_count": 2 if instancing else 1, "start_instance": 0, "primitive_restart": False},
                "shaders": {"vs": {"assembly": vertex + "stop_core\n", "layout": layouts["vertex"]},
                            "fs": {"assembly": fragment + "stop_core\n", "layout": layouts["fragment"]}},
                "vertex_buffers": [{"stride": 16, "data_hex": b"".join(struct.pack("<4f", *point)
                    for point in ((-1, -1, 0, 1), (1, -1, 0, 1), (0, 1, 0, 1))).hex()}],
                "vertex_elements": [{"src_offset": 0, "vertex_buffer_index": 0,
                                     "instance_divisor": 0, "format": "R32G32B32A32_FLOAT"}],
                "constants": {"vs": [], "fs": []}, "indices": None}
    if instancing:
        artifact["host_instancing"] = dict(HOST_INSTANCING_CONTRACT)
    return artifact


@pytest.mark.parametrize("instancing", (False, True))
@pytest.mark.parametrize("stage", ("vs", "fs"))
def test_draw_deadline_bounds_actual_shader_submission_and_wait_without_committing(monkeypatch, instancing, stage):
    artifact = deadline_artifact(instancing)
    artifact["shaders"][stage]["assembly"] += "never_done:\ngoto never_done\n"
    artifact["shaders"][stage]["assembly"] = artifact["shaders"][stage]["assembly"].replace("stop_core\n", "")
    original = copy.deepcopy(artifact)
    image = np.full((1, 1, 4), 91, np.uint8)
    depth = np.full((1, 1), 65535, np.uint16)
    depth_output = np.full((1, 1), 12345, np.uint16)
    submissions, waits = [], []
    original_submit, original_wait = Driver.submit, Fence.wait

    def bounded_submit(driver, program, **options):
        submissions.append(options["execution_timeout"])
        return original_submit(driver, program, **options)

    def bounded_wait(fence, timeout=None):
        waits.append(timeout)
        return original_wait(fence, timeout)

    monkeypatch.setattr(Driver, "submit", bounded_submit)
    monkeypatch.setattr(Fence, "wait", bounded_wait)
    started = time.monotonic()
    with pytest.raises((ValueError, TimeoutError), match="deadline|timeout|expired|instruction|step|budget"):
        render_artifact(artifact, width=1, height=1, depth_test=True, initial_image=image,
                        initial_depth=depth, depth_output=depth_output, execution_timeout=1)
    assert submissions and waits
    assert all(0 < timeout <= 1 for timeout in submissions + waits)
    assert time.monotonic() - started < 3
    assert artifact == original
    assert np.all(image == 91) and np.all(depth == 65535) and np.all(depth_output == 12345)


@pytest.mark.parametrize("instancing", (False, True))
@pytest.mark.parametrize("stage", ("vs", "fs"))
def test_prepared_nonterminating_invocation_reaches_deadline_before_instruction_limit(instancing, stage):
    shader = deadline_artifact(instancing)["shaders"][stage]
    assembly = shader["assembly"].replace("stop_core\n", "never_done:\ngoto never_done\n")
    executor = ShaderExecutor(assembly, ShaderLayout(shader["layout"]), max_steps=1000000)
    executor.execution_deadline = time.monotonic() + 0.02
    started = time.monotonic()
    try:
        with pytest.raises((ValueError, TimeoutError), match="deadline|timeout|expired"):
            executor.invoke({component: 15360 for component in range(4)} if stage == "vs" else {},
                            instance_id=0 if instancing and stage == "vs" else None)
        assert executor.invocations == 0 and executor.steps == 0
        assert time.monotonic() - started < 1
    finally:
        executor.close()


def test_standalone_executor_retains_original_submission_and_wait_defaults(monkeypatch):
    artifact = deadline_artifact()
    submitted, waited = [], []
    original_submit, original_wait = Driver.submit, Fence.wait

    def record_submit(driver, program, **options):
        submitted.append(options["execution_timeout"])
        return original_submit(driver, program, **options)

    def record_wait(fence, timeout=None):
        waited.append(timeout)
        return original_wait(fence, timeout)

    monkeypatch.setattr(Driver, "submit", record_submit)
    monkeypatch.setattr(Fence, "wait", record_wait)
    shader = artifact["shaders"]["fs"]
    executor = ShaderExecutor(shader["assembly"], ShaderLayout(shader["layout"]))
    try:
        assert executor.invoke({}) == {0: 15360, 1: 0, 2: 0, 3: 15360}
    finally:
        executor.close()
    assert submitted == [5] and waited == [10]


@pytest.mark.parametrize("deadline", (True, float("nan"), float("inf"), "1"))
def test_shader_deadline_type_is_fail_closed(deadline):
    shader = deadline_artifact()["shaders"]["fs"]
    with pytest.raises(ValueError, match="deadline"):
        ShaderExecutor(shader["assembly"], ShaderLayout(shader["layout"]), execution_deadline=deadline)
