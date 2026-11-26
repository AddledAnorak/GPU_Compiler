import struct

import pytest

from tools.host_quad import LaneRegion, QuadSimulationBackend
from tools.mgcasm import assemble
from tools.mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from tools.mgcdriver import Driver, Program, ProgramFormat


def program(operation=1):
    return Program.from_words(assemble(f"""
movi_i32 65536, r0
load r0, r1
wait_load
movi_i32 1114073, r0
store r1, r0
wait_short
movi_i32 {operation}, r1
movi_i32 1114072, r0
store r1, r0
wait_short
movi_i32 131034, r0
load r0, r1
wait_load
movi_i32 1048576, r0
store r1, r0
wait_short
stop_core
"""), format=ProgramFormat.LEGACY)


def resources(driver, backend, values):
    inputs, outputs = [], []
    for lane, value in enumerate(values):
        inputs.append(driver.allocate(65536 * 4, alignment=4, address=65536 * 4 + lane * (6 << 20)))
        outputs.append(driver.allocate(65536 * 4, alignment=4, address=1048576 * 4 + lane * (6 << 20)))
        inputs[-1].upload(struct.pack("<I", float_to_fp16_bits(value)))
    backend.configure([[LaneRegion(65536 * 4, source), LaneRegion(1048576 * 4, target, True)]
                       for source, target in zip(inputs, outputs)])
    return inputs + outputs, outputs


@pytest.mark.parametrize("operation,expected", [(1, [2, 2, 4, 4]), (2, [3, 5, 3, 5])])
def test_actual_cooperative_quads_preserve_finite_differences_and_managed_buffers(operation, expected):
    backend = QuadSimulationBackend(input_base=65536, output_base=1048576)
    with Driver(backend) as driver:
        buffers, outputs = resources(driver, backend, [1, 3, 4, 8])
        result = driver.submit(program(operation), buffers=buffers, max_steps=200).wait(5)
        assert result.backend == "legacy-host-quad" and result.steps == 92
        actual = [fp16_bits_to_float(struct.unpack("<I", target.download(size=4))[0]) for target in outputs]
        assert actual == expected


def test_quad_rejects_unsubmitted_lane_resource():
    backend = QuadSimulationBackend(input_base=65536, output_base=1048576)
    with Driver(backend) as driver:
        buffers, outputs = resources(driver, backend, [1, 2, 3, 4])
        with pytest.raises(Exception, match="unsubmitted"):
            driver.submit(program(), buffers=buffers[:-1], max_steps=200).wait(5)
        assert all(target.download(size=4) == bytes(4) for target in outputs)


def test_quad_execution_budget_fails_without_substituting_derivatives():
    backend = QuadSimulationBackend(input_base=65536, output_base=1048576)
    with Driver(backend) as driver:
        buffers, outputs = resources(driver, backend, [1, 2, 3, 4])
        with pytest.raises(Exception, match="max_steps"):
            driver.submit(program(), buffers=buffers, max_steps=10).wait(5)
        assert all(target.download(size=4) == bytes(4) for target in outputs)


@pytest.mark.parametrize("operation", [0, 3])
def test_quad_unknown_mailbox_operations_fail_closed(operation):
    backend = QuadSimulationBackend(input_base=65536, output_base=1048576)
    with Driver(backend) as driver:
        buffers, outputs = resources(driver, backend, [1, 2, 3, 4])
        with pytest.raises(Exception, match="mailbox"):
            driver.submit(program(operation), buffers=buffers, max_steps=200).wait(5)
        assert all(target.download(size=4) == bytes(4) for target in outputs)
