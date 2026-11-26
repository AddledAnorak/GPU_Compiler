"""Literal vector artifacts only: no compiler build, legacy execution or board I/O."""

import copy
import math

import pytest

from tools.mgcdriver import Buffer, Driver, Fence, FenceTimeoutError, HardwareSimulationBackend, ProgramFormat
from tools.mgchw import assemble_hardware
from tools.mgcvector import VECTOR_TARGET, VectorExecutionError, VectorShaderExecutor
from tools.mgcvector import runtime
from tools.shader_layout import ShaderLayout


BASES = {"input": 0x40000, "uniform": 0x80000, "vertex": 0xC0000, "fragment": 0x400000}
LAYOUT = {
    "version": 1,
    "supported": True,
    "program_format": "fp16-hardware-v1",
    "compiler_target": "fp16-hardware-vector-experimental-v1",
    "shader_abi_version": 2,
    "address_units": "bytes",
    "stage": "fragment",
    "bases": BASES,
    "inputs": [{"word_offset": 2, "location": 1, "component": 0, "type": "float",
                "bit_size": 32, "interpolation": "smooth"}],
    "uniforms": [{"word_offset": 5, "type": "float", "bit_size": 32}],
    "outputs": [{"word_offset": 3, "location": 0, "component": 0, "type": "float",
                 "bit_size": 32, "interpolation": "smooth"}],
    "vector_contract": {
        "version": 1,
        "lanes": 4,
        "lane_mode": "broadcast-scalar",
        "scratch_word_offset": 32768,
        "scratch_words": 2,
        "status_word_offset": 65535,
        "pending": 0,
        "failed": 1,
        "committed": 2,
        "approximate_fp16_required": True,
        "vector_spacing_nops": 16,
        "rtl_timing_validated": False,
    },
}
SPACING = "nop\n" * 16


def layout(stage="fragment"):
    reflected = copy.deepcopy(LAYOUT)
    reflected["stage"] = stage
    return reflected


def source(body, stage="fragment"):
    return f"; mgc {VECTOR_TARGET} backend, ABI=2, stage={stage}\n{body}\n"


def committed_source(body="nop", stage="fragment", status=2):
    return source(f"""{body}
movi_i32 {BASES[stage] + 65535 * 4}, r1
movi_u16 {status}, r5
store r5, r1""", stage)


def multiply(stage="fragment", operation="vfmul16"):
    return committed_source(f"""movi_i32 {BASES['input'] + 2 * 4}, r1
load r1, r2
wait_load
vbcast r2, v0
{SPACING}movi_i32 {BASES['uniform'] + 5 * 4}, r1
load r1, r3
wait_load
vbcast r3, v1
{SPACING}{operation} v0, v1, v2
{SPACING}vget_lane0_i32 v2, r4
{SPACING}movi_i32 {BASES[stage] + 32768 * 4}, r1
store r4, r1
load r1, r6
wait_load
copy r6, r4
movi_i32 {BASES[stage] + 3 * 4}, r1
store r4, r1""", stage)


def executor(assembly=None, reflected=None, **options):
    return VectorShaderExecutor(multiply() if assembly is None else assembly,
                                layout() if reflected is None else reflected,
                                allow_approximate_fp16=True, **options)


@pytest.mark.parametrize("stage", ["vertex", "fragment"])
@pytest.mark.parametrize("checked", [False, True])
def test_vector_multiply_uses_managed_hardware_format_and_real_fence_steps(stage, checked):
    reflected = layout(stage)
    assembly = multiply(stage)
    with executor(assembly, ShaderLayout(reflected) if checked else reflected) as shader:
        report = shader.execute({2: 1.5}, {5: 2.0})
        assert report == {
            "outputs": {3: 3.0}, "raw_outputs": {3: 0x4200}, "fence": "completed",
            "host_steps": len(assemble_hardware(assembly)), "program_format": "fp16-hardware-v1",
            "compiler_target": VECTOR_TARGET, "status": "committed", "fpga_execution": False,
            "rtl_timing_validated": False, "approximate_fp16": True,
        }
        assert type(shader.driver.backend) is HardwareSimulationBackend
        assert shader.driver.backend.allow_approximate_fp16 is True
        assert shader.driver.backend.quantum == 4096
        assert shader.driver.backend.memory_size == max(BASES.values()) + 262144
        assert shader.program.format == ProgramFormat.HARDWARE
        assert shader.driver.allocated_bytes == 24
        assert {(buffer.address, buffer.size) for buffer in shader.buffers.values()} == {
            (BASES["input"] + 8, 4), (BASES["uniform"] + 20, 4), (BASES[stage] + 12, 4),
            (BASES[stage] + 32768 * 4, 8), (BASES[stage] + 65535 * 4, 4),
        }
    assert shader.driver.closed
    assert shader.driver.allocated_bytes == 0


@pytest.mark.parametrize("operation,expected", [("vfadd16", 3.5), ("vfsub16", -0.5),
                                                ("vfmul16", 3.0), ("vfmax16", 2.0)])
def test_supported_vector_arithmetic(operation, expected):
    with executor(multiply(operation=operation)) as shader:
        assert shader.execute({2: 1.5}, {5: 2.0})["outputs"] == {3: expected}


@pytest.mark.parametrize("allowed", [False, None, 0, 1, "true"])
def test_approximation_requires_literal_true_before_backend_creation(monkeypatch, allowed):
    def forbidden_backend(**options):
        pytest.fail("an unapproved vector artifact reached the backend")

    monkeypatch.setattr(runtime, "HardwareSimulationBackend", forbidden_backend)
    with pytest.raises(VectorExecutionError, match="explicit allow_approximate_fp16=True"):
        VectorShaderExecutor(multiply(), layout(), allow_approximate_fp16=allowed)


def test_default_approximation_guard():
    with pytest.raises(VectorExecutionError, match="explicit"):
        VectorShaderExecutor(multiply(), layout())


@pytest.mark.parametrize("name,value", [
    ("version", True), ("version", 1.0), ("version", 2), ("lanes", 1), ("lanes", 4.0),
    ("lane_mode", "packed"), ("scratch_word_offset", 32769), ("scratch_words", 0),
    ("scratch_words", 16385), ("scratch_words", True), ("scratch_words", 2.0),
    ("status_word_offset", 65534), ("pending", False), ("failed", True), ("committed", 3),
    ("approximate_fp16_required", False), ("vector_spacing_nops", 15), ("rtl_timing_validated", True),
])
def test_contract_values_are_exact_and_typed(name, value):
    reflected = layout()
    reflected["vector_contract"][name] = value
    with pytest.raises(VectorExecutionError, match="vector_contract|scratch_words"):
        executor(reflected=reflected)


@pytest.mark.parametrize("name", list(LAYOUT["vector_contract"]))
def test_contract_requires_every_key(name):
    reflected = layout()
    del reflected["vector_contract"][name]
    with pytest.raises(VectorExecutionError, match="exactly"):
        executor(reflected=reflected)


def test_contract_rejects_unknown_keys():
    reflected = layout()
    reflected["vector_contract"]["extra"] = 1
    with pytest.raises(VectorExecutionError, match="exactly"):
        executor(reflected=reflected)


@pytest.mark.parametrize("name,value", [
    ("version", 2), ("version", True), ("shader_abi_version", 1), ("shader_abi_version", 2.0),
    ("program_format", VECTOR_TARGET), ("program_format", "legacy-mgcsim-v1"),
    ("compiler_target", "fp16-hardware-v1"), ("address_units", "words"),
    ("stage", "compute"), ("supported", False),
])
def test_reflection_target_abi_and_stage_are_checked(name, value):
    reflected = layout()
    reflected[name] = value
    with pytest.raises(VectorExecutionError):
        executor(reflected=reflected)


@pytest.mark.parametrize("category", ["inputs", "uniforms", "outputs"])
@pytest.mark.parametrize("change", [{"type": "uint"}, {"type": "int"}, {"bit_size": 32.0},
                                     {"word_offset": True}, {"word_offset": 65536}])
def test_reflected_scalars_are_bounded_floats(category, change):
    reflected = layout()
    reflected[category][0].update(change)
    with pytest.raises(VectorExecutionError):
        executor(reflected=reflected)


@pytest.mark.parametrize("offset", [32768, 65535])
def test_output_cannot_overlap_scratch_or_status(offset):
    reflected = layout()
    reflected["outputs"][0]["word_offset"] = offset
    with pytest.raises(VectorExecutionError, match="below scratch"):
        executor(reflected=reflected)


@pytest.mark.parametrize("inputs,uniforms", [({}, {5: 2.0}), ({2: 1.5}, {}),
                                           ({2: 1.5, 3: 0.0}, {5: 2.0}),
                                           ({2: 1.5}, {5: 2.0, 6: 0.0})])
def test_payload_requires_exact_reflected_sets(inputs, uniforms):
    with executor() as shader:
        with pytest.raises(VectorExecutionError, match="match reflection exactly"):
            shader.execute(inputs, uniforms)
        assert shader.driver.pending_count == 0


@pytest.mark.parametrize("payload", [None, [], {True: 1.0}, {2.0: 1.0}, {"2": 1.0}, {-1: 1.0}])
def test_payload_dict_and_offsets_are_canonical(payload):
    with executor() as shader:
        with pytest.raises(VectorExecutionError):
            shader.execute(payload, {5: 2.0})


@pytest.mark.parametrize("value", [0, 0x3C00, True, "1.0", None, float("nan"),
                                   float("inf"), -float("inf"), 65520.0, 1e100])
@pytest.mark.parametrize("category", ["inputs", "uniforms"])
def test_payloads_reject_raw_integers_nonfinite_and_fp16_overflow(category, value):
    inputs, uniforms = {2: 1.5}, {5: 2.0}
    (inputs if category == "inputs" else uniforms)[2 if category == "inputs" else 5] = value
    with executor() as shader:
        with pytest.raises(VectorExecutionError, match="finite"):
            shader.execute(inputs, uniforms)
        assert shader.driver.pending_count == 0


@pytest.mark.parametrize("value,bits", [(65504.0, 0x7BFF), (-65504.0, 0xFBFF),
                                       (2.0 ** -24, 1), (-0.0, 0x8000), (1e-100, 0)])
def test_finite_fp16_payload_boundaries(value, bits):
    reflected = layout()
    reflected["uniforms"] = []
    assembly = committed_source(f"""movi_i32 {BASES['input'] + 8}, r1
load r1, r4
movi_i32 {BASES['fragment'] + 12}, r1
store r4, r1""")
    with executor(assembly, reflected) as shader:
        report = shader.execute({2: value}, {})
        assert report["raw_outputs"] == {3: bits}
        if value == 0.0:
            assert math.copysign(1.0, report["outputs"][3]) == math.copysign(1.0, value)


@pytest.mark.parametrize("body", ["fadd16 r1, r2, r3", "add r1, r2, r3", "istep4 r1, r2, r3",
                                 "vset4 r1, r2, v0", "vadd_i32 v0, v1, v2", "vfset4 r1, r2, v0",
                                 "vrgb565_ceramic v0, v1", "vstore_mask_r r1, v0, r2", "stop_core"])
def test_only_vector_compiler_allowlist_is_accepted(body):
    with pytest.raises(VectorExecutionError, match="unsupported vector instruction"):
        executor(source(body))


@pytest.mark.parametrize("assembly", [None, 17, [], "", "nop", source("nop", "vertex"),
                                      source("nop").replace(VECTOR_TARGET, "fp16-hardware-v1"),
                                      source("nop").replace("ABI=2", "ABI=1"),
                                      source("nop").replace("stage=fragment", "stage=fragmentx")])
def test_assembly_provenance_is_required(assembly):
    with pytest.raises(VectorExecutionError, match="provenance"):
        VectorShaderExecutor(assembly, layout(), allow_approximate_fp16=True)


@pytest.mark.parametrize("body", ["", "nop r1", "load r40, r2", "jump undefined"])
def test_empty_or_malformed_assembly_fails_closed(body):
    with pytest.raises(VectorExecutionError):
        executor(source(body))


def test_noncanonical_decoded_encoding_does_not_roundtrip(monkeypatch):
    def contaminated(assembly):
        words = assemble_hardware(assembly)
        if assembly.startswith("; mgc"):
            words[0] |= 1 << 25
        return words

    monkeypatch.setattr(runtime, "assemble_hardware", contaminated)
    with pytest.raises(VectorExecutionError, match="round-trip canonically"):
        executor(source("nop"))


def test_assembler_word_limit_is_rechecked(monkeypatch):
    monkeypatch.setattr(runtime, "assemble_hardware", lambda assembly: [0] * 262145)
    with pytest.raises(VectorExecutionError, match="program words"):
        executor(source("nop"))


def test_last_valid_and_first_invalid_program_word_counts():
    assembly = source("nop\n" * 262144)
    with executor(assembly, max_steps=1) as shader:
        assert len(shader.program.words) == 262144
    with pytest.raises(VectorExecutionError, match="program words"):
        executor(assembly + "nop\n")


@pytest.mark.parametrize("budget", [0, -1, True, 1.0, 1000001, None])
def test_instruction_budget_cannot_be_widened(budget):
    with pytest.raises(VectorExecutionError, match="max_steps"):
        executor(max_steps=budget)


@pytest.mark.parametrize("budget", [1, 1000000])
def test_instruction_budget_endpoints(budget):
    with executor(max_steps=budget) as shader:
        assert shader.max_steps == budget


def test_exact_executed_word_budget_succeeds_and_one_less_fails():
    assembly = multiply()
    steps = len(assemble_hardware(assembly))
    with executor(assembly, max_steps=steps) as shader:
        assert shader.execute({2: 1.5}, {5: 2.0})["host_steps"] == steps
    with executor(assembly, max_steps=steps - 1) as shader:
        with pytest.raises(VectorExecutionError, match="max_steps"):
            shader.execute({2: 1.5}, {5: 2.0})
        assert shader.driver.pending_count == 0
        assert all(not buffer.in_use for buffer in shader.buffers.values())


@pytest.mark.parametrize("body", ["jump -1", committed_source() + "jump -1"])
def test_budget_failure_returns_no_outputs_even_after_commit(body):
    assembly = body if body.startswith("; mgc") else source(body)
    with executor(assembly, max_steps=32) as shader:
        with pytest.raises(VectorExecutionError, match="max_steps") as failure:
            shader.execute({2: 1.5}, {5: 2.0})
        assert not hasattr(failure.value, "outputs")
        assert shader.driver.pending_count == 0


@pytest.mark.parametrize("offset", [-2, 4096])
def test_escaped_program_counter_is_an_execution_error(offset):
    with executor(source(f"jump {offset}")) as shader:
        with pytest.raises(VectorExecutionError, match="program counter|branch target"):
            shader.execute({2: 1.5}, {5: 2.0})


@pytest.mark.parametrize("address", [BASES["input"] + 4, BASES["input"] + 12,
                                    BASES["uniform"] + 16, BASES["uniform"] + 24,
                                    BASES["fragment"] + 8, BASES["fragment"] + 16,
                                    BASES["fragment"] + 32770 * 4, BASES["fragment"] + 49152 * 4,
                                    BASES["fragment"] + 65534 * 4, BASES["vertex"] + 12,
                                    0x90000000, BASES["input"] + 9])
def test_access_outside_reflected_or_declared_buffers_fails(address):
    assembly = committed_source(f"movi_i32 {address}, r1\nload r1, r4")
    with executor(assembly) as shader:
        with pytest.raises(VectorExecutionError, match="unmapped|unaligned"):
            shader.execute({2: 1.5}, {5: 2.0})


@pytest.mark.parametrize("fault", ["overlap", "alignment", "aperture"])
def test_invalid_abi_regions_fail_before_allocation(fault):
    reflected = layout()
    if fault == "overlap":
        reflected["bases"]["uniform"] = BASES["input"] + 4
    elif fault == "alignment":
        reflected["bases"]["input"] += 1
    else:
        reflected["bases"]["fragment"] = (32 << 20) - 262144 + 4
    with pytest.raises(VectorExecutionError, match="overlap|aligned|aperture"):
        executor(reflected=reflected)


def test_last_valid_32mb_aperture_and_scalar_boundaries():
    reflected = layout()
    reflected["bases"]["fragment"] = (32 << 20) - 262144
    reflected["inputs"][0]["word_offset"] = 65535
    reflected["uniforms"][0]["word_offset"] = 65535
    reflected["outputs"][0]["word_offset"] = 32767
    with executor(source("nop"), reflected) as shader:
        assert shader.driver.backend.memory_size == 32 << 20
        assert shader.buffers["input"].address == BASES["input"] + 65535 * 4
        assert shader.buffers["uniform"].address == BASES["uniform"] + 65535 * 4
        assert shader.buffers["fragment"].address == reflected["bases"]["fragment"] + 32767 * 4
        assert shader.driver.allocated_bytes == 24


@pytest.mark.parametrize("scratch_words", [1, 16384])
def test_scratch_has_exact_declared_size_and_guard(scratch_words):
    reflected = layout()
    reflected["vector_contract"]["scratch_words"] = scratch_words
    last = BASES["fragment"] + (32768 + scratch_words - 1) * 4
    assembly = committed_source(f"movi_i32 {last}, r1\nstore r0, r1")
    with executor(assembly, reflected) as shader:
        assert shader.buffers["scratch"].size == scratch_words * 4
        assert shader.execute({2: 1.5}, {5: 2.0})["status"] == "committed"
    with executor(assembly.replace(str(last), str(last + 4)), reflected) as shader:
        with pytest.raises(VectorExecutionError, match="unmapped"):
            shader.execute({2: 1.5}, {5: 2.0})


@pytest.mark.parametrize("status", [0, 1, 3, 65535])
def test_uncommitted_status_never_reads_or_returns_outputs(monkeypatch, status):
    downloads = []
    original = Buffer.download

    def record_download(buffer, **options):
        downloads.append(buffer.address)
        return original(buffer, **options)

    monkeypatch.setattr(Buffer, "download", record_download)
    with executor(committed_source(status=status)) as shader:
        with pytest.raises(VectorExecutionError, match="not committed") as failure:
            shader.execute({2: 1.5}, {5: 2.0})
        assert downloads == [BASES["fragment"] + 65535 * 4]
        assert not hasattr(failure.value, "outputs")
        assert not hasattr(failure.value, "raw_outputs")


def test_missing_commit_store_fails():
    with executor(source("nop")) as shader:
        with pytest.raises(VectorExecutionError, match="status=0"):
            shader.execute({2: 1.5}, {5: 2.0})


@pytest.mark.parametrize("word", [0x7C00, 0xFC00, 0x7E00, 0xFFFF, 0x10000, 0x13C00])
def test_nonfinite_or_high_bit_output_fails_after_commit(word):
    assembly = committed_source(f"movi_i32 {word}, r4\nmovi_i32 {BASES['fragment'] + 12}, r1\nstore r4, r1")
    with executor(assembly) as shader:
        with pytest.raises(VectorExecutionError, match="finite low16 FP16 with upper16 zero") as failure:
            shader.execute({2: 1.5}, {5: 2.0})
        assert not hasattr(failure.value, "outputs")


def test_finite_inputs_can_still_overflow_vector_output():
    with executor() as shader:
        with pytest.raises(VectorExecutionError, match="finite low16"):
            shader.execute({2: 65504.0}, {5: 2.0})


def test_every_output_span_word_is_validated_including_unreflected_holes():
    reflected = layout()
    second = dict(reflected["outputs"][0], word_offset=5, component=1)
    reflected["outputs"].append(second)
    assembly = committed_source(f"movi_u16 0x7E00, r4\nmovi_i32 {BASES['fragment'] + 16}, r1\nstore r4, r1")
    with executor(assembly, reflected) as shader:
        assert shader.buffers["fragment"].size == 12
        with pytest.raises(VectorExecutionError, match="finite low16"):
            shader.execute({2: 1.5}, {5: 2.0})


def conditional_source(skip_commit):
    body = f"""movi_i32 {BASES['input'] + 8}, r1
load r1, r2
movi_u16 0, r6
icmp_gt r2, r6, cr0
[!cr0] jump skipped
movi_f16 1.0, r4
movi_i32 {BASES['fragment'] + 12}, r1
store r4, r1
movi_i32 {BASES['fragment'] + 32768 * 4}, r1
store r4, r1
skipped:
{"[!cr0] jump done" if skip_commit else "nop"}"""
    return committed_source(body) + "done: nop\n"


def test_reused_mutable_buffers_clear_output_status_and_scratch():
    with executor(conditional_source(False)) as shader:
        assert shader.execute({2: 1.0}, {5: 2.0})["outputs"] == {3: 1.0}
        shader.buffers["fragment"].upload((0x7E00).to_bytes(4, "little"))
        shader.buffers["status"].upload((1).to_bytes(4, "little"))
        shader.buffers["scratch"].upload(bytes([255]) * 8)
        report = shader.execute({2: 0.0}, {5: 2.0})
        assert report["outputs"] == {3: 0.0}
        assert report["raw_outputs"] == {3: 0}
        assert report["host_steps"] < len(shader.program.words)
        assert shader.buffers["scratch"].download() == bytes(8)
        assert shader.buffers["status"].download() == (2).to_bytes(4, "little")


def test_previous_commit_cannot_hide_a_missing_commit_on_reuse():
    with executor(conditional_source(True)) as shader:
        assert shader.execute({2: 1.0}, {5: 2.0})["status"] == "committed"
        with pytest.raises(VectorExecutionError, match="status=0"):
            shader.execute({2: 0.0}, {5: 2.0})
        assert shader.buffers["fragment"].download() == bytes(4)
        assert shader.buffers["scratch"].download() == bytes(8)
        assert shader.execute({2: 1.0}, {5: 2.0})["outputs"] == {3: 1.0}


@pytest.mark.parametrize("checked", [False, True])
def test_reflection_mutation_cannot_change_validated_invocation(checked):
    reflected = layout()
    with executor(reflected=ShaderLayout(reflected) if checked else reflected) as shader:
        reflected["inputs"].clear()
        reflected["vector_contract"]["scratch_words"] = 16385
        reflected["bases"]["fragment"] = 0x90000000
        shader.layout.source["outputs"][0]["word_offset"] = 65535
        shader.layout.source["stage"] = "vertex"
        shader.layout.source["compiler_target"] = "legacy-mgcsim-v1"
        assert shader.execute({2: 1.5}, {5: 2.0})["outputs"] == {3: 3.0}
        assert shader.driver.allocated_bytes == 24


def test_payload_is_snapshotted_before_mutable_buffer_upload(monkeypatch):
    inputs, uniforms = {2: 1.5}, {5: 2.0}
    original = Buffer.upload

    def mutate_payload(buffer, data, **options):
        inputs[2] = float("nan")
        uniforms.clear()
        return original(buffer, data, **options)

    with executor() as shader:
        monkeypatch.setattr(Buffer, "upload", mutate_payload)
        assert shader.execute(inputs, uniforms)["outputs"] == {3: 3.0}


@pytest.mark.parametrize("operation", ["icmp_lt", "icmp_ge", "icmp_gt"])
def test_predication_and_actual_steps_are_not_static_program_size(operation):
    assembly = committed_source(f"""movi_u16 0, r2
movi_u16 1, r3
{operation} r2, r3, cr0
[cr0] nop
jump done
nop
nop
done: nop""")
    with executor(assembly) as shader:
        assert shader.execute({2: 1.5}, {5: 2.0})["host_steps"] == len(shader.program.words) - 2


def test_fixed_wait_and_execution_timeout_and_no_user_timeout_knob(monkeypatch):
    observed = {}
    original_wait, original_submit = Fence.wait, Driver.submit

    def wait(fence, timeout=None):
        observed["wait"] = timeout
        return original_wait(fence, timeout=timeout)

    def submit(driver, program, **options):
        observed["execution"] = options["execution_timeout"]
        observed["budget"] = options["max_steps"]
        return original_submit(driver, program, **options)

    monkeypatch.setattr(Fence, "wait", wait)
    monkeypatch.setattr(Driver, "submit", submit)
    with executor(max_steps=1000) as shader:
        shader.execute({2: 1.5}, {5: 2.0})
        assert observed == {"wait": 5.0, "execution": 5.0, "budget": 1000}
        with pytest.raises(TypeError):
            shader.execute({2: 1.5}, {5: 2.0}, timeout=100)
    with pytest.raises(TypeError):
        executor(timeout=100)


def test_wait_timeout_cancels_live_submission_and_allows_safe_reuse(monkeypatch):
    original = Fence.wait

    def expire(fence, timeout=None):
        raise FenceTimeoutError("fixed wait expired")

    with executor() as shader:
        monkeypatch.setattr(Fence, "wait", expire)
        with pytest.raises(VectorExecutionError, match="fixed wait expired"):
            shader.execute({2: 1.5}, {5: 2.0})
        assert shader.driver.pending_count == 0
        assert all(not buffer.in_use for buffer in shader.buffers.values())
        monkeypatch.setattr(Fence, "wait", original)
        assert shader.execute({2: 1.5}, {5: 2.0})["outputs"] == {3: 3.0}


def test_allocation_failure_closes_partially_created_driver(monkeypatch):
    drivers = []
    original = Driver.allocate

    def allocate(driver, size, **options):
        if driver not in drivers:
            drivers.append(driver)
        if driver.allocated_bytes:
            raise RuntimeError("allocation failed")
        return original(driver, size, **options)

    monkeypatch.setattr(Driver, "allocate", allocate)
    with pytest.raises(VectorExecutionError, match="allocation failed"):
        executor()
    assert drivers[0].closed
    assert drivers[0].allocated_bytes == 0


def test_empty_reflected_domains_allocate_only_scratch_and_status():
    reflected = layout()
    reflected.update(inputs=[], outputs=[], uniforms=[])
    with executor(committed_source(), reflected) as shader:
        assert set(shader.buffers) == {"scratch", "status"}
        assert shader.driver.allocated_bytes == 12
        assert shader.execute({}, {})["outputs"] == {}


def test_closed_executor_fails_without_outputs_and_close_is_idempotent():
    shader = executor()
    shader.close()
    shader.close()
    with pytest.raises(VectorExecutionError, match="closed"):
        shader.execute({2: 1.5}, {5: 2.0})
    with pytest.raises(VectorExecutionError, match="closed"):
        shader.__enter__()
