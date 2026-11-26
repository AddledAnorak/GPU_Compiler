"""Explicit legacy fragment status, tested independently of framebuffer values."""

import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

import numpy as np
import pytest

from tools.host_render import FragmentResult, FragmentStatus, ShaderExecutor, pack_float, render_artifact
from tools.mgcasm import assemble
from tools.mgcdriver import ExecutionTimeoutError, Fence, ProgramFormat, ResourceError, SimulationBackend, SubmissionError
from tools.run_render import capture_draw, softpipe_reference
from tools.shader_layout import ShaderLayout


ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
FIXTURE = ROOT / "tests/fixtures/fragment-discard"
BASES = {"input": 0x10000, "uniform": 0x20000, "vertex": 0x30000, "fragment": 0x100000}
STATUS = {"version": 1, "word_offset": 65535, "pending": 0, "discarded": 1, "committed": 2}
RGBA = (0.25, 0.5, 0.75, 0.0)


def scalar(offset, component=0, location=2, kind="float"):
    return {"word_offset": offset, "location": location, "component": component,
            "type": kind, "bit_size": 32, "interpolation": "smooth"}


def make_layout(*, stage="fragment", target=ProgramFormat.LEGACY, status=True, discard=False):
    hardware = target == ProgramFormat.HARDWARE
    source = {"version": 1, "supported": True, "errors": [], "program_format": target.value,
              "shader_abi_version": 2 if hardware else 1, "address_units": "bytes" if hardware else "words",
              "stage": stage, "bases": {name: base * (4 if hardware else 1) for name, base in BASES.items()},
              "inputs": [], "outputs": [scalar(component, component) for component in range(4)],
              "uniforms": [], "uses_discard": discard}
    if status:
        source["fragment_status"] = dict(STATUS)
    return source


def header(stage="fragment"):
    return f"; mgc legacy-simulator backend, ABI=1, stage=MESA_SHADER_{stage.upper()}\n"


def store_word(domain, offset, value):
    immediate = f"movi_i32 {value}, r1\n"
    if value == 0xFFFFFFFF:
        immediate = ("movi_i32 134217727, r1\n" + "add r1, r1, r1\nwait_short\n" * 5
                     + "movi_i32 31, r2\nadd r1, r2, r1\nwait_short\n")
    return immediate + f"movi_i32 {BASES[domain] + offset}, r0\nstore r1, r0\n"


def fragment_assembly(signal=2, values=RGBA, ending="stop_core\n"):
    assembly = header()
    for component, value in enumerate(values):
        assembly += store_word("fragment", component, pack_float(value))
    if signal is not None:
        assembly += store_word("fragment", STATUS["word_offset"], signal)
    return assembly + ending


def invoke(assembly, source, inputs=None, *, fragment=True, max_steps=100000):
    executor = ShaderExecutor(assembly, ShaderLayout(source), max_steps=max_steps)
    try:
        assert isinstance(executor.driver.backend, SimulationBackend)
        return executor.invoke_fragment(inputs or {}) if fragment else executor.invoke(inputs or {})
    finally:
        executor.close()


def assert_result(result, status, outputs):
    assert isinstance(result, FragmentResult)
    assert result.status is status
    assert isinstance(result.status, str)
    assert result.status == status.value
    assert result.outputs == outputs


def output_words(values):
    return {component: pack_float(value) for component, value in enumerate(values)}


def quad_records(depths=(0.0,), left=-1.0, right=1.0):
    corners = [(left, -1), (right, -1), (right, 1),
               (left, -1), (right, 1), (left, 1)]
    return [(horizontal, vertical, depth, 1.0, 0.0, 0.0, 1.0)
            for depth in depths for horizontal, vertical in corners]


def write_mesh(path, depths=(0.0,), left=-1.0):
    records = quad_records(depths, left)
    path.write_bytes(b"".join(struct.pack("<7f", *record) for record in records))
    assert path.stat().st_size == len(records) * 28
    return path


def synthetic_draw(signal=2, values=RGBA, *, status=True, discard=False):
    vertex = make_layout(stage="vertex", status=False)
    vertex["inputs"] = [scalar(component, component, 0) for component in range(4)]
    vertex["outputs"] = copy.deepcopy(vertex["inputs"])
    vertex_assembly = header("vertex")
    for component in range(4):
        vertex_assembly += (f"movi_i32 {BASES['input'] + component}, r0\nload r0, r1\nwait_load\n"
                            f"movi_i32 {BASES['vertex'] + component}, r0\nstore r1, r0\n")
    vertex_assembly += "stop_core\n"
    return {"schema": "mgc-host-draw-v1", "program_format": ProgramFormat.LEGACY.value,
            "shader_abi_version": 1,
            "draw": {"mode": "triangles", "count": 6, "start": 0, "instance_count": 1,
                     "start_instance": 0, "primitive_restart": False, "index_size": 0, "index_bias": 0},
            "shaders": {"vs": {"assembly": vertex_assembly, "layout": vertex},
                        "fs": {"assembly": fragment_assembly(signal, values),
                               "layout": make_layout(status=status, discard=discard)}},
            "vertex_buffers": [{"stride": 28,
                                "data_hex": b"".join(struct.pack("<7f", *record)
                                                     for record in quad_records()).hex()}],
            "vertex_elements": [{"src_offset": 0, "vertex_buffer_index": 0,
                                 "instance_divisor": 0, "format": "R32G32B32A32_FLOAT"}],
            "constants": {"vs": [], "fs": []}, "indices": None}


@pytest.mark.parametrize("target", list(ProgramFormat))
@pytest.mark.parametrize("stage", ["vertex", "fragment"])
def test_fragment_status_is_absent_without_opt_in(target, stage):
    assert ShaderLayout(make_layout(stage=stage, target=target, status=False)).fragment_status is None


@pytest.mark.parametrize("discard", [False, True])
def test_fragment_status_is_exact_legacy_fragment_contract(discard):
    reflected = ShaderLayout(make_layout(discard=discard))
    assert reflected.fragment_status == STATUS


@pytest.mark.parametrize("invalid", [None, False, 1, "1", [], {}, {**STATUS, "extra": 0}])
def test_fragment_status_rejects_nonobjects_and_wrong_keys(invalid):
    source = make_layout()
    source["fragment_status"] = invalid
    with pytest.raises(ValueError):
        ShaderLayout(source)


@pytest.mark.parametrize("field", list(STATUS))
def test_fragment_status_requires_every_field(field):
    source = make_layout()
    del source["fragment_status"][field]
    with pytest.raises(ValueError):
        ShaderLayout(source)


@pytest.mark.parametrize("field", list(STATUS))
@pytest.mark.parametrize("invalid", [None, True, False, "1", 1.0, -1, 65536])
def test_fragment_status_fields_are_strict_exact_integers(field, invalid):
    source = make_layout()
    source["fragment_status"][field] = invalid
    with pytest.raises(ValueError):
        ShaderLayout(source)


@pytest.mark.parametrize("field,value", [("version", 0), ("version", 2), ("word_offset", 65534),
                                       ("pending", 1), ("discarded", 0), ("discarded", 2),
                                       ("committed", 0), ("committed", 1)])
def test_fragment_status_rejects_other_versions_offsets_and_state_encodings(field, value):
    source = make_layout()
    source["fragment_status"][field] = value
    with pytest.raises(ValueError):
        ShaderLayout(source)


@pytest.mark.parametrize("target,stage", [(ProgramFormat.LEGACY, "vertex"),
                                        (ProgramFormat.HARDWARE, "fragment"),
                                        (ProgramFormat.HARDWARE, "vertex")])
def test_fragment_status_is_legacy_fragment_only(target, stage):
    with pytest.raises(ValueError):
        ShaderLayout(make_layout(target=target, stage=stage))


def test_fragment_status_reserves_only_its_fragment_output_word():
    source = make_layout()
    source["inputs"] = [scalar(65535, location=32)]
    source["uniforms"] = [{"word_offset": 65535, "type": "uint", "bit_size": 32}]
    source["outputs"] = [scalar(65534)]
    assert ShaderLayout(source).fragment_status == STATUS
    source["outputs"][0]["word_offset"] = 65535
    with pytest.raises(ValueError):
        ShaderLayout(source)
    del source["fragment_status"]
    assert ShaderLayout(source).entries("outputs")[0]["word_offset"] == 65535


@pytest.mark.parametrize("values", [RGBA, (0.0, 0.0, 0.0, 0.0)])
def test_explicit_commit_accepts_zero_alpha_and_all_zero_rgba(values):
    source = make_layout()
    assert_result(invoke(fragment_assembly(2, values), source), FragmentStatus.COMMITTED, output_words(values))
    assert invoke(fragment_assembly(2, values), source, fragment=False) == output_words(values)


@pytest.mark.parametrize("values", [RGBA, (0.0, 0.0, 0.0, 0.0)])
def test_explicit_discard_returns_no_outputs_and_normal_invoke_raises(values):
    source = make_layout(discard=True)
    assembly = fragment_assembly(1, values)
    assert_result(invoke(assembly, source), FragmentStatus.DISCARDED, {})
    with pytest.raises(ValueError, match="discard"):
        invoke(assembly, source, fragment=False)


def test_discard_does_not_read_or_decode_poisoned_outputs(monkeypatch):
    assembly = header() + "".join(store_word("fragment", component, 0x7C00) for component in range(4))
    assembly += store_word("fragment", 65535, 1) + "stop_core\n"
    executor = ShaderExecutor(assembly, ShaderLayout(make_layout(discard=True)))
    original = executor.buffers["fragment"].download

    def download_status_only(*, offset=0, size=None):
        assert offset == 65535 * 4 and size == 4
        return original(offset=offset, size=size)

    monkeypatch.setattr(executor.buffers["fragment"], "download", download_status_only)
    try:
        assert_result(executor.invoke_fragment({}), FragmentStatus.DISCARDED, {})
    finally:
        executor.close()


@pytest.mark.parametrize("signal", [None, 0, 3, 0xFFFFFFFF])
@pytest.mark.parametrize("values", [RGBA, (0.0, 0.0, 0.0, 0.0)])
@pytest.mark.parametrize("fragment", [True, False])
def test_pending_or_unknown_status_is_not_a_discard_or_commit(signal, values, fragment):
    with pytest.raises(ValueError, match="signal|status|completion"):
        invoke(fragment_assembly(signal, values), make_layout(), fragment=fragment)


def test_explicit_status_is_read_only_after_successful_managed_fence(monkeypatch):
    executor = ShaderExecutor(fragment_assembly(), ShaderLayout(make_layout()))
    events = []
    original_wait = Fence.wait
    original_download = executor.buffers["fragment"].download

    def wait_then_record(fence, *args, **kwargs):
        result = original_wait(fence, *args, **kwargs)
        events.append("successful-fence")
        return result

    def download_after_fence(*, offset=0, size=None):
        assert events and events[0] == "successful-fence"
        events.append("status" if offset == 65535 * 4 else "output")
        return original_download(offset=offset, size=size)

    monkeypatch.setattr(Fence, "wait", wait_then_record)
    monkeypatch.setattr(executor.buffers["fragment"], "download", download_after_fence)
    try:
        assert_result(executor.invoke_fragment({}), FragmentStatus.COMMITTED, output_words(RGBA))
        assert events == ["successful-fence", "status", "output", "output", "output", "output"]
    finally:
        executor.close()


@pytest.mark.parametrize("signal", [1, 2])
@pytest.mark.parametrize("failure", ["execution", "budget"])
@pytest.mark.parametrize("fragment", [True, False])
def test_written_status_never_hides_execution_or_budget_failure(monkeypatch, signal, failure, fragment):
    ending = "movi_i32 0, r0\nload r0, r1\nwait_load\nstop_core\n" if failure == "execution" else "forever:\ngoto forever\n"
    executor = ShaderExecutor(fragment_assembly(signal, ending=ending),
                              ShaderLayout(make_layout(discard=True)), max_steps=64)

    def forbidden_download(*args, **kwargs):
        pytest.fail("fragment status/outputs read despite a failed managed fence")

    monkeypatch.setattr(executor.buffers["fragment"], "download", forbidden_download)
    try:
        with pytest.raises(SubmissionError if failure == "execution" else ExecutionTimeoutError) as caught:
            if fragment:
                executor.invoke_fragment({})
            else:
                executor.invoke({})
        if failure == "execution":
            assert isinstance(caught.value.__cause__, ResourceError)
    finally:
        executor.close()


def test_each_invocation_clears_previous_completion_signal():
    source = make_layout(discard=True)
    source["inputs"] = [scalar(0, location=32, kind="uint")]
    assembly = (header() + f"movi_i32 {BASES['input']}, r0\nload r0, r1\nwait_load\n"
                "movi_i32 0, r2\nicmp_eq r1, r2, cr0\nwait_short\ngoto.cr0 done\n"
                f"movi_i32 {BASES['fragment'] + 65535}, r0\nstore r1, r0\n"
                "done:\nstop_core\n")
    executor = ShaderExecutor(assembly, ShaderLayout(source))
    try:
        for signal in (2, 1, 2, 1):
            expected = FragmentStatus.COMMITTED if signal == 2 else FragmentStatus.DISCARDED
            assert_result(executor.invoke_fragment({0: signal}), expected,
                          output_words((0, 0, 0, 0)) if signal == 2 else {})
            with pytest.raises(ValueError, match="signal|status|completion"):
                executor.invoke_fragment({0: 0})
    finally:
        executor.close()


def test_unversioned_invoke_preserves_dictionary_behavior_without_claiming_discard():
    source = make_layout(status=False, discard=True)
    result = invoke(fragment_assembly(None), source, fragment=False)
    assert isinstance(result, dict) and result == output_words(RGBA)
    with pytest.raises(ValueError, match="discard"):
        invoke(fragment_assembly(None), source)


def test_fragment_invocation_rejects_vertex_layout():
    source = make_layout(stage="vertex", status=False)
    with pytest.raises(ValueError, match="fragment"):
        invoke(header("vertex") + "stop_core\n", source)


def test_identical_zero_framebuffers_have_distinct_explicit_commit_coverage():
    zero = (0.0, 0.0, 0.0, 0.0)
    committed_image, committed_coverage, committed = render_artifact(synthetic_draw(2, zero), width=4, height=4)
    discarded_image, discarded_coverage, discarded = render_artifact(
        synthetic_draw(1, zero, discard=True), width=4, height=4)
    np.testing.assert_array_equal(committed_image, discarded_image)
    assert np.all(committed_coverage == 1) and np.all(discarded_coverage == 0)
    assert committed["committed_fragments"] == 16 and committed["discarded_fragments"] == 0
    assert discarded["committed_fragments"] == 0 and discarded["discarded_fragments"] == 16
    assert committed["covered_pixels"] == 16 and discarded["covered_pixels"] == 0


def test_renderer_rejects_discard_without_explicit_status():
    with pytest.raises(ValueError, match="discard"):
        render_artifact(synthetic_draw(None, status=False, discard=True), width=4, height=4)


@pytest.mark.parametrize("signal", [None, 0, 3, 0xFFFFFFFF])
def test_renderer_rejects_invalid_completion_even_for_zero_rgba(signal):
    with pytest.raises(ValueError, match="signal|status|completion"):
        render_artifact(synthetic_draw(signal, (0, 0, 0, 0)), width=4, height=4)


@pytest.mark.parametrize("budget", ["instructions", "fragments"])
def test_discarded_invocations_count_toward_renderer_budgets(budget):
    artifact = synthetic_draw(1, discard=True)
    vertex_steps = 6 * len(assemble(artifact["shaders"]["vs"]["assembly"]))
    fragment_steps = len(assemble(artifact["shaders"]["fs"]["assembly"]))
    options = {"max_steps": vertex_steps + fragment_steps - 1} if budget == "instructions" else {"max_fragments": 1}
    with pytest.raises(ValueError, match="budget"):
        render_artifact(artifact, width=4, height=4, **options)


def run_checked(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def fragment_backend(tmp_path_factory):
    if not (BUILD / "build.ninja").is_file() or not (BUILD / "src/gallium/drivers/mgc/libmgc.a").is_file():
        pytest.skip("already-built Mesa required for direct-NIR fragment contract tests")
    ninja = shutil.which("ninja") or str(ROOT / ".venv/bin/ninja")
    directory = tmp_path_factory.mktemp("fragment-commit-helper")

    def command(target):
        result = run_checked([ninja, "-C", str(BUILD), "-t", "commands", target])
        return shlex.split(result.stdout.splitlines()[-1])

    target = "src/gallium/drivers/mgc/mgc_test_trig"
    original_object = target + ".p/mgc_test_trig.c.o"
    object_path = directory / "mgc_fragment_commit.c.o"
    binary = directory / "mgc_fragment_commit"
    compile_command = [str(object_path) + token[len(original_object):]
                       if token.startswith(original_object) else
                       str(FIXTURE / "mgc_fragment_commit.c") if token.endswith("/mgc_test_trig.c") else token
                       for token in command(original_object)]
    run_checked(compile_command, cwd=BUILD)
    link_command = [str(binary) if token == target else str(object_path) if token == original_object else token
                    for token in command(target)]
    run_checked(link_command, cwd=BUILD)
    return binary


def compile_fragment(binary, tmp_path, mode, version=1, target=ProgramFormat.LEGACY, *, success=True):
    result = subprocess.run([str(binary), mode, str(version), "hardware" if target == ProgramFormat.HARDWARE else "legacy"],
                            capture_output=True, text=True, timeout=60)
    (tmp_path / f"{mode}-{version}-{target.name}.log").write_text(result.stderr)
    assert result.returncode == (0 if success else 1), result.stdout + result.stderr
    if not success:
        assert result.stdout == ""
        return result.stderr
    reflected = [line.removeprefix("layout=") for line in result.stderr.splitlines() if line.startswith("layout=")]
    assert len(reflected) == 1, result.stderr
    source = json.loads(reflected[0])
    return result.stdout, ShaderLayout(source)


def compiled_inputs(reflected, coordinates):
    return {entry["word_offset"]: pack_float(coordinates[entry["component"]]) for entry in reflected.entries("inputs")}


def compiled_outputs(reflected, values):
    return {entry["word_offset"]: pack_float(values[entry["component"]]) for entry in reflected.entries("outputs")}


def test_compile_option_is_appended_uint32_and_defaults_remain_zero(fragment_backend):
    defaults = json.loads(run_checked([str(fragment_backend), "defaults"]).stdout)
    assert defaults == {"legacy": 0, "hardware": 0}


@pytest.mark.parametrize("target", list(ProgramFormat))
def test_default_and_explicit_zero_compile_options_preserve_assembly(fragment_backend, tmp_path, target):
    default_assembly, default_layout = compile_fragment(fragment_backend, tmp_path, "zero", "default", target)
    zero_assembly, zero_layout = compile_fragment(fragment_backend, tmp_path, "zero", 0, target)
    assert default_assembly == zero_assembly
    assert default_layout.source == zero_layout.source
    assert default_layout.fragment_status is None


@pytest.mark.parametrize("version,target", [(2, ProgramFormat.LEGACY), (0xFFFFFFFF, ProgramFormat.LEGACY),
                                          (1, ProgramFormat.HARDWARE)])
def test_compiler_rejects_unknown_status_version_and_hardware_opt_in(fragment_backend, tmp_path, version, target):
    diagnostic = compile_fragment(fragment_backend, tmp_path, "zero", version, target, success=False)
    assert "status" in diagnostic.lower()


def test_legacy_opt_in_applies_only_to_fragment_stage(fragment_backend, tmp_path):
    assembly, reflected = compile_fragment(fragment_backend, tmp_path, "vertex")
    default_assembly, default_layout = compile_fragment(fragment_backend, tmp_path, "vertex", 0)
    assert reflected.fragment_status is None
    assert assembly == default_assembly and reflected.source == default_layout.source


def test_fresh_compiler_commits_all_zero_rgba(fragment_backend, tmp_path):
    assembly, reflected = compile_fragment(fragment_backend, tmp_path, "zero")
    assert reflected.fragment_status == STATUS and reflected.source["uses_discard"] is False
    assert_result(invoke(assembly, reflected.source), FragmentStatus.COMMITTED,
                  compiled_outputs(reflected, (0, 0, 0, 0)))


@pytest.mark.parametrize("mode", ["early-demote-if", "early-terminate-if", "late-demote-if", "late-terminate-if"])
def test_direct_nir_conditional_discard_and_commit(fragment_backend, tmp_path, mode):
    assembly, reflected = compile_fragment(fragment_backend, tmp_path, mode)
    assert reflected.fragment_status == STATUS and reflected.source["uses_discard"] is True
    executor = ShaderExecutor(assembly, reflected)
    try:
        for coordinate in (-0.5, 0.5, -0.5, 0.5):
            inputs = compiled_inputs(reflected, (coordinate,))
            if coordinate < 0:
                assert_result(executor.invoke_fragment(inputs), FragmentStatus.DISCARDED, {})
                with pytest.raises(ValueError, match="discard"):
                    executor.invoke(inputs)
            else:
                expected = compiled_outputs(reflected, (0, 1, 0, 0))
                assert_result(executor.invoke_fragment(inputs), FragmentStatus.COMMITTED, expected)
                assert executor.invoke(inputs) == expected
    finally:
        executor.close()


@pytest.mark.parametrize("mode", ["early-demote-if", "early-terminate-if"])
def test_unversioned_early_discard_keeps_legacy_normal_invoke(fragment_backend, tmp_path, mode):
    assembly, reflected = compile_fragment(fragment_backend, tmp_path, mode, 0)
    assert reflected.fragment_status is None and reflected.source["uses_discard"] is True
    outputs = invoke(assembly, reflected.source, compiled_inputs(reflected, (-0.5,)), fragment=False)
    assert isinstance(outputs, dict)
    assert set(outputs) == {entry["word_offset"] for entry in reflected.entries("outputs")}


@pytest.mark.parametrize("mode", ["late-demote-if", "late-terminate-if", "late-cfg"])
def test_late_discard_requires_explicit_opt_in(fragment_backend, tmp_path, mode):
    diagnostic = compile_fragment(fragment_backend, tmp_path, mode, 0, success=False)
    assert "discard" in diagnostic.lower()


def test_acyclic_late_discard_commits_correct_branch_outputs(fragment_backend, tmp_path):
    assembly, reflected = compile_fragment(fragment_backend, tmp_path, "late-cfg")
    for coordinates, values in [((-0.5, -0.5), None), ((-0.5, 0.5), (0, 1, 0, 0)),
                                ((0.5, -0.5), (0, 0, 1, 0))]:
        result = invoke(assembly, reflected.source, compiled_inputs(reflected, coordinates))
        assert_result(result, FragmentStatus.DISCARDED if values is None else FragmentStatus.COMMITTED,
                      {} if values is None else compiled_outputs(reflected, values))


def test_every_normal_cfg_exit_reports_committed(fragment_backend, tmp_path):
    assembly, reflected = compile_fragment(fragment_backend, tmp_path, "returns")
    for coordinate, values in [(-0.5, (1, 0, 0, 0)), (0.5, (0, 0, 1, 0))]:
        assert_result(invoke(assembly, reflected.source, compiled_inputs(reflected, (coordinate,))),
                      FragmentStatus.COMMITTED, compiled_outputs(reflected, values))


@pytest.mark.parametrize("mode", ["demote", "terminate"])
def test_unconditional_discard_has_explicit_status(fragment_backend, tmp_path, mode):
    assembly, reflected = compile_fragment(fragment_backend, tmp_path, mode)
    assert_result(invoke(assembly, reflected.source, compiled_inputs(reflected, (0.5,))), FragmentStatus.DISCARDED, {})


def test_unversioned_discard_in_runtime_loop_remains_rejected(fragment_backend, tmp_path):
    diagnostic = compile_fragment(fragment_backend, tmp_path, "loop", 0, success=False)
    assert "loop" in diagnostic.lower() and "discard" in diagnostic.lower()


def test_opted_in_runtime_loop_has_distinct_commit_discard_and_budget_outcomes(fragment_backend, tmp_path):
    assembly, reflected = compile_fragment(fragment_backend, tmp_path, "loop", 1)
    executor = ShaderExecutor(assembly, reflected, max_steps=500)
    try:
        assert_result(executor.invoke_fragment(compiled_inputs(reflected, (0.5, -0.5))),
                      FragmentStatus.COMMITTED, compiled_outputs(reflected, (1, 0, 0, 0)))
        assert_result(executor.invoke_fragment(compiled_inputs(reflected, (-0.5, -0.5))),
                      FragmentStatus.DISCARDED, {})
        with pytest.raises(ExecutionTimeoutError, match="max_steps"):
            executor.invoke_fragment(compiled_inputs(reflected, (-0.5, 0.5)))
    finally:
        executor.close()


@pytest.mark.parametrize("mode,version", [("early-demote-if", 0), ("early-terminate-if", 0),
                                       ("late-demote-if", 0), ("early-demote-if", 1)])
def test_hardware_discard_is_rejected(fragment_backend, tmp_path, mode, version):
    compile_fragment(fragment_backend, tmp_path, mode, version, ProgramFormat.HARDWARE, success=False)


def test_compiler_guards_reserved_status_word_without_changing_old_boundary(fragment_backend, tmp_path):
    _, unversioned = compile_fragment(fragment_backend, tmp_path, "collision", 0)
    assert unversioned.entries("outputs")[0]["word_offset"] == 65535
    diagnostic = compile_fragment(fragment_backend, tmp_path, "collision", 1, success=False)
    assert "reserved" in diagnostic.lower() or "status" in diagnostic.lower()
    _, opted_in = compile_fragment(fragment_backend, tmp_path, "boundary", 1)
    assert opted_in.entries("outputs")[0]["word_offset"] == 65534


@pytest.mark.parametrize("left", [-1.0, 0.0], ids=["conditional-kill", "no-kill"])
def test_fresh_glsl_conditional_discard_matches_same_input_softpipe(tmp_path, left):
    mesh = write_mesh(tmp_path / "quad.host.bin", left=left)
    artifact = capture_draw(FIXTURE, BUILD, tmp_path / "capture", mesh=mesh)
    reflected = ShaderLayout(artifact["shaders"]["fs"]["layout"])
    assert reflected.fragment_status == STATUS and reflected.source["uses_discard"] is True
    assert ShaderLayout(artifact["shaders"]["vs"]["layout"]).fragment_status is None
    image, coverage, report = render_artifact(artifact, width=16, height=16)
    reference = softpipe_reference(FIXTURE, BUILD, tmp_path / "reference", mesh=mesh, depth_test=False)
    np.testing.assert_array_equal(image, reference)
    np.testing.assert_array_equal(image[:, 8:], np.broadcast_to([64, 128, 191, 0], (16, 8, 4)))
    assert np.all(coverage[:, :8] == 0) and np.all(coverage[:, 8:] == 1)
    assert report["committed_fragments"] == 128
    assert report["discarded_fragments"] == (128 if left < 0 else 0)
    assert report["fragments"] == report["committed_fragments"] + report["discarded_fragments"]
    assert report["covered_pixels"] == 128 and report["fpga_execution"] is False


def test_capture_explicit_opt_out_keeps_unversioned_discard_rejected_by_renderer(tmp_path):
    artifact = capture_draw(FIXTURE, BUILD, tmp_path / "capture", fragment_status=False)
    assert ShaderLayout(artifact["shaders"]["fs"]["layout"]).fragment_status is None
    with pytest.raises(ValueError, match="discard"):
        render_artifact(artifact, width=16, height=16)


def test_fresh_zero_rgba_is_committed_and_matches_softpipe(tmp_path):
    shader = FIXTURE / "zero-rgba"
    mesh = write_mesh(tmp_path / "quad.host.bin")
    artifact = capture_draw(shader, BUILD, tmp_path / "capture", mesh=mesh, fragment_status=True)
    reflected = ShaderLayout(artifact["shaders"]["fs"]["layout"])
    assert reflected.fragment_status == STATUS and reflected.source["uses_discard"] is False
    assert_result(invoke(artifact["shaders"]["fs"]["assembly"], reflected.source), FragmentStatus.COMMITTED,
                  compiled_outputs(reflected, (0, 0, 0, 0)))
    image, coverage, report = render_artifact(artifact, width=16, height=16)
    reference = softpipe_reference(shader, BUILD, tmp_path / "reference", mesh=mesh, depth_test=False)
    np.testing.assert_array_equal(image, reference)
    assert np.all(image == 0) and np.all(coverage == 1)
    assert report["committed_fragments"] == 256 and report["discarded_fragments"] == 0
    assert report["covered_pixels"] == 256


@pytest.mark.parametrize("depths", [(-0.5, 0.5), (0.5, -0.5)], ids=["near-then-far", "far-then-near"])
@pytest.mark.parametrize("shader_name", ["overlap", "branch-color", "runtime-loop"])
def test_discard_never_commits_depth_or_color_and_zero_alpha_kept_fragments_do(tmp_path, depths, shader_name):
    shader = FIXTURE / shader_name
    mesh = write_mesh(tmp_path / "overlap.host.bin", depths)
    artifact = capture_draw(shader, BUILD, tmp_path / "capture", mesh=mesh, fragment_status=True, depth_test=True)
    if shader_name == "runtime-loop":
        assert "mgc_block_" in artifact["shaders"]["fs"]["assembly"]
    assert ShaderLayout(artifact["shaders"]["fs"]["layout"]).fragment_status == STATUS
    image, coverage, report = render_artifact(artifact, width=16, height=16, depth_test=True)
    reference = softpipe_reference(shader, BUILD, tmp_path / "reference", mesh=mesh, depth_test=True)
    np.testing.assert_array_equal(image, reference)
    np.testing.assert_array_equal(image[:, :8], np.broadcast_to([0, 0, 255, 255], (16, 8, 4)))
    np.testing.assert_array_equal(image[:, 8:], np.broadcast_to([255, 0, 0, 0], (16, 8, 4)))
    assert np.all(coverage[:, :8] == 1)
    assert np.all(coverage[:, 8:] == (1 if depths[0] < 0 else 2))
    assert report["discarded_fragments"] == 128
    assert report["committed_fragments"] == int(coverage.sum())
    assert report["fragments"] == report["committed_fragments"] + report["discarded_fragments"]
    assert report["covered_pixels"] == 256 and report["depth"] == "less"
    assert report["blend"] == "disabled" and report["fpga_execution"] is False
