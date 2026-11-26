"""Explicit bounded FP32 instance files, real GLES extension draws and independent commits."""

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from tools import instancing_acceptance as acceptance
from tools import run_render
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
HARNESS = Path(os.environ.get("MGC_INSTANCING_TEST_HARNESS", ROOT / "tools/mgc_harness"))
HELPERS = (capture_draw, softpipe_reference, native_host_readback)


class InterceptedDraw(Exception):
    pass


def helper_options(helper):
    return {"socket_path": Path("/tmp/mgc-instancing-test.sock")} if helper is native_host_readback else {}


@pytest.mark.parametrize("helper", HELPERS)
def test_instance_data_is_keyword_only_explicit_path_with_disabled_default(helper):
    parameter = inspect.signature(helper).parameters["instance_data"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is None


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("invalid", (True, 1, "instances.bin", [], {}))
def test_instance_api_rejects_non_path_before_gl(monkeypatch, tmp_path, helper, invalid):
    monkeypatch.setattr(run_render.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid instancing reached GL"))
    with pytest.raises(ValueError, match="explicit Path"):
        helper(tmp_path, BUILD, tmp_path, instance_data=invalid, **helper_options(helper))


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_binary_contract_has_exact_divisor_records_and_finite_native_fp32(tmp_path, case):
    path = tmp_path / "instances.bin"
    path.write_bytes(acceptance.instance_bytes(case))
    options = run_render._validate_instance_data(path)
    assert options == {"instance_count": case.count, "offset_divisor": case.offset_divisor,
                       "color_divisor": case.color_divisor, "offset_components": case.offset_components,
                       "color_components": case.color_components, "start_instance": 0}
    assert path.stat().st_size <= run_render.INSTANCE_MAX_BYTES


@pytest.mark.parametrize("field,value", ((0, 0), (0, 17), (1, 0), (1, 1048577), (2, 0), (2, 1048577),
                                       (3, 2), (3, 5), (4, 0), (4, 5)))
def test_binary_invalid_count_divisor_or_vector_width_rejected(tmp_path, field, value):
    contents = bytearray(acceptance.instance_bytes(acceptance.CASES[1]))
    struct.pack_into("<I", contents, 8 + field * 4, value)
    path = tmp_path / "instances.bin"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        run_render._validate_instance_data(path)


@pytest.mark.parametrize("mutation", ("magic", "truncated", "trailing", "nan", "inf", "oversized"))
def test_exact_binary_size_magic_and_finite_values_are_required(tmp_path, mutation):
    contents = bytearray(acceptance.instance_bytes(acceptance.CASES[0]))
    if mutation == "magic":
        contents[0] = 0
    elif mutation == "truncated":
        contents.pop()
    elif mutation == "trailing":
        contents.append(0)
    elif mutation in ("nan", "inf"):
        struct.pack_into("<f", contents, run_render.INSTANCE_HEADER.size, float(mutation))
    else:
        contents.extend(bytes(run_render.INSTANCE_MAX_BYTES))
    path = tmp_path / "instances.bin"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        run_render._validate_instance_data(path)


def test_symlink_and_nonordinary_instance_input_are_rejected(tmp_path):
    target, link = tmp_path / "target", tmp_path / "link"
    target.write_bytes(acceptance.instance_bytes(acceptance.CASES[0]))
    link.symlink_to(target)
    with pytest.raises(OSError):
        run_render._validate_instance_data(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="ordinary bounded"):
        run_render._validate_instance_data(fifo)


@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("enabled", (False, True))
def test_helpers_clear_inherited_instancing_and_never_automatically_opt_in(monkeypatch, tmp_path, helper, enabled):
    for name in run_render.INSTANCING_ENV:
        monkeypatch.setenv(name, "inherited-invalid")
    monkeypatch.setenv("MGC_HOST_GLSL_PROFILE", "desktop-120")
    path = tmp_path / "instances.bin"
    path.write_bytes(acceptance.instance_bytes(acceptance.CASES[2]))
    def intercepted(command, *, env, **kwargs):
        assert env.get("MGC_HOST_INSTANCING") == ("1" if enabled else None)
        assert env.get("MGC_HOST_INSTANCE_DATA") == (str(path) if enabled else None)
        assert not any(name in env for name in run_render.INSTANCING_ENV[2:])
        assert "MGC_HOST_GLSL_PROFILE" not in env
        assert env["GALLIUM_DRIVER"] == ("softpipe" if helper is softpipe_reference else "mgc")
        raise InterceptedDraw
    monkeypatch.setattr(run_render.subprocess, "run", intercepted)
    with pytest.raises(InterceptedDraw):
        helper(tmp_path, BUILD, tmp_path, instance_data=path if enabled else None, **helper_options(helper))


def test_hardware_instancing_and_saved_capture_cli_override_are_rejected(tmp_path, capsys):
    with pytest.raises(ValueError, match="instancing is unsupported on hardware"):
        capture_draw(tmp_path, BUILD, tmp_path, "fp16-hardware-v1", instance_data=tmp_path / "absent.bin")
    assert run_render.main(["--capture", str(tmp_path / "absent.json"), "--instance-data", str(tmp_path / "instances.bin")]) == 1
    assert "require a fresh shader draw" in capsys.readouterr().err


def harness_environment(tmp_path, driver="softpipe"):
    if not HARNESS.is_file() or not (BUILD / "src/egl/libEGL.so").is_file():
        pytest.skip("rebuild Mesa bounded instancing and the GL harness first")
    environment = run_render.mesa_environment(BUILD, tmp_path)
    for name in tuple(environment):
        if name.startswith("MGC_HOST_") or name == "MGC_FRAGMENT_STATUS":
            environment.pop(name, None)
    environment.update(GALLIUM_DRIVER=driver, MESA_GLES_VERSION_OVERRIDE="2.0")
    return environment


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_real_gles2_softpipe_all_divisor_rgba_oracles(monkeypatch, tmp_path, case):
    harness_environment(tmp_path)
    monkeypatch.setenv("MESA_GLES_VERSION_OVERRIDE", "2.0")
    monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
    shader = acceptance._shader(case, tmp_path)
    mesh, instances = tmp_path / "vertices.bin", tmp_path / "instances.bin"
    mesh.write_bytes(acceptance.case_mesh())
    instances.write_bytes(acceptance.instance_bytes(case))
    image = softpipe_reference(shader, BUILD, tmp_path / "reference", mesh=mesh, instance_data=instances,
                               depth_test=case.depth_test, blend=case.blend, repeats=case.repeats)
    images, coverages, depths, discarded = acceptance.rational_oracle(case)
    assert acceptance.image_comparison(image, images[-1], 1 if case.blend != "disabled" else 0)["passed"]
    log = (tmp_path / "reference/softpipe.log").read_text()
    assert "MGC_HOST_INSTANCING_GL alias=EXT" in log and "base_instance=0" in log
    assert np.count_nonzero(coverages[0]) > 0


@pytest.mark.parametrize("mutation", ("no-opt-in", "no-path", "zero-flag", "short-file", "oversized-budget"))
def test_real_harness_fails_closed_before_draw_on_invalid_instancing(tmp_path, mutation):
    case = acceptance.CASES[1]
    shader = acceptance._shader(case, tmp_path)
    instances, mesh = tmp_path / "instances.bin", tmp_path / "vertices.bin"
    instances.write_bytes(acceptance.instance_bytes(case))
    mesh.write_bytes(acceptance.case_mesh())
    environment = harness_environment(tmp_path)
    environment.update(MGC_HOST_INSTANCING="1", MGC_HOST_INSTANCE_DATA=str(instances), MGC_HOST_MESH=str(mesh))
    if mutation == "no-opt-in":
        environment.pop("MGC_HOST_INSTANCING")
    elif mutation == "no-path":
        environment.pop("MGC_HOST_INSTANCE_DATA")
    elif mutation == "zero-flag":
        environment["MGC_HOST_INSTANCING"] = "0"
    elif mutation == "short-file":
        instances.write_bytes(b"MGCINS01")
    else:
        mesh.write_bytes(acceptance.case_mesh() * 200)
    raw = tmp_path / "rejected.rgba8"
    completed = subprocess.run([str(HARNESS), str(shader / "vert.glsl"), str(shader / "frag.glsl"), "--reference", str(raw)],
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0 and not raw.exists(), completed.stdout + completed.stderr
    assert "instanc" in completed.stderr.lower()


def test_zero_alpha_commits_and_depth_kill_repeats_are_not_rgba_coverage_proxies():
    case = next(case for case in acceptance.CASES if case.name == "zero-alpha-count4")
    images, coverages, depths, discarded = acceptance.rational_oracle(case)
    assert np.count_nonzero(images[0][:, :, 3]) == 0 and coverages[0].sum() == 64
    case = next(case for case in acceptance.CASES if case.name == "depth-kill-repeat")
    images, coverages, depths, discarded = acceptance.rational_oracle(case)
    assert coverages[0].sum() == 16 and coverages[1].sum() == 0
    assert np.count_nonzero(depths[-1] == 16384) == 8 and discarded == [40, 40]


def test_indexed_and_nonzero_base_are_only_independent_python_fetch_evidence():
    from test_host_instancing import instanced_fetch_fixture
    from tools.host_render import fetch_vertices
    for indexed in (False, True):
        artifact, layout = instanced_fetch_fixture(instances=5, base=2, divisor=3, indexed=indexed)
        values = fetch_vertices(artifact, layout)
        assert len(values) == 15 and artifact["draw"]["start_instance"] == 2


def contract_fixture(case):
    from tools.run_shader import ABI_BASES
    offsets, colors = acceptance.instance_records(case)
    common = {"version": 3, "supported": True, "program_format": "legacy-mgcsim-v1",
              "shader_abi_version": 3, "address_units": "words", "bases": dict(ABI_BASES),
              "inputs": [], "outputs": [], "uniforms": []}
    inputs = [{"word_offset": slot * 4 + component, "location": location, "component": component,
               "type": "float", "bit_size": 32, "interpolation": "smooth"}
              for slot, location, components in ((0, 15, 4), (1, 18, case.offset_components),
                                                 (2, 19, case.color_components))
              for component in range(components)]
    buffers = [{"stride": 40, "data_hex": bytes(240).hex()}] + [
        {"stride": components * 4, "data_hex": b"".join(struct.pack("<" + "f" * len(record), *record)
                                                       for record in records).hex()}
        for components, records in ((case.offset_components, offsets), (case.color_components, colors))]
    elements = [{"src_offset": 0, "vertex_buffer_index": slot, "instance_divisor": divisor}
                for slot, divisor in enumerate((0, case.offset_divisor, case.color_divisor))]
    return {"schema": "mgc-host-draw-v3", "host_instancing": {**acceptance.HOST_INSTANCING_CONTRACT,
                "shader_abi_version": 3, "instance_id_word": 65535},
                "program_format": "legacy-mgcsim-v1", "shader_abi_version": 3,
                "draw": {"instance_count": case.count, "start_instance": 0, "count": 6, "index_size": 0},
                "indices": None, "vertex_buffers": buffers, "vertex_elements": elements,
                "shaders": {"vs": {"layout": {**common, "stage": "vertex", "inputs": inputs,
                    "host_instancing": acceptance.shader_instancing_contract(case, "vertex")}},
                            "fs": {"layout": {**common, "stage": "fragment", "fragment_status": dict(acceptance.STATUS),
                                               "host_instancing": acceptance.shader_instancing_contract(case, "fragment"),
                                               "uses_discard": case.kill}}}}


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_contract_uses_actual_dense_driver_slots_not_gl_binding_numbers(case):
    artifact = contract_fixture(case)
    acceptance.check_contract(artifact, case)
    artifact["indices"] = "00"
    with pytest.raises(ValueError, match="direct-draw"):
        acceptance.check_contract(artifact, case)


@pytest.mark.parametrize("mutation", ("top-abi1", "top-word0", "top-missing-word", "vs-layout1", "fs-layout1",
    "vs-abi1", "fs-abi1", "vs-missing-metadata", "fs-missing-metadata", "vs-unused-id",
    "fs-uses-id", "vs-int-bool", "vs-extra-field", "vs-word0"))
def test_abi3_contract_rejects_old_abi_or_incorrect_instance_id_metadata(mutation):
    case = next(case for case in acceptance.CASES if case.name == "instance-id1-count4")
    artifact = contract_fixture(case)
    acceptance.check_contract(artifact, case)
    vertex, fragment = (artifact["shaders"][stage]["layout"] for stage in ("vs", "fs"))
    if mutation == "top-abi1":
        artifact["shader_abi_version"] = 1
    elif mutation == "top-word0":
        artifact["host_instancing"]["instance_id_word"] = 0
    elif mutation == "top-missing-word":
        artifact["host_instancing"].pop("instance_id_word")
    elif mutation.endswith("layout1"):
        (vertex if mutation.startswith("vs") else fragment)["version"] = 1
    elif mutation.endswith("abi1"):
        (vertex if mutation.startswith("vs") else fragment)["shader_abi_version"] = 1
    elif mutation.endswith("missing-metadata"):
        (vertex if mutation.startswith("vs") else fragment).pop("host_instancing")
    elif mutation == "vs-unused-id":
        vertex["host_instancing"]["uses_instance_id"] = False
    elif mutation == "fs-uses-id":
        fragment["host_instancing"]["uses_instance_id"] = True
    elif mutation == "vs-int-bool":
        vertex["host_instancing"]["uses_instance_id"] = 1
    elif mutation == "vs-extra-field":
        vertex["host_instancing"]["unused"] = 0
    else:
        vertex["host_instancing"]["instance_id_word"] = 0
    with pytest.raises(ValueError):
        acceptance.check_contract(artifact, case)


@pytest.mark.parametrize("case", acceptance.CASES, ids=lambda case: case.name)
def test_fresh_gles100_sources_use_only_explicit_integer_instance_id_branches(tmp_path, case):
    shader = acceptance._shader(case, tmp_path)
    vertex, fragment = ((shader / f"{stage}.glsl").read_text() for stage in ("vert", "frag"))
    assert vertex.startswith("#version 100\n") and fragment.startswith("#version 100\n")
    assert "gl_Position = a_position + vec4(a_instance_offset.xyz, 0.0);" in vertex
    assert "a_instance_color" in vertex and "gl_InstanceID" not in fragment
    if case.instance_id_compare is None:
        assert "#extension" not in vertex and "gl_InstanceID" not in vertex
    else:
        assert "#extension GL_EXT_draw_instanced : require\n" in vertex
        assert f"if (gl_InstanceIDEXT == {case.instance_id_compare})" in vertex
        assert "float(gl_InstanceID" not in vertex
    assert acceptance.shader_instancing_contract(case, "fragment")["uses_instance_id"] is False


@pytest.mark.parametrize("case", tuple(case for case in acceptance.CASES if case.instance_id_compare is not None),
                         ids=lambda case: case.name)
def test_independent_instance_id_branch_oracle_distinguishes_always_zero_and_attribute_divisors(case):
    actual_images, coverages, depths, discarded = acceptance.rational_oracle(case)
    always_zero = acceptance.rational_oracle(case, force_instance_id=0)[0][-1]
    no_branch = acceptance.rational_oracle(replace(case, instance_id_compare=None))[0][-1]
    assert not acceptance.image_comparison(actual_images[-1], always_zero, 0)["passed"]
    assert not acceptance.image_comparison(actual_images[-1], no_branch, 0)["passed"]
    assert int(coverages[0].sum()) == case.count * 16
    alternate = np.array([round(channel * 255) for channel in acceptance.INSTANCE_ID_COLOR], np.uint8)
    assert np.count_nonzero(np.all(actual_images[-1] == alternate, axis=2)) == 16


@pytest.mark.parametrize("invalid", (True, -1, 16, 1.0, "1"))
def test_instance_id_oracle_controls_are_strict_local_integers(invalid):
    with pytest.raises(ValueError, match="local integer"):
        acceptance.rational_oracle(acceptance.CASES[-1], force_instance_id=invalid)


def test_matrix_preserves_attribute_cases_and_adds_real_first_middle_last_instance_ids():
    assert len(acceptance.CASES) == 19 and len(acceptance.DEFAULT_CASES) == 14
    assert sum(case.instance_id_compare is None for case in acceptance.CASES) == 15
    builtin = [case for case in acceptance.CASES if case.instance_id_compare is not None]
    assert {case.instance_id_compare for case in builtin} == {0, 1, 15}
    assert {case.instance_id_compare for case in builtin if case.count == 16} == {0, 15}
    assert any(case.offset_divisor == 2 and case.color_divisor == 3 for case in builtin)


@pytest.fixture(scope="module")
def fresh_instancing(tmp_path_factory):
    output = tmp_path_factory.mktemp("fresh-instancing")
    harness_environment(output, "mgc")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("MESA_GLES_VERSION_OVERRIDE", "2.0")
        monkeypatch.setattr(run_render, "ROOT", HARNESS.parent.parent)
        return output, acceptance.validate(output, BUILD)


def test_fresh_actual_gl_capture_managed_native_independent_softpipe(fresh_instancing):
    output, summary = fresh_instancing
    assert summary["passed"] and summary["native_gl_readback"] and summary["fpga_execution"] is False
    assert summary["tested_cases"] == [case.name for case in acceptance.DEFAULT_CASES]
    for result in summary["cases"]:
        assert result["comparisons"]["native_vs_managed"]["max_channel_error"] == 0
        assert result["comparisons"]["native_vs_rational"]["max_channel_error"] == 0
        assert result["start_instance"] == 0
        for name in ("managed", "native", "softpipe", "rational"):
            assert (output / result["case"] / f"{name}.rgba8").stat().st_size == 1024


@pytest.mark.parametrize("cases", ((), (acceptance.CASES[0], acceptance.CASES[0]),
                                  (replace(acceptance.CASES[0], name="unknown"),)))
def test_invalid_case_selection_never_reaches_gl(tmp_path, cases):
    with pytest.raises(ValueError, match="distinct named bounded"):
        acceptance.validate(tmp_path, BUILD, cases=cases)


def test_acceptance_persists_failure_without_claiming_native_evidence(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("missing advertised instancing extension")
    monkeypatch.setattr(acceptance, "_run_case", fail)
    with pytest.raises(ValueError, match="advertised instancing"):
        acceptance.validate(tmp_path, BUILD, cases=(acceptance.CASES[0],))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] is False and summary["native_gl_readback"] is False and summary["status"] == "error"
