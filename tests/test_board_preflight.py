"""Offline preflight tests; fixture addresses and references are wholly synthetic."""

import builtins
from copy import deepcopy
from contextlib import contextmanager
import importlib
import json
import mmap
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.mgcdriver import Driver, MMIOBackend, MMIOConfig, MMIOTransport
from tools.mgcdriver import preflight

ROOT = Path(__file__).resolve().parents[1]


def synthetic_manifest():
    maps = [
        {"name": "synthetic_data", "index": 0, "size": 4096, "offset": 0,
         "page_size": 4096, "mapping_ref": "synthetic-not-board-evidence:map-data"},
        {"name": "synthetic_program", "index": 1, "size": 8192, "offset": 0,
         "page_size": 4096, "mapping_ref": "synthetic-not-board-evidence:map-program"},
        {"name": "synthetic_registers", "index": 2, "size": 4096, "offset": 128,
         "page_size": 4096, "mapping_ref": "synthetic-not-board-evidence:map-registers"},
    ]
    windows = {}
    for role, name, size, gpu_base in (
        ("data", "synthetic_data", 256, 4096),
        ("program", "synthetic_program", 256, 8192),
        ("registers", "synthetic_registers", 64, None),
    ):
        windows[role] = {
            "map": name, "byte_offset": 64, "size": size,
            "reservation": {"byte_offset": 32, "size": 512,
                            "reference": "synthetic-not-board-evidence:reserved-" + role},
        }
        if gpu_base is not None:
            windows[role]["gpu_byte_base"] = gpu_base
    contracts = {name: "synthetic-not-board-evidence:" + name for name in preflight.CONTRACT_REFS}
    contracts["stop_policy"] = "quarantine"
    return {
        "manifest_version": 1,
        "device": {
            "identity": "SYNTHETIC TEST ONLY: no board or bitstream",
            "identity_ref": "synthetic-not-board-evidence:identity",
            "access": "synthetic:/dev/uio-not-a-real-node",
            "access_ref": "synthetic-not-board-evidence:access",
            "ownership_ref": "synthetic-not-board-evidence:ownership",
        },
        "maps": maps,
        "windows": windows,
        "mmio": {
            "memory_base": 4096, "memory_size": 256,
            "program_base": 8192, "program_capacity": 256,
            "control_offset": 0, "status_offset": 4, "entry_offset": 8,
            "word_count_offset": 12, "start_value": 1,
            "busy_mask": 1, "done_mask": 2, "error_mask": 4,
            "acknowledge_offset": 16, "acknowledge_value": 7,
            "abort_offset": None, "abort_value": None, "byteorder": "little",
        },
        "abi": dict(preflight.ABI),
        "submission": {"program_word_count": 8, "entry_word": 2},
        "contracts": contracts,
    }


def changed(path, value):
    manifest = synthetic_manifest()
    parent = manifest
    for member in path[:-1]:
        parent = parent[member]
    parent[path[-1]] = value
    return manifest


def leaf_paths(value, prefix=()):
    if type(value) is dict:
        for name, member in value.items():
            yield from leaf_paths(member, prefix + (name,))
    elif type(value) is list:
        for position, member in enumerate(value):
            yield from leaf_paths(member, prefix + (position,))
    else:
        yield prefix, value


LEAVES = list(leaf_paths(synthetic_manifest()))
REQUIRED_LEAVES = [path for path, value in LEAVES if value is not None]
INTEGER_LEAVES = [path for path, value in LEAVES if type(value) is int]


def assert_report(report, status):
    assert report["status"] == status
    assert report["hardware_execution"] is False
    assert report["readiness_unverified"] is True
    assert report["references_verified"] is False
    assert (report["plan"] is not None) == (status == "structurally-consistent")
    if status != "structurally-consistent":
        assert report["findings"]


def run_cli(*arguments, env=None):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, "-B", "-m", "tools.mgcdriver.preflight", *map(str, arguments)],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=10,
    )


def write_manifest(tmp_path, manifest=None):
    source = tmp_path / "synthetic.json"
    source.write_text(json.dumps(synthetic_manifest() if manifest is None else manifest), encoding="utf-8")
    return source


def test_synthetic_plan_uses_supplied_page_sizes_offsets_and_word_units():
    manifest = synthetic_manifest()
    before = deepcopy(manifest)
    report = preflight.assess_manifest(manifest)
    assert_report(report, "structurally-consistent")
    assert manifest == before
    plan = report["plan"]
    assert [descriptor["mmap_selection_offset"] for descriptor in plan["maps"]] == [0, 4096, 8192]
    assert plan["windows"]["registers"]["within_map_start_offset"] == 192
    assert plan["windows"]["registers"]["register_within_map_offsets"]["acknowledge_offset"] == 208
    assert plan["windows"]["data"]["gpu_byte_base"] == 4096
    assert plan["windows"]["data"]["within_map_start_offset"] == 64
    assert plan["submission"] == {
        "program_word_count": 8, "program_size_bytes": 32,
        "entry_word": 2, "entry_register_value": 8, "word_count_register_value": 8,
    }
    assert MMIOConfig(**plan["mmio"]).program_base == 8192


def test_supplied_gpu_translation_can_cover_a_subwindow_without_physical_inference():
    manifest = synthetic_manifest()
    manifest["windows"]["data"].update(gpu_byte_base=4032, size=320)
    report = preflight.assess_manifest(manifest)
    assert_report(report, "structurally-consistent")
    assert report["plan"]["windows"]["data"]["within_map_start_offset"] == 128
    assert report["plan"]["windows"]["data"]["gpu_byte_base"] == 4096


def test_plan_does_not_retain_mutable_manifest_objects():
    manifest = synthetic_manifest()
    report = preflight.assess_manifest(manifest)
    manifest["maps"][0]["size"] = 0
    manifest["contracts"]["coherency_ref"] = "changed"
    assert report["plan"]["maps"][0]["size"] == 4096
    assert report["plan"]["contracts"]["coherency_ref"] != "changed"


@pytest.mark.parametrize("path", REQUIRED_LEAVES)
def test_unknown_required_leaf_is_missing_and_never_emits_plan(path):
    assert_report(preflight.assess_manifest(changed(path, None)), "missing")


@pytest.mark.parametrize("name", list(synthetic_manifest()))
def test_missing_top_level_field_is_reported(name):
    manifest = synthetic_manifest()
    del manifest[name]
    assert_report(preflight.assess_manifest(manifest), "missing")


@pytest.mark.parametrize("name", ["abort_offset", "abort_value", "byteorder"])
def test_mmio_defaults_are_not_silently_supplied(name):
    manifest = synthetic_manifest()
    del manifest["mmio"][name]
    assert_report(preflight.assess_manifest(manifest), "missing")


@pytest.mark.parametrize("path", INTEGER_LEAVES)
@pytest.mark.parametrize("value", [True, False, "1", 1.0])
def test_numeric_fields_reject_booleans_strings_and_floats(path, value):
    assert_report(preflight.assess_manifest(changed(path, value)), "incompatible")


@pytest.mark.parametrize("path", [
    (), ("device",), ("maps", 0), ("windows",), ("windows", "data"),
    ("windows", "data", "reservation"), ("windows", "program"),
    ("windows", "registers"), ("mmio",), ("abi",), ("submission",), ("contracts",),
])
def test_unknown_fields_are_rejected_at_every_schema_level(path):
    manifest = synthetic_manifest()
    parent = manifest
    for member in path:
        parent = parent[member]
    parent["unknown"] = None
    report = preflight.assess_manifest(manifest)
    assert_report(report, "incompatible")
    assert any(finding["message"] == "unknown field" for finding in report["findings"])


@pytest.mark.parametrize("path,value", [
    (("manifest_version",), 2), (("device",), []), (("device", "identity"), ""),
    (("device", "access_ref"), " reference "), (("contracts", "barrier_ref"), "bad\nreference"),
    (("maps",), []), (("maps",), {}), (("maps", 0, "name"), "invalid name"),
    (("maps", 0, "index"), -1), (("maps", 0, "size"), 4095),
    (("maps", 0, "offset"), 4096), (("maps", 0, "page_size"), 6),
    (("maps", 0, "page_size"), 2), (("maps", 0, "page_size"), 1 << 21),
    (("maps", 0, "index"), preflight.MAX_FILE_OFFSET),
    (("maps", 1, "index"), 0), (("maps", 1, "name"), "synthetic_data"),
    (("maps", 1, "page_size"), 8192), (("windows", "data", "map"), "unknown_map"),
    (("windows", "data", "byte_offset"), 4096), (("windows", "data", "byte_offset"), 1),
    (("windows", "data", "size"), 255), (("windows", "data", "size"), 4096),
    (("windows", "data", "gpu_byte_base"), 8192),
    (("windows", "data", "gpu_byte_base"), (1 << 32) - 4),
    (("windows", "data", "reservation", "byte_offset"), 65),
    (("windows", "data", "reservation", "size"), 64),
    (("windows", "data", "reservation", "size"), 8192),
    (("windows", "registers", "size"), 16),
    (("mmio", "memory_base"), 8192), (("mmio", "memory_base"), (1 << 32) - 4),
    (("mmio", "program_capacity"), 255), (("mmio", "control_offset"), 2),
    (("mmio", "word_count_offset"), 8), (("mmio", "start_value"), 0),
    (("mmio", "busy_mask"), 0), (("mmio", "done_mask"), 1),
    (("mmio", "error_mask"), 2), (("mmio", "status_offset"), 1 << 32),
    (("mmio", "abort_value"), 99), (("mmio", "byteorder"), "big"),
    (("abi", "program_format"), "legacy-mgcsim-v1"), (("abi", "program_word_bits"), 16),
    (("abi", "register_width_bits"), 64), (("abi", "register_byteorder"), "big"),
    (("abi", "entry_units"), "bytes"), (("abi", "entry_register_units"), "instruction-words"),
    (("abi", "count_register_units"), "bytes"), (("contracts", "stop_policy"), "ignore"),
    (("submission", "program_word_count"), 0), (("submission", "program_word_count"), 65),
    (("submission", "program_word_count"), preflight.MAX_PROGRAM_WORDS + 1),
    (("submission", "entry_word"), 8), (("submission", "entry_word"), -1),
])
def test_incompatible_geometry_registers_and_abi_never_emit_a_plan(path, value):
    assert_report(preflight.assess_manifest(changed(path, value)), "incompatible")


def test_register_aliases_allowed_by_existing_config_are_not_invented_restrictions():
    manifest = synthetic_manifest()
    manifest["mmio"].update(acknowledge_offset=4, error_mask=0)
    assert_report(preflight.assess_manifest(manifest), "structurally-consistent")


def test_mapped_windows_must_not_alias_even_with_distinct_gpu_bases():
    manifest = synthetic_manifest()
    manifest["windows"]["program"]["map"] = "synthetic_data"
    assert_report(preflight.assess_manifest(manifest), "incompatible")


def test_one_map_can_cover_nonoverlapping_windows_and_a_shared_reservation():
    manifest = synthetic_manifest()
    manifest["maps"] = manifest["maps"][:1]
    for position, role in enumerate(("data", "program", "registers")):
        manifest["windows"][role]["map"] = "synthetic_data"
        manifest["windows"][role]["byte_offset"] = position * 512
        manifest["windows"][role]["reservation"].update(byte_offset=0, size=2048)
    assert_report(preflight.assess_manifest(manifest), "structurally-consistent")


def test_abort_policy_requires_configured_abort_and_complete_contract():
    manifest = synthetic_manifest()
    manifest["contracts"]["stop_policy"] = "abort"
    assert_report(preflight.assess_manifest(manifest), "missing")
    manifest["mmio"].update(abort_offset=20, abort_value=99)
    assert_report(preflight.assess_manifest(manifest), "structurally-consistent")
    manifest["contracts"]["stop_ref"] = None
    assert_report(preflight.assess_manifest(manifest), "missing")


def test_quarantine_policy_rejects_implicit_abort_attempts():
    manifest = synthetic_manifest()
    manifest["mmio"].update(abort_offset=20, abort_value=99)
    assert_report(preflight.assess_manifest(manifest), "incompatible")


def test_all_protocol_methods_require_opaque_contract_references():
    methods = {name for name, member in vars(MMIOTransport).items()
               if not name.startswith("_") and callable(member)}
    assert methods == set(preflight.TRANSPORT_METHODS)
    for method in methods:
        manifest = synthetic_manifest()
        del manifest["contracts"][method + "_ref"]
        assert_report(preflight.assess_manifest(manifest), "missing")


def test_mmio_config_is_the_source_of_register_validation(monkeypatch):
    observed = []
    original = preflight.MMIOConfig

    def validate(**configuration):
        observed.append(configuration)
        return original(**configuration)

    monkeypatch.setattr(preflight, "MMIOConfig", validate)
    monkeypatch.setattr(preflight, "fields", lambda unused: importlib.import_module("dataclasses").fields(original))
    assert_report(preflight.assess_manifest(synthetic_manifest()), "structurally-consistent")
    assert observed == [synthetic_manifest()["mmio"]]


@pytest.mark.parametrize("data", [
    b"", b"{} trailing", b"{\"manifest_version\": 0x1}", b"// comment\n{}", b"{\"value\": 01}",
    b"{\"value\": NaN}", b"{\"value\": Infinity}", b"{\"value\": -Infinity}",
    b"{\"value\": 1e999}", b"{\"value\": 1.0}", b"\xff", b"\xef\xbb\xbf{}",
    b"{\"manifest_version\": 1, \"manifest_version\": 1}",
    b"{\"mmio\": {\"status_offset\": 4, \"status_offset\": 8}}",
    b"{\"maps\": [{\"index\": 0, \"index\": 0}]}",
    b"[" * 17 + b"0" + b"]" * 17,
    b"{\"value\": " + b"9" * 100 + b"}",
    b" " * (preflight.MAX_JSON_BYTES + 1),
    " " * (preflight.MAX_JSON_BYTES + 1),
])
def test_strict_bounded_json_returns_complete_failure_reports(data):
    assert_report(preflight.assess_json(data), "incompatible")


@pytest.mark.parametrize("manifest", [True, False, 1, "not an object", [], {1: "wrong key"},
                                            {"value": float("nan")}, {"value": object()},
                                            {"value": "x" * 513}, {"value": "\ud800"}])
def test_api_rejects_non_manifest_or_non_json_values(manifest):
    assert_report(preflight.assess_manifest(manifest), "incompatible")


def test_api_rejects_cycles_and_bounded_tree_overflow():
    cyclic = {}
    cyclic["self"] = cyclic
    assert_report(preflight.assess_manifest(cyclic), "incompatible")
    assert_report(preflight.assess_manifest([None] * (preflight.MAX_NODES + 1)), "incompatible")


def test_brackets_and_escapes_in_strings_do_not_count_as_nesting():
    manifest = synthetic_manifest()
    manifest["device"]["identity"] = 'synthetic [ { \\" ] }' * 20
    assert_report(preflight.assess_json(json.dumps(manifest)), "structurally-consistent")


def test_findings_are_bounded_without_hiding_incompatible_status():
    manifest = synthetic_manifest()
    manifest["device"] = {"unknown" + str(position): None for position in range(200)}
    report = preflight.assess_manifest(manifest)
    assert_report(report, "incompatible")
    assert len(report["findings"]) == preflight.MAX_FINDINGS
    assessment = preflight._Assessment()
    for position in range(preflight.MAX_FINDINGS):
        assessment.issue("missing", str(position), "missing")
    assessment.issue("incompatible", "$", "incompatible")
    assert_report(preflight._report(assessment.findings), "incompatible")


def test_api_uses_no_driver_transport_files_mmap_socket_or_metadata_reads(monkeypatch):
    manifest = synthetic_manifest()
    data = json.dumps(manifest)

    def forbidden(*arguments, **keywords):
        raise AssertionError("offline API attempted runtime access")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(os, "open", forbidden)
        guard.setattr(os, "stat", forbidden)
        guard.setattr(mmap, "mmap", forbidden)
        guard.setattr(socket, "socket", forbidden)
        guard.setattr(Driver, "__init__", forbidden)
        guard.setattr(Driver, "allocate", forbidden)
        guard.setattr(MMIOBackend, "__init__", forbidden)
        for method in preflight.TRANSPORT_METHODS:
            guard.setattr(MMIOTransport, method, forbidden)
        assert_report(preflight.assess_manifest(manifest), "structurally-consistent")
        assert_report(preflight.assess_json(data), "structurally-consistent")


def test_cli_writes_complete_new_regular_report_and_returns_zero(tmp_path):
    source = write_manifest(tmp_path)
    destination = tmp_path / "report.json"
    completed = run_cli(source, "--output", destination)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == completed.stderr == ""
    assert_report(json.loads(destination.read_text()), "structurally-consistent")
    assert destination.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".board-preflight-*.tmp"))


def test_cli_stdout_is_a_report_not_hardware_execution(tmp_path):
    completed = run_cli(write_manifest(tmp_path))
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert_report(json.loads(completed.stdout), "structurally-consistent")


@pytest.mark.parametrize("manifest,status", [
    ({"manifest_version": 1}, "missing"),
    (changed(("abi", "entry_units"), "bytes"), "incompatible"),
])
def test_cli_missing_or_incompatible_manifests_emit_full_findings_and_exit_one(tmp_path, manifest, status):
    destination = tmp_path / "failure-report.json"
    completed = run_cli(write_manifest(tmp_path, manifest), "--output", destination)
    assert completed.returncode == 1
    assert_report(json.loads(destination.read_text()), status)
    assert not list(tmp_path.glob(".board-preflight-*.tmp"))


def test_cli_invalid_json_emits_failure_report_and_preserves_source(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_text('{"version": NaN}')
    destination = tmp_path / "failure.json"
    completed = run_cli(source, "--output", destination)
    assert completed.returncode == 1
    assert source.read_text() == '{"version": NaN}'
    assert_report(json.loads(destination.read_text()), "incompatible")


@pytest.mark.parametrize("kind", ["ordinary", "symlink", "dangling-symlink", "directory", "input"])
def test_cli_never_overwrites_existing_outputs_or_follows_output_symlinks(tmp_path, kind):
    source = write_manifest(tmp_path)
    destination = tmp_path / "report.json"
    target = tmp_path / "user-file"
    target.write_bytes(b"preserve user bytes")
    if kind == "ordinary":
        destination.write_bytes(b"existing report")
    elif kind == "symlink":
        destination.symlink_to(target)
    elif kind == "dangling-symlink":
        destination.symlink_to(tmp_path / "missing-user-file")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination = source
    before = source.read_bytes()
    completed = run_cli(source, "--output", destination)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert_report(json.loads(completed.stderr), "incompatible")
    assert source.read_bytes() == before
    assert target.read_bytes() == b"preserve user bytes"
    if kind == "ordinary":
        assert destination.read_bytes() == b"existing report"
    if "symlink" in kind:
        assert destination.is_symlink()
    assert not (tmp_path / "missing-user-file").exists()
    assert not list(tmp_path.glob(".board-preflight-*.tmp"))


@pytest.mark.parametrize("which", ["input", "output"])
def test_cli_rejects_symlinked_parent_directories(tmp_path, which):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    source = write_manifest(actual if which == "input" else tmp_path)
    destination = actual / "report.json"
    completed = run_cli(alias / source.name if which == "input" else source,
                        "--output", alias / destination.name if which == "output" else destination)
    assert completed.returncode == (1 if which == "input" else 2)
    if which == "input":
        assert_report(json.loads(destination.read_text()), "incompatible")
    else:
        assert not destination.exists()


def test_cli_rejects_symlink_input_without_reading_target(tmp_path):
    source = write_manifest(tmp_path)
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    completed = run_cli(alias)
    assert completed.returncode == 1
    assert_report(json.loads(completed.stdout), "incompatible")


@pytest.mark.parametrize("path", ["/dev/uio0", "/dev/mem", "/sys/class/uio/uio0/maps/map0/size", "/proc/self/mem"])
def test_special_namespaces_are_rejected_before_any_filesystem_call(monkeypatch, path):
    def forbidden(*arguments, **keywords):
        raise AssertionError("special path attempted filesystem access")

    with monkeypatch.context() as guard:
        guard.setattr(os, "open", forbidden)
        guard.setattr(os, "stat", forbidden)
        with pytest.raises(preflight.ValidationError, match="forbidden"):
            preflight._read_ordinary_json(path)
        with pytest.raises(preflight.ValidationError, match="forbidden"):
            preflight._write_atomic_report(path, preflight.assess_manifest(synthetic_manifest()))


def test_nonregular_input_is_rejected_before_open_without_creating_a_device(monkeypatch):
    @contextmanager
    def parent(unused):
        yield 123, "synthetic-nonregular"

    def forbidden(*arguments, **keywords):
        raise AssertionError("nonregular input was opened")

    monkeypatch.setattr(preflight, "_ordinary_parent", parent)
    monkeypatch.setattr(os, "stat", lambda *arguments, **keywords: SimpleNamespace(st_mode=0o010600))
    monkeypatch.setattr(os, "open", forbidden)
    with pytest.raises(preflight.ValidationError, match="regular"):
        preflight._read_ordinary_json("synthetic")


def test_oversized_input_is_rejected_before_leaf_open(tmp_path, monkeypatch):
    source = tmp_path / "large.json"
    source.write_bytes(b" " * (preflight.MAX_JSON_BYTES + 1))
    opened = []
    original = os.open

    def tracked(path, *arguments, **keywords):
        opened.append(path)
        return original(path, *arguments, **keywords)

    monkeypatch.setattr(os, "open", tracked)
    with pytest.raises(preflight.ValidationError, match="byte limit"):
        preflight._read_ordinary_json(source)
    assert source.name not in opened


def test_input_growth_is_still_bounded_after_size_check(tmp_path, monkeypatch):
    source = write_manifest(tmp_path)
    consumed = []

    def growing(descriptor, count):
        consumed.append(count)
        return b"x" * count

    monkeypatch.setattr(os, "read", growing)
    with pytest.raises(preflight.ValidationError, match="byte limit"):
        preflight._read_ordinary_json(source)
    assert consumed == [preflight.MAX_JSON_BYTES + 1]


@pytest.mark.parametrize("failure", ["partial-write", "zero-write", "fsync", "publish"])
def test_atomic_write_failure_leaves_no_partial_output_or_temporary(tmp_path, monkeypatch, failure):
    destination = tmp_path / "report.json"
    original_write = os.write

    def partial(descriptor, data):
        original_write(descriptor, data[:8])
        raise OSError("synthetic partial failure")

    def failed(*arguments, **keywords):
        raise OSError("synthetic report failure")

    if failure == "partial-write":
        monkeypatch.setattr(os, "write", partial)
    elif failure == "zero-write":
        monkeypatch.setattr(os, "write", lambda *arguments: 0)
    else:
        monkeypatch.setattr(os, "fsync" if failure == "fsync" else "link", failed)
    with pytest.raises(OSError):
        preflight._write_atomic_report(destination, preflight.assess_manifest(synthetic_manifest()))
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publication_does_not_replace_a_racing_user_file(tmp_path, monkeypatch):
    destination = tmp_path / "report.json"
    original = os.link

    def race(*arguments, **keywords):
        destination.write_bytes(b"user won publication race")
        return original(*arguments, **keywords)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(FileExistsError):
        preflight._write_atomic_report(destination, preflight.assess_manifest(synthetic_manifest()))
    assert destination.read_bytes() == b"user won publication race"
    assert not list(tmp_path.glob(".board-preflight-*.tmp"))


def test_atomic_writer_handles_short_writes(tmp_path, monkeypatch):
    destination = tmp_path / "report.json"
    original = os.write
    monkeypatch.setattr(os, "write", lambda descriptor, data: original(descriptor, data[:7]))
    preflight._write_atomic_report(destination, preflight.assess_manifest(synthetic_manifest()))
    assert_report(json.loads(destination.read_text()), "structurally-consistent")


def test_cli_import_and_execution_have_audited_no_runtime_hardware_access(tmp_path):
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import os, sys\n"
        "def audit(event, args):\n"
        "    if event.startswith('socket.') or event == 'mmap.__new__':\n"
        "        raise RuntimeError('runtime transport access forbidden')\n"
        "    if event == 'open' and isinstance(args[0], (str, bytes)):\n"
        "        path = os.fsdecode(args[0])\n"
        "        if any(path == root or path.startswith(root + '/') for root in ('/dev', '/sys', '/proc')):\n"
        "            raise RuntimeError('special filesystem access forbidden')\n"
        "def profile(frame, event, arg):\n"
        "    if event == 'call':\n"
        "        filename, name = frame.f_code.co_filename, frame.f_code.co_name\n"
        "        method = 'self' in frame.f_locals or 'cls' in frame.f_locals\n"
        "        if method and filename.endswith('/mgcdriver/driver.py'):\n"
        "            raise RuntimeError('Driver or buffer execution forbidden')\n"
        "        if method and filename.endswith('/mgcdriver/backends.py') and name != '__post_init__':\n"
        "            raise RuntimeError('backend or transport execution forbidden')\n"
        "sys.addaudithook(audit)\n"
        "sys.setprofile(profile)\n",
        encoding="utf-8",
    )
    destination = tmp_path / "audited-report.json"
    completed = run_cli(write_manifest(tmp_path), "--output", destination,
                        env={"PYTHONPATH": str(guard) + os.pathsep + str(ROOT)})
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert_report(json.loads(destination.read_text()), "structurally-consistent")
    assert not list(tmp_path.rglob("__pycache__"))


def test_cli_has_no_hardware_option(tmp_path):
    completed = run_cli(write_manifest(tmp_path), "--hardware")
    assert completed.returncode == 2
    assert "unrecognized arguments" in completed.stderr


def test_null_template_can_never_pass():
    template = ROOT / "docs/driver/board-manifest.template.json"
    assert_report(preflight.assess_json(template.read_bytes()), "missing")
