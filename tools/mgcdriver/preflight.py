"""Bounded, metadata-only board preflight; never opens or executes hardware."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields
import json
import os
import re
import secrets
import stat
import sys

from .backends import MMIOConfig
from .errors import ValidationError
from .program import ADDRESS_LIMIT, MAX_PROGRAM_WORDS

MAX_JSON_BYTES = 64 * 1024
MAX_REPORT_BYTES = 256 * 1024
MAX_DEPTH = 16
MAX_NODES = 4096
MAX_STRING = 512
MAX_MAPS = 16
MAX_FINDINGS = 128
MAX_FILE_OFFSET = (1 << 63) - 1
TRANSPORT_METHODS = (
    "read32", "write32", "read", "write", "sync_for_device", "sync_for_cpu", "barrier",
)
ABI = {
    "program_format": "fp16-hardware-v1",
    "program_word_bits": 32,
    "register_width_bits": 32,
    "register_byteorder": "little",
    "entry_units": "instruction-words",
    "entry_register_units": "program-byte-offset",
    "count_register_units": "32-bit-words",
}
DEVICE_FIELDS = {"identity", "identity_ref", "access", "access_ref", "ownership_ref"}
MAP_FIELDS = {"name", "index", "size", "offset", "page_size", "mapping_ref"}
CONTRACT_REFS = {
    *(method + "_ref" for method in TRANSPORT_METHODS),
    "coherency_ref", "bounded_io_ref", "start_acceptance_ref",
    "acknowledgement_ref", "completion_ref", "idle_writes_drained_ref", "stop_ref",
}
REGISTER_FIELDS = (
    "control_offset", "status_offset", "entry_offset", "word_count_offset",
    "acknowledge_offset", "abort_offset",
)
TOP_FIELDS = {
    "manifest_version", "device", "maps", "windows", "mmio", "abi", "submission", "contracts",
}
NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def _check_tree(value):
    pending = [(value, 0, frozenset())]
    nodes = 0
    text_bytes = 0
    while pending:
        current, depth, ancestors = pending.pop()
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise ValidationError("manifest exceeds node or nesting limit")
        if type(current) in (dict, list):
            if id(current) in ancestors:
                raise ValidationError("manifest must be a JSON tree, not a cyclic object")
            if len(current) > MAX_NODES:
                raise ValidationError("manifest exceeds node limit")
            descendants = ancestors | {id(current)}
            if type(current) is dict:
                for key, child in current.items():
                    if type(key) is not str:
                        raise ValidationError("JSON object keys must be strings")
                    pending.append((key, depth + 1, descendants))
                    pending.append((child, depth + 1, descendants))
            else:
                pending.extend((child, depth + 1, descendants) for child in current)
        elif type(current) is str:
            if len(current) > MAX_STRING:
                raise ValidationError("manifest string exceeds 512-character limit")
            try:
                text_bytes += len(current.encode("utf-8"))
            except UnicodeError as error:
                raise ValidationError("manifest contains invalid Unicode") from error
            if text_bytes > MAX_JSON_BYTES:
                raise ValidationError("manifest text exceeds byte limit")
        elif type(current) is int:
            if abs(current) > MAX_FILE_OFFSET:
                raise ValidationError("manifest integer exceeds signed 64-bit limit")
        elif current is not None:
            raise ValidationError("manifest permits only objects, arrays, strings, integers and null")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_number(value):
    raise ValidationError("floating-point and non-finite JSON numbers are not permitted")


def _parse_integer(value):
    if len(value.lstrip("-")) > 20:
        raise ValidationError("JSON integer exceeds loading limit")
    return int(value)


def _parse_json(data):
    if type(data) is str:
        if len(data) > MAX_JSON_BYTES:
            raise ValidationError("JSON input exceeds 65536-byte limit")
        try:
            data = data.encode("utf-8")
        except UnicodeError as error:
            raise ValidationError("manifest contains invalid Unicode") from error
    if type(data) is not bytes:
        raise ValidationError("JSON input must be bytes or text")
    if len(data) > MAX_JSON_BYTES:
        raise ValidationError("JSON input exceeds 65536-byte limit")
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise ValidationError("JSON input must be UTF-8") from error
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise ValidationError("JSON input exceeds nesting limit")
        elif character in "]}":
            depth -= 1
    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=_reject_number, parse_float=_reject_number,
                          parse_int=_parse_integer)
    except (ValueError, RecursionError) as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError("invalid JSON syntax") from error


def _report(findings, version=None, plan=None):
    kinds = {finding["kind"] for finding in findings}
    status = "incompatible" if "incompatible" in kinds else "missing" if kinds else "structurally-consistent"
    return {
        "report_version": 1,
        "manifest_version": version,
        "status": status,
        "hardware_execution": False,
        "readiness_unverified": True,
        "references_verified": False,
        "findings": findings,
        "plan": plan if not findings else None,
    }


def _failure(path, message):
    return _report([{"kind": "incompatible", "path": path, "message": message[:1024]}])


class _Assessment:
    def __init__(self):
        self.findings = []
        self.maps_present = False
        self.map_names_complete = True
        self.declared_map_names = set()

    def issue(self, kind, path, message):
        finding = {"kind": kind, "path": path[:512], "message": message[:1024]}
        if len(self.findings) < MAX_FINDINGS:
            self.findings.append(finding)
        elif kind == "incompatible" and not any(item["kind"] == kind for item in self.findings):
            self.findings[-1] = finding

    def object(self, value, path, expected):
        if value is None:
            self.issue("missing", path, "required object is unknown or absent")
            return None
        if type(value) is not dict:
            self.issue("incompatible", path, "must be an object")
            return None
        for key in sorted(value):
            if key not in expected:
                self.issue("incompatible", path + "[" + json.dumps(key) + "]", "unknown field")
        return value

    def text(self, value, path, *, name=False):
        if value is None:
            self.issue("missing", path, "required text/reference is unknown or absent")
            return None
        if (type(value) is not str or not value.strip() or value != value.strip()
                or any(ord(character) < 32 or ord(character) == 127 for character in value)):
            self.issue("incompatible", path, "must be nonempty text without surrounding whitespace or controls")
            return None
        if name and NAME_PATTERN.fullmatch(value) is None:
            self.issue("incompatible", path, "must be an ASCII map name of 1..64 characters")
            return None
        return value

    def integer(self, value, path, minimum=0, maximum=ADDRESS_LIMIT - 1):
        if value is None:
            self.issue("missing", path, "required integer is unknown or absent")
            return None
        if type(value) is not int or not minimum <= value <= maximum:
            self.issue("incompatible", path, f"must be an integer in {minimum}..{maximum}; booleans are not integers")
            return None
        return value

    def mmio(self, value):
        expected = {field.name for field in fields(MMIOConfig)}
        configuration = self.object(value, "$.mmio", expected)
        if configuration is None:
            return None
        valid = True
        for name in sorted(expected):
            member = configuration.get(name)
            path = "$.mmio." + name
            if name not in configuration:
                self.issue("missing", path, "every MMIOConfig field must be explicit")
                valid = False
            elif name == "byteorder":
                if member is None:
                    self.issue("missing", path, "program word byte order must be explicit")
                    valid = False
                elif type(member) is not str or member != "little":
                    self.issue("incompatible", path, "this preflight profile supports only little-endian program words")
                    valid = False
            elif member is None and name in ("abort_offset", "abort_value"):
                continue
            elif self.integer(member, path, maximum=ADDRESS_LIMIT) is None:
                valid = False
        if not valid:
            return None
        try:
            return MMIOConfig(**{name: configuration[name] for name in expected})
        except ValidationError as error:
            self.issue("incompatible", "$.mmio", str(error))
            return None

    def maps(self, value):
        if value is None:
            self.issue("missing", "$.maps", "supplied map descriptors are unknown or absent")
            return {}
        if type(value) is not list or not 1 <= len(value) <= MAX_MAPS:
            self.issue("incompatible", "$.maps", "must be an array of 1..16 supplied map descriptors")
            return {}
        self.maps_present = True
        result = {}
        indices = set()
        page_sizes = set()
        for position, member in enumerate(value):
            path = f"$.maps[{position}]"
            descriptor = self.object(member, path, MAP_FIELDS)
            if descriptor is None:
                self.map_names_complete = False
                continue
            name = self.text(descriptor.get("name"), path + ".name", name=True)
            if name is not None:
                self.declared_map_names.add(name)
            else:
                self.map_names_complete = False
            index = self.integer(descriptor.get("index"), path + ".index", maximum=MAX_FILE_OFFSET)
            size = self.integer(descriptor.get("size"), path + ".size", 1, ADDRESS_LIMIT)
            offset = self.integer(descriptor.get("offset"), path + ".offset")
            page_size = self.integer(descriptor.get("page_size"), path + ".page_size", 4, 1 << 20)
            reference = self.text(descriptor.get("mapping_ref"), path + ".mapping_ref")
            if any(item is None for item in (name, index, size, offset, page_size, reference)):
                continue
            if name in result or index in indices:
                self.issue("incompatible", path, "map names and indices must be unique for the supplied access identity")
            if page_size & (page_size - 1) or size % page_size or offset >= min(page_size, size):
                self.issue("incompatible", path, "page size must be a power of two; size must be page-multiple; offset must fit the first page")
            if index * page_size + size > MAX_FILE_OFFSET:
                self.issue("incompatible", path, "UIO selection offset and mapping span exceed signed 64-bit range")
            indices.add(index)
            page_sizes.add(page_size)
            result[name] = dict(descriptor)
        if len(page_sizes) > 1:
            self.issue("incompatible", "$.maps", "maps for one access identity must declare the same page size")
        return result

    def windows(self, value, maps, configuration):
        windows = self.object(value, "$.windows", {"program", "data", "registers"})
        if windows is None:
            return {}
        result = {}
        ranges = []
        for role in ("program", "data", "registers"):
            path = "$.windows." + role
            expected = {"map", "byte_offset", "size", "reservation"}
            if role != "registers":
                expected.add("gpu_byte_base")
            window = self.object(windows.get(role), path, expected)
            if window is None:
                continue
            name = self.text(window.get("map"), path + ".map", name=True)
            offset = self.integer(window.get("byte_offset"), path + ".byte_offset")
            size = self.integer(window.get("size"), path + ".size", 4, ADDRESS_LIMIT)
            gpu_base = None
            if role != "registers":
                gpu_base = self.integer(window.get("gpu_byte_base"), path + ".gpu_byte_base")
                if gpu_base is not None and size is not None and gpu_base + size > ADDRESS_LIMIT:
                    self.issue("incompatible", path, "GPU window exceeds the current 32-bit byte address space")
            reservation = self.object(window.get("reservation"), path + ".reservation",
                                      {"byte_offset", "size", "reference"})
            reserved_offset = reserved_size = reference = None
            if reservation is not None:
                reserved_offset = self.integer(reservation.get("byte_offset"), path + ".reservation.byte_offset")
                reserved_size = self.integer(reservation.get("size"), path + ".reservation.size", 4, ADDRESS_LIMIT)
                reference = self.text(reservation.get("reference"), path + ".reservation.reference")
            required = (name, offset, size, reserved_offset, reserved_size, reference)
            if any(item is None for item in required) or (role != "registers" and gpu_base is None):
                continue
            descriptor = maps.get(name)
            if descriptor is None:
                incomplete = not self.maps_present or not self.map_names_complete or name in self.declared_map_names
                kind = "missing" if incomplete else "incompatible"
                self.issue(kind, path + ".map", "does not identify a complete supplied map descriptor")
                continue
            start = descriptor["offset"] + offset
            reserved_start = descriptor["offset"] + reserved_offset
            if start % 4 or size % 4:
                self.issue("incompatible", path, "mapped window start and size must be 4-byte aligned")
            if start + size > descriptor["size"] or reserved_start + reserved_size > descriptor["size"]:
                self.issue("incompatible", path, "window and reservation must lie completely inside the supplied mapping span")
            if reserved_offset > offset or reserved_offset + reserved_size < offset + size:
                self.issue("incompatible", path + ".reservation", "reservation does not cover the complete declared window")
            for previous_name, previous_start, previous_end in ranges:
                if name == previous_name and start < previous_end and previous_start < start + size:
                    self.issue("incompatible", path, "program, data and register windows must not overlap in one map")
            ranges.append((name, start, start + size))
            target_base = gpu_base
            target_size = size
            if configuration is not None and role != "registers":
                target_base = configuration.program_base if role == "program" else configuration.memory_base
                target_size = configuration.program_capacity if role == "program" else configuration.memory_size
                if gpu_base > target_base or target_base + target_size > gpu_base + size:
                    self.issue("incompatible", path, "supplied GPU-to-map window does not cover the complete MMIOConfig aperture")
                else:
                    start += target_base - gpu_base
            result[role] = {
                "map": name, "within_map_start_offset": start, "size": target_size,
                "gpu_byte_base": target_base,
                "reservation_within_map_start_offset": reserved_start,
                "reservation_size": reserved_size, "reservation_ref": reference,
            }
        if configuration is not None and "registers" in result:
            register_window = result["registers"]
            registers = {}
            for name in REGISTER_FIELDS:
                offset = getattr(configuration, name)
                if offset is not None:
                    if offset + 4 > register_window["size"]:
                        self.issue("incompatible", "$.mmio." + name, "32-bit register access escapes the supplied register window")
                    registers[name] = register_window["within_map_start_offset"] + offset
            register_window["register_within_map_offsets"] = registers
        return result


def assess_manifest(manifest):
    """Validate supplied metadata only. No paths, references or devices are read."""
    try:
        _check_tree(manifest)
    except ValidationError as error:
        return _failure("$", str(error))
    assessment = _Assessment()
    root = assessment.object(manifest, "$", TOP_FIELDS)
    if root is None:
        return _report(assessment.findings)
    version = assessment.integer(root.get("manifest_version"), "$.manifest_version", 1, MAX_FILE_OFFSET)
    if version is not None and version != 1:
        assessment.issue("incompatible", "$.manifest_version", "only manifest version 1 is supported")
    device = assessment.object(root.get("device"), "$.device", DEVICE_FIELDS)
    if device is not None:
        for name in sorted(DEVICE_FIELDS):
            assessment.text(device.get(name), "$.device." + name)
    abi = assessment.object(root.get("abi"), "$.abi", set(ABI))
    if abi is not None:
        for name, expected in ABI.items():
            actual = abi.get(name)
            if actual is None:
                assessment.issue("missing", "$.abi." + name, "current backend ABI declaration is unknown or absent")
            elif type(actual) is not type(expected) or actual != expected:
                assessment.issue("incompatible", "$.abi." + name, f"current preflight profile requires {expected!r}")
    configuration = assessment.mmio(root.get("mmio"))
    maps = assessment.maps(root.get("maps"))
    windows = assessment.windows(root.get("windows"), maps, configuration)
    contracts = assessment.object(root.get("contracts"), "$.contracts", CONTRACT_REFS | {"stop_policy"})
    if contracts is not None:
        for name in sorted(CONTRACT_REFS):
            assessment.text(contracts.get(name), "$.contracts." + name)
        policy = assessment.text(contracts.get("stop_policy"), "$.contracts.stop_policy")
        if policy is not None and policy not in ("abort", "quarantine"):
            assessment.issue("incompatible", "$.contracts.stop_policy", "must be 'abort' or 'quarantine'")
        if configuration is not None:
            if policy == "abort" and configuration.abort_offset is None:
                assessment.issue("missing", "$.mmio.abort_offset", "abort policy requires explicit abort offset and value")
            elif policy == "quarantine" and configuration.abort_offset is not None:
                assessment.issue("incompatible", "$.mmio.abort_offset", "quarantine-only policy requires both abort fields to be null")
    submission = assessment.object(root.get("submission"), "$.submission", {"program_word_count", "entry_word"})
    count = entry = None
    if submission is not None:
        count = assessment.integer(submission.get("program_word_count"), "$.submission.program_word_count", 1, MAX_PROGRAM_WORDS)
        entry = assessment.integer(submission.get("entry_word"), "$.submission.entry_word")
        if count is not None and entry is not None and entry >= count:
            assessment.issue("incompatible", "$.submission.entry_word", "entry word must be inside the declared uploaded program")
        if configuration is not None and count is not None and count * 4 > configuration.program_capacity:
            assessment.issue("incompatible", "$.submission.program_word_count", "declared program exceeds the MMIOConfig instruction capacity")
    if assessment.findings:
        return _report(assessment.findings, version)
    plan = {
        "device": dict(device), "abi": dict(abi), "mmio": asdict(configuration),
        "contracts": dict(contracts),
        "maps": [dict(descriptor, mmap_selection_offset=descriptor["index"] * descriptor["page_size"])
                 for descriptor in maps.values()],
        "windows": windows,
        "submission": {
            "program_word_count": count, "program_size_bytes": count * 4,
            "entry_word": entry, "entry_register_value": entry * 4,
            "word_count_register_value": count,
        },
    }
    return _report([], version, plan)


def assess_json(data):
    """Return a complete failure report for malformed or incompatible JSON."""
    try:
        manifest = _parse_json(data)
    except ValidationError as error:
        return _failure("$", str(error))
    return assess_manifest(manifest)


@contextmanager
def _ordinary_parent(path):
    path = os.fspath(path)
    if type(path) is not str or not path or "\0" in path:
        raise ValidationError("file path must be nonempty ordinary text")
    absolute = os.path.abspath(path)
    if any(absolute == root or absolute.startswith(root + "/") for root in ("/dev", "/sys", "/proc")):
        raise ValidationError("device, sysfs and procfs paths are forbidden")
    components = absolute.split("/")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for component in components[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        yield directory, components[-1]
    finally:
        os.close(directory)


def _read_ordinary_json(path):
    with _ordinary_parent(path) as (directory, name):
        metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError("input must be a regular, non-symlink JSON file")
        if metadata.st_size > MAX_JSON_BYTES:
            raise ValidationError("JSON input exceeds 65536-byte limit")
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                             dir_fd=directory)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ValidationError("input changed while opening")
            chunks = []
            size = 0
            while size <= MAX_JSON_BYTES:
                chunk = os.read(descriptor, MAX_JSON_BYTES + 1 - size)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            if size > MAX_JSON_BYTES:
                raise ValidationError("JSON input exceeds 65536-byte limit")
            return b"".join(chunks)
        finally:
            os.close(descriptor)


def _serialized(report):
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > MAX_REPORT_BYTES:
        raise ValidationError("report exceeds byte limit")
    return data


def _write_atomic_report(path, report):
    data = _serialized(report)
    with _ordinary_parent(path) as (directory, name):
        try:
            os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValidationError("output already exists; overwrites and symlink outputs are forbidden")
        temporary = None
        descriptor = None
        try:
            for attempt in range(8):
                candidate = ".board-preflight-" + secrets.token_hex(16) + ".tmp"
                try:
                    descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                         0o600, dir_fd=directory)
                except FileExistsError:
                    continue
                temporary = candidate
                break
            if descriptor is None:
                raise ValidationError("could not create a private report file")
            written = 0
            while written < len(data):
                amount = os.write(descriptor, data[written:])
                if amount <= 0:
                    raise OSError("short report write")
                written += amount
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                os.unlink(temporary, dir_fd=directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline metadata preflight only; never executes hardware or verifies references.")
    parser.add_argument("manifest", help="bounded, regular, non-symlink UTF-8 JSON file")
    parser.add_argument("--output", help="new ordinary report file; no overwrites or symlinks (default: stdout)")
    arguments = parser.parse_args(argv)
    try:
        report = assess_json(_read_ordinary_json(arguments.manifest))
    except (OSError, ValidationError, ValueError) as error:
        report = _failure("$input", str(error))
    try:
        if arguments.output is None:
            sys.stdout.write(_serialized(report).decode("utf-8"))
        else:
            _write_atomic_report(arguments.output, report)
    except (OSError, ValidationError, ValueError) as error:
        sys.stderr.write(_serialized(_failure("$output", str(error))).decode("utf-8"))
        return 2
    return 0 if report["status"] == "structurally-consistent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
