"""Explicit, same-user Unix IPC for the existing managed legacy host renderer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time

import numpy as np

from tools.host_blend import BLEND_MODES
from tools.host_pipeline_state import validate_state
from tools.host_render import render_artifact
from tools.mgcdriver import DriverError, ProgramFormat
from tools.shader_layout import bounded_integer


REQUEST_MAGIC = b"MGCHE01\0"
REPLY_MAGIC = b"MGCRG01\0"
REQUEST_DEPTH_MAGIC = b"MGCHE02\0"
REPLY_DEPTH_MAGIC = b"MGCRG02\0"
REQUEST_FEATURES_MAGIC = b"MGCHE04\0"
REPLY_FEATURES_MAGIC = b"MGCRG04\0"
REQUEST = struct.Struct("<8sIQIIII")
REPLY = struct.Struct("<8sQIIII")
REQUEST_FEATURES = struct.Struct("<8sIQIIIIII")
REPLY_FEATURES = struct.Struct("<8sQIIIIIII")
COLOR_FORMATS = {1: ("rgba8", 4), 2: ("bgra8", 4), 3: ("rgb565", 2), 4: ("r8", 1)}
DEPTH_FORMATS = {0: ("none", 0), 1: ("z16", 2), 2: ("z24s8", 4)}
MAX_JSON_BYTES = 12 << 20


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate host execution JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite host execution JSON constant: {value}")


def _receive(connection, size, deadline):
    result = bytearray()
    while len(result) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("host execution input deadline exceeded")
        connection.settimeout(remaining)
        chunk = connection.recv(min(size - len(result), 65536))
        if not chunk:
            raise ValueError("truncated host execution request")
        result.extend(chunk)
    return bytes(result)


def decode_native_color(contents, width, height, format_tag):
    if type(format_tag) is not int or format_tag not in COLOR_FORMATS:
        raise ValueError("unsupported native color format tag")
    width = bounded_integer(width, "native color width", 128, 1)
    height = bounded_integer(height, "native color height", 128, 1)
    if type(contents) is not bytes or len(contents) != width * height * COLOR_FORMATS[format_tag][1]:
        raise ValueError("native color payload must match format and geometry")
    if format_tag in (1, 2):
        image = np.frombuffer(contents, dtype=np.uint8).reshape(height, width, 4).copy()
        return image if format_tag == 1 else image[:, :, [2, 1, 0, 3]].copy()
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    if format_tag == 4:
        image[:, :, 0] = np.frombuffer(contents, dtype=np.uint8).reshape(height, width)
    else:
        packed = np.frombuffer(contents, dtype="<u2").reshape(height, width).astype(np.uint32)
        for channel, shift, maximum in ((0, 11, 31), (1, 5, 63), (2, 0, 31)):
            image[:, :, channel] = np.rint(((packed >> shift) & maximum) * (255 / maximum)).astype(np.uint8)
    return image


def encode_native_color(image, format_tag):
    if (type(format_tag) is not int or format_tag not in COLOR_FORMATS or
            not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 4):
        raise ValueError("native color encoding requires a typed RGBA8 image and format tag")
    bounded_integer(image.shape[1], "native color width", 128, 1)
    bounded_integer(image.shape[0], "native color height", 128, 1)
    if format_tag == 1:
        return image.tobytes()
    if format_tag == 2:
        return image[:, :, [2, 1, 0, 3]].tobytes()
    if format_tag == 4:
        return image[:, :, 0].tobytes()
    packed = np.zeros(image.shape[:2], dtype=np.uint16)
    for channel, shift, maximum in ((0, 11, 31), (1, 5, 63), (2, 0, 31)):
        packed |= np.rint(image[:, :, channel].astype(np.float64) * (maximum / 255)).astype(np.uint16) << shift
    return packed.astype("<u2").tobytes()


class HostExecutorServer:
    def __init__(self, path, *, max_requests=32, max_steps=2000000,
                 max_fragments=8192, timeout=5):
        self.path = Path(path)
        if not hasattr(socket, "SO_PEERCRED"):
            raise ValueError("host executor requires Linux same-user Unix peer credentials")
        if not self.path.is_absolute() or b"\0" in os.fsencode(self.path) or len(os.fsencode(self.path)) > 107:
            raise ValueError("host executor requires an absolute bounded Unix socket path")
        self.max_requests = bounded_integer(max_requests, "host request count", 128, 1)
        self.max_steps = bounded_integer(max_steps, "host instruction budget", 100000000, 1)
        self.max_fragments = bounded_integer(max_fragments, "host fragment budget", 65536, 1)
        if type(timeout) not in (int, float) or not np.isfinite(timeout) or not 0 < timeout <= 5:
            raise ValueError("host executor timeout must be finite and at most five seconds")
        self.timeout = float(timeout)
        self.records = []
        self.errors = []
        self._stop = threading.Event()
        self._socket = None
        self._identity = None
        self._thread = None

    def __enter__(self):
        if self._socket is not None or self._thread is not None:
            raise ValueError("host executor server cannot be entered twice")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.path))
            identity = self.path.lstat()
            self._identity = (identity.st_dev, identity.st_ino)
            os.chmod(self.path, 0o600)
            listener.listen(1)
            listener.settimeout(0.1)
        except Exception:
            listener.close()
            self._remove_owned_socket()
            raise
        self._socket = listener
        self._thread = threading.Thread(target=self._serve, name="mgc-host-executor", daemon=True)
        self._thread.start()
        return self

    def _remove_owned_socket(self):
        if self._identity is None:
            return
        try:
            identity = self.path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(identity.st_mode) and (identity.st_dev, identity.st_ino) == self._identity:
            self.path.unlink()

    def __exit__(self, error_type, error, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.timeout + 6)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._remove_owned_socket()
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("bounded host executor failed to stop")

    def _serve(self):
        for request in range(self.max_requests):
            while not self._stop.is_set():
                try:
                    connection, address = self._socket.accept()
                    break
                except socket.timeout:
                    continue
                except OSError:
                    return
            else:
                return
            with connection:
                try:
                    credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                    pid, uid, gid = struct.unpack("3i", credentials)
                    if uid != os.geteuid():
                        raise ValueError("host executor peer must have the same effective user")
                    self._handle(connection, pid)
                except (ValueError, OSError, DriverError, RecursionError, MemoryError,
                        KeyError, TypeError, OverflowError) as error:
                    self.errors.append(str(error))

    def _handle(self, connection, pid):
        deadline = time.monotonic() + self.timeout
        magic = _receive(connection, 8, deadline)
        if magic == REQUEST_FEATURES_MAGIC:
            return self._handle_features(connection, pid, deadline, magic)
        header = magic + _receive(connection, REQUEST.size - 8, deadline)
        magic, json_size, sequence, width, height, blend, image_size = REQUEST.unpack(header)
        if magic not in (REQUEST_MAGIC, REQUEST_DEPTH_MAGIC) or sequence == 0:
            raise ValueError("invalid host executor protocol or sequence")
        bounded_integer(json_size, "host artifact bytes", MAX_JSON_BYTES, 1)
        bounded_integer(width, "host framebuffer width", 128, 1)
        bounded_integer(height, "host framebuffer height", 128, 1)
        if blend >= len(BLEND_MODES) or image_size != width * height * 4:
            raise ValueError("unsupported host state or framebuffer payload size")
        depth_size = 0
        if magic == REQUEST_DEPTH_MAGIC:
            depth_size = struct.unpack("<I", _receive(connection, 4, deadline))[0]
            if depth_size != width * height * 2:
                raise ValueError("native Z16 payload must match framebuffer geometry")
        try:
            artifact = json.loads(_receive(connection, json_size, deadline).decode("utf-8"),
                                  object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            initial = np.frombuffer(_receive(connection, image_size, deadline), dtype=np.uint8)
            initial = initial.reshape(height, width, 4).copy()
            initial_depth = None
            output_depth = None
            if depth_size:
                initial_depth = np.frombuffer(_receive(connection, depth_size, deadline), dtype="<u2")
                initial_depth = initial_depth.reshape(height, width).astype(np.uint16)
                output_depth = np.empty((height, width), dtype=np.uint16)
            if not isinstance(artifact, dict) or artifact.get("program_format") != ProgramFormat.LEGACY.value:
                raise ValueError("native host execution accepts only explicitly tagged legacy programs")
            if type(artifact.get("sequence")) is not int or artifact["sequence"] != sequence:
                raise ValueError("host artifact and execution sequence disagree")
            if artifact.get("schema") == "mgc-host-draw-v4":
                raise ValueError("ABI4 execution requires the version4 native protocol")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("host execution input deadline exceeded")
            image, coverage, report = render_artifact(
                artifact, width=width, height=height, blend=BLEND_MODES[blend],
                initial_image=initial, max_steps=self.max_steps, max_fragments=self.max_fragments,
                execution_timeout=remaining, depth_test=bool(depth_size),
                initial_depth=initial_depth, depth_output=output_depth,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("host execution result deadline exceeded")
            connection.settimeout(remaining)
            reply = REPLY.pack(REPLY_DEPTH_MAGIC if depth_size else REPLY_MAGIC, sequence, 0, width, height, image_size)
            if depth_size:
                reply += struct.pack("<I", depth_size)
            reply += image.tobytes()
            if depth_size:
                reply += output_depth.astype("<u2").tobytes()
            connection.sendall(reply)
            self.records.append({"sequence": sequence, "peer_pid": pid, "artifact": artifact,
                                 "report": report, "rgba": image.copy(), "coverage": coverage.copy(),
                                 "depth": None if output_depth is None else output_depth.copy()})
        except (ValueError, OSError, DriverError, RecursionError, MemoryError,
                KeyError, TypeError, OverflowError) as error:
            self.errors.append(str(error))
            try:
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                reply = REPLY.pack(REPLY_DEPTH_MAGIC if depth_size else REPLY_MAGIC, sequence, 1, width, height, 0)
                if depth_size:
                    reply += struct.pack("<I", 0)
                connection.sendall(reply)
            except OSError:
                pass

    def _handle_features(self, connection, pid, deadline, magic):
        header = magic + _receive(connection, REQUEST_FEATURES.size - 8, deadline)
        magic, json_size, sequence, width, height, color_tag, depth_tag, color_size, depth_size = REQUEST_FEATURES.unpack(header)
        if sequence == 0:
            raise ValueError("invalid host executor protocol or sequence")
        bounded_integer(json_size, "host artifact bytes", MAX_JSON_BYTES, 1)
        bounded_integer(width, "host framebuffer width", 128, 1)
        bounded_integer(height, "host framebuffer height", 128, 1)
        if (color_tag not in COLOR_FORMATS or depth_tag not in DEPTH_FORMATS or
                color_size != width * height * COLOR_FORMATS[color_tag][1] or
                depth_size != width * height * DEPTH_FORMATS[depth_tag][1]):
            raise ValueError("native ABI4 attachment tags and byte sizes must match geometry")
        try:
            from tools.host_features import validate_draw_features

            artifact = json.loads(_receive(connection, json_size, deadline).decode("utf-8"),
                                  object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if type(artifact) is not dict or not validate_draw_features(artifact):
                raise ValueError("native ABI4 execution requires an exact version4 draw artifact")
            if type(artifact.get("sequence")) is not int or artifact["sequence"] != sequence:
                raise ValueError("host artifact and execution sequence disagree")
            state = validate_state(artifact.get("render_state"))
            if (state.color_format != COLOR_FORMATS[color_tag][0] or
                    state.depth_stencil_format != DEPTH_FORMATS[depth_tag][0]):
                raise ValueError("native attachment tags differ from captured render state")
            initial_native = _receive(connection, color_size, deadline)
            initial = decode_native_color(initial_native, width, height, color_tag)
            initial_depth = _receive(connection, depth_size, deadline) if depth_size else None
            output_depth = bytearray(depth_size) if depth_size else None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("host execution input deadline exceeded")
            image, coverage, report = render_artifact(
                artifact, width=width, height=height, initial_image=initial,
                max_steps=self.max_steps, max_fragments=self.max_fragments,
                execution_timeout=remaining, render_state=state.to_dict(),
                initial_depth_stencil=initial_depth, depth_stencil_output=output_depth,
            )
            color = encode_native_color(image, color_tag)
            depth = b"" if output_depth is None else bytes(output_depth)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("host execution result deadline exceeded")
            connection.settimeout(remaining)
            connection.sendall(REPLY_FEATURES.pack(REPLY_FEATURES_MAGIC, sequence, 0, width, height,
                                                  color_tag, depth_tag, color_size, depth_size) + color + depth)
            self.records.append({"sequence": sequence, "peer_pid": pid, "artifact": artifact,
                                 "report": report, "rgba": image.copy(), "coverage": coverage.copy(),
                                 "depth": depth or None, "native_color": color,
                                 "color_format": COLOR_FORMATS[color_tag][0],
                                 "depth_format": DEPTH_FORMATS[depth_tag][0]})
        except (ValueError, OSError, DriverError, RecursionError, MemoryError,
                KeyError, TypeError, OverflowError) as error:
            self.errors.append(str(error))
            try:
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                connection.sendall(REPLY_FEATURES.pack(REPLY_FEATURES_MAGIC, sequence, 1, width, height,
                                                      color_tag, depth_tag, 0, 0))
            except OSError:
                pass
