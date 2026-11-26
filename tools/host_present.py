"""Bounded, optional CPU-host RGBA8 presentation; never physical-GPU evidence."""

from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.util
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
MAX_DIMENSION = 128
MAX_FRAMES = 120
MAX_LIFETIME = 30
MAX_SCALE = 8
MAX_REPORT_BYTES = 1 << 20
TOP_LEFT = ("top-left", "exported top-left; raster coordinates bottom-left")
SCOPE = "bounded CPU-host software presentation; not fresh GL, physical GPU, Khronos or FPGA certification"


def bounded_integer(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def bounded_seconds(value, name, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be finite and in (0, {maximum}]")
    return value


def read_bounded(path, maximum):
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        storage = os.fstat(source.fileno())
        if not stat.S_ISREG(storage.st_mode) or storage.st_size > maximum:
            raise ValueError(f"{path} must be an ordinary bounded file (at most {maximum} bytes)")
        contents = source.read(maximum + 1)
    if len(contents) != storage.st_size or len(contents) > maximum:
        raise ValueError(f"{path} changed size during bounded read")
    return contents


def parse_json(contents):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"nonfinite JSON number: {value}")
        return number

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(contents, parse_constant=reject_constant, parse_float=finite_float,
                      object_pairs_hook=unique_object)


def file_source(path, contents):
    return {"path": str(Path(path).absolute()), "size_bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest()}


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    rgba: bytes
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        bounded_integer(self.width, "width", MAX_DIMENSION)
        bounded_integer(self.height, "height", MAX_DIMENSION)
        if type(self.rgba) is not bytes or len(self.rgba) != self.width * self.height * 4:
            raise ValueError("RGBA8 requires exactly width * height * 4 bytes")
        if type(self.source) is not dict:
            raise ValueError("source provenance must be an object")

    @property
    def rgb(self):
        return b"".join(self.rgba[offset:offset + 3] for offset in range(0, len(self.rgba), 4))

    @property
    def rgba_sha256(self):
        return hashlib.sha256(self.rgba).hexdigest()

    def ppm(self):
        return f"P6\n{self.width} {self.height}\n255\n".encode("ascii") + self.rgb


def load_frame(*, report=None, raw=None, width=None, height=None, origin=None):
    metadata = None
    provenance = {"kind": "saved-raw", "fresh_host_render": False, "fresh_gl_draw": False}
    if report is not None:
        contents = read_bounded(report, MAX_REPORT_BYTES)
        metadata = parse_json(contents)
        if type(metadata) is not dict or metadata.get("fpga_execution") is not False:
            raise ValueError("renderer report must explicitly describe CPU-host, not FPGA execution")
        if any(metadata.get(marker) is True for marker in
               ("physical_gpu_execution", "physical_or_rtl_validation", "rtl_timing_validated", "khronos_certification")):
            raise ValueError("renderer report exceeds bounded CPU-host scope")
        if metadata.get("passed") is False or metadata.get("status") in ("error", "failed", "running"):
            raise ValueError("renderer report is not a successful completed result")
        comparison = metadata.get("softpipe_comparison")
        if comparison is not None and (type(comparison) is not dict or comparison.get("passed") is not True):
            raise ValueError("renderer report contains a failed comparison")
        if metadata.get("native_gl_readback") is True:
            if metadata.get("host_executor_committed") is not True:
                raise ValueError("native report requires an accepted host executor commit")
            if metadata.get("format") != "R8G8B8A8_UNORM" or metadata.get("pixel_origin") != "top-left":
                raise ValueError("native report requires top-left exported RGBA8")
            recorded_origin = metadata.get("raw_pixel_origin")
            default_raw = metadata.get("raw_path")
            if type(default_raw) is not str or not default_raw:
                raise ValueError("native report requires a raw_path")
            default_raw = Path(default_raw)
            if not default_raw.is_absolute():
                default_raw = Path(report).parent / default_raw
            kind = "saved-native-readback"
        else:
            if (type(metadata.get("pipeline")) is not str or
                    any(part not in metadata["pipeline"] for part in ("managed VS", "managed FS", "RGBA8"))):
                raise ValueError("unsupported managed renderer report")
            recorded_origin = metadata.get("pixel_origin")
            if recorded_origin not in TOP_LEFT:
                raise ValueError("managed renderer report requires exported top-left pixels")
            default_raw = Path(report).parent / "frame.rgba8"
            kind = "saved-managed-render"
        if recorded_origin in TOP_LEFT:
            recorded_origin = "top-left"
        if recorded_origin not in ("top-left", "bottom-left"):
            raise ValueError("renderer report must declare the raw pixel origin")
        for provided, recorded, name in ((width, metadata.get("width"), "width"),
                                         (height, metadata.get("height"), "height"),
                                         (origin, recorded_origin, "origin")):
            if provided is not None and provided != recorded:
                raise ValueError(f"explicit {name} conflicts with renderer report")
            if provided is not None and name in ("width", "height"):
                bounded_integer(provided, name, MAX_DIMENSION)
        width, height, origin = metadata.get("width"), metadata.get("height"), recorded_origin
        raw = default_raw if raw is None else raw
        provenance.update(kind=kind, report=file_source(report, contents))
    bounded_integer(width, "width", MAX_DIMENSION)
    bounded_integer(height, "height", MAX_DIMENSION)
    origin = "top-left" if origin is None else origin
    if origin not in ("top-left", "bottom-left") or raw is None:
        raise ValueError("raw input and top-left/bottom-left pixel origin are required")
    contents = read_bounded(raw, width * height * 4)
    provenance.update(raw=file_source(raw, contents), raw_pixel_origin=origin)
    stride = width * 4
    rgba = contents if origin == "top-left" else b"".join(
        contents[offset:offset + stride] for offset in range((height - 1) * stride, -1, -stride))
    frame = Frame(width, height, rgba, provenance)
    if metadata is not None:
        expected = metadata.get("rgba_sha256")
        if metadata.get("native_gl_readback") is not True and expected is None:
            raise ValueError("managed renderer report requires rgba_sha256")
        if expected is not None and expected != frame.rgba_sha256:
            raise ValueError("raw pixels do not match renderer rgba_sha256")
    return frame


def render_capture(capture, *, width=16, height=16, timeout=10):
    bounded_integer(width, "width", MAX_DIMENSION)
    bounded_integer(height, "height", MAX_DIMENSION)
    bounded_seconds(timeout, "render timeout", 60)
    contents = read_bounded(capture, 64 << 20)
    from tools.host_render import render_artifact

    image, _, report = render_artifact(parse_json(contents), width=width, height=height,
                                     execution_timeout=timeout)
    return Frame(width, height, image.tobytes(),
                 {"kind": "fresh-managed-render-of-saved-capture", "fresh_host_render": True,
                  "fresh_gl_draw": False, "capture": file_source(capture, contents), "renderer": report})


def _subprocess(command, *, timeout, input=None, env=None):
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE if input is not None else None,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        output, errors = process.communicate(input=input, timeout=timeout)
        if process.returncode:
            raise ValueError(errors.decode("utf-8", errors="replace").strip() or
                             f"host subprocess exited {process.returncode}")
        return output
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"host subprocess exceeded {timeout:g}s hard deadline") from error
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=2)


def render_shader(shader_directory, *, build=None, target="legacy-mgcsim-v1", width=16, height=16, timeout=60):
    bounded_integer(width, "width", MAX_DIMENSION)
    bounded_integer(height, "height", MAX_DIMENSION)
    bounded_seconds(timeout, "render timeout", 120)
    if target not in ("legacy-mgcsim-v1", "fp16-hardware-v1"):
        raise ValueError("unsupported CPU-host render target")
    with tempfile.TemporaryDirectory(prefix="btp-present-render-") as temporary:
        command = [sys.executable, "-m", "tools.run_render", str(Path(shader_directory).absolute()),
                   "--target", target, "--width", str(width), "--height", str(height), "--output", temporary]
        if build is not None:
            command.extend(("--build", str(Path(build).absolute())))
        output = _subprocess(command, timeout=timeout)
        frame = load_frame(report=Path(temporary) / "render.json")
        return Frame(frame.width, frame.height, frame.rgba,
                     {**frame.source, "kind": "fresh-GL-capture-and-managed-render", "fresh_gl_draw": True,
                      "fresh_host_render": True, "command": command,
                      "stdout_sha256": hashlib.sha256(output).hexdigest(),
                      "temporary_renderer_files_retained": False})


class _XImage(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int),
                ("format", ctypes.c_int), ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
                ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int),
                ("depth", ctypes.c_int), ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int),
                ("red_mask", ctypes.c_ulong), ("green_mask", ctypes.c_ulong), ("blue_mask", ctypes.c_ulong)]


def _channel(pixel, mask):
    if not mask:
        raise ValueError("X11 drawable has no RGB channel mask")
    shift = (mask & -mask).bit_length() - 1
    maximum = mask >> shift
    if maximum & (maximum + 1):
        raise ValueError("X11 drawable has a noncontiguous RGB channel mask")
    return ((pixel & mask) >> shift) * 255 // maximum


def readback_x11(display, window, width, height, scale=1):
    bounded_integer(width, "width", MAX_DIMENSION)
    bounded_integer(height, "height", MAX_DIMENSION)
    bounded_integer(scale, "scale", MAX_SCALE)
    library = ctypes.util.find_library("X11")
    if not library:
        raise ValueError("libX11 is required for actual drawable readback")
    xlib = ctypes.CDLL(library)
    xlib.XOpenDisplay.argtypes, xlib.XOpenDisplay.restype = [ctypes.c_char_p], ctypes.c_void_p
    xlib.XCloseDisplay.argtypes, xlib.XCloseDisplay.restype = [ctypes.c_void_p], ctypes.c_int
    xlib.XSync.argtypes, xlib.XSync.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
    xlib.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                               ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
    xlib.XGetImage.restype = ctypes.POINTER(_XImage)
    xlib.XGetPixel.argtypes, xlib.XGetPixel.restype = [ctypes.POINTER(_XImage), ctypes.c_int, ctypes.c_int], ctypes.c_ulong
    xlib.XDestroyImage.argtypes, xlib.XDestroyImage.restype = [ctypes.POINTER(_XImage)], ctypes.c_int
    connection = xlib.XOpenDisplay(display.encode("ascii"))
    if not connection:
        raise ValueError("cannot open X11 display for drawable readback")
    image = None
    try:
        xlib.XSync(connection, 0)
        image = xlib.XGetImage(connection, window, 0, 0, width * scale, height * scale,
                              ctypes.c_ulong(-1).value, 2)
        if not image:
            raise ValueError("XGetImage returned no actual drawable pixels")
        masks = (image.contents.red_mask, image.contents.green_mask, image.contents.blue_mask)
        if image.contents.depth != 24 or masks != (0xFF0000, 0xFF00, 0xFF):
            raise ValueError("bounded presenter readback requires a 24-bit TrueColor X11 drawable")
        rgb = bytearray()
        for row in range(height):
            for column in range(width):
                pixel = xlib.XGetPixel(image, column * scale + scale // 2, row * scale + scale // 2)
                rgb.extend(_channel(pixel, mask) for mask in masks)
        return bytes(rgb)
    finally:
        if image:
            xlib.XDestroyImage(image)
        xlib.XCloseDisplay(connection)


def _present_window(frame, *, frames, lifetime, scale):
    try:
        import tkinter
    except ImportError as error:
        raise ValueError("window mode requires installed tkinter/Tk; use --headless") from error

    try:
        root = tkinter.Tk()
    except tkinter.TclError as error:
        raise ValueError(f"cannot open Tk/X11 display: {error}") from error
    closed_reason = None
    submitted = 0
    readback = None
    started = time.monotonic()

    def close():
        nonlocal closed_reason
        closed_reason = "wm-close"
        root.destroy()

    try:
        if root.tk.call("tk", "windowingsystem") != "x11":
            raise ValueError("bounded window presenter currently requires Tk/X11")
        root.title("BTP bounded CPU-host RGBA8")
        root.resizable(False, False)
        root.geometry(f"{frame.width * scale}x{frame.height * scale}")
        root.protocol("WM_DELETE_WINDOW", close)
        photo = tkinter.PhotoImage(master=root, data=frame.ppm(), format="PPM").zoom(scale, scale)
        surface = tkinter.Label(root, image=photo, borderwidth=0, highlightthickness=0, padx=0, pady=0)
        surface.pack()
        expected_rgb = frame.rgb
        deadline = started + lifetime
        while time.monotonic() < deadline and closed_reason is None:
            root.update()
            if closed_reason is not None:
                break
            now = time.monotonic()
            if surface.winfo_viewable() and submitted < frames and now >= started + submitted * lifetime / frames:
                surface.configure(image=photo)
                root.update_idletasks()
                if readback is None:
                    rgb = readback_x11(os.environ["DISPLAY"], surface.winfo_id(), frame.width, frame.height, scale)
                    mismatch = sum(rgb[offset:offset + 3] != expected_rgb[offset:offset + 3]
                                   for offset in range(0, len(rgb), 3))
                    if mismatch:
                        raise ValueError(f"actual X11 readback differs at {mismatch} source pixels")
                    readback = {"backend": "XGetImage/XGetPixel", "pixel_origin": "top-left",
                                "rgb_sha256": hashlib.sha256(rgb).hexdigest(), "mismatch_pixels": mismatch,
                                "sampled_source_pixels": frame.width * frame.height,
                                "scale_sampling": "one center pixel per nearest-neighbor source block"}
                submitted += 1
            time.sleep(min(0.005, max(0, deadline - time.monotonic())))
        if readback is None:
            raise ValueError("window closed or expired before actual mapped drawable readback")
        return {"displayed": True, "frames_presented": submitted, "close_reason": closed_reason or "lifetime",
                "window_closed": True, "readback": readback, "window_system": "Tk/X11",
                "elapsed_seconds": time.monotonic() - started}
    except tkinter.TclError as error:
        raise ValueError(f"Tk/X11 presentation failed: {error}") from error
    finally:
        if closed_reason is None:
            root.destroy()


def present_frame(frame, *, headless=False, frames=1, lifetime=1, scale=4):
    if not isinstance(frame, Frame) or type(headless) is not bool:
        raise ValueError("presentation requires a Frame and explicit boolean headless mode")
    bounded_integer(frames, "frames", MAX_FRAMES)
    bounded_seconds(lifetime, "lifetime", MAX_LIFETIME)
    bounded_integer(scale, "scale", MAX_SCALE)
    report = {"schema": "mgc-host-presentation-v1", "passed": True, "scope": SCOPE,
              "fpga_execution": False, "physical_gpu_execution": False, "fresh_gl_draw": False,
              "width": frame.width, "height": frame.height, "pixel_origin": "top-left",
              "input_rgba_sha256": frame.rgba_sha256, "source": frame.source,
              "alpha_policy": "RGB shown opaque, alpha preserved in input hash; no compositing claim",
              "frames_requested": frames, "lifetime_seconds": lifetime, "scale": scale}
    if headless:
        return {**report, "mode": "headless", "displayed": False, "frames_presented": 0,
                "window_closed": None, "readback": None,
                "ppm_sha256": hashlib.sha256(frame.ppm()).hexdigest()}
    display = os.environ.get("DISPLAY", "")
    if not re.fullmatch(r"(?:unix)?:[0-9]{1,5}(?:\.[0-9]{1,2})?", display):
        raise ValueError("window mode requires an explicit local X11 DISPLAY; use --headless to avoid a window")
    payload = json.dumps({"width": frame.width, "height": frame.height,
                          "rgba": base64.b64encode(frame.rgba).decode("ascii"),
                          "frames": frames, "lifetime": lifetime, "scale": scale}).encode("ascii")
    output = _subprocess([sys.executable, "-m", "tools.host_present", "--window-worker"],
                         timeout=lifetime + 5, input=payload)
    result = parse_json(output)
    if type(result) is not dict or result.get("displayed") is not True or result.get("window_closed") is not True:
        raise ValueError("window worker produced no verified closed presentation")
    readback = result.get("readback")
    if (type(readback) is not dict or readback.get("backend") != "XGetImage/XGetPixel" or
            readback.get("mismatch_pixels") != 0 or
            readback.get("rgb_sha256") != hashlib.sha256(frame.rgb).hexdigest() or
            readback.get("sampled_source_pixels") != frame.width * frame.height or
            readback.get("pixel_origin") != "top-left" or result.get("window_system") != "Tk/X11" or
            result.get("close_reason") not in ("wm-close", "lifetime")):
        raise ValueError("window worker did not verify this frame's actual X11 RGB readback")
    bounded_integer(result.get("frames_presented"), "presented frames", frames)
    bounded_seconds(result.get("elapsed_seconds"), "window elapsed time", lifetime + 5)
    return {**report, **result, "mode": "window", "display": display, "hard_timeout_seconds": lifetime + 5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--report", type=Path, help="saved render.json or native-report.json; replay, not a fresh draw")
    inputs.add_argument("--raw", type=Path, help="raw RGBA8 with explicit width and height")
    inputs.add_argument("--capture", type=Path, help="fresh managed render_artifact execution of a saved draw")
    inputs.add_argument("--shader-directory", type=Path, help="fresh bounded tools.run_render capture and CPU render")
    parser.add_argument("--report-raw", type=Path, help="explicit raw companion to --report")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--origin", choices=("top-left", "bottom-left"))
    parser.add_argument("--build", type=Path)
    parser.add_argument("--target", choices=("legacy-mgcsim-v1", "fp16-hardware-v1"), default="legacy-mgcsim-v1")
    parser.add_argument("--render-timeout", type=float, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--lifetime", type=float, default=1)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output", type=Path, help="new JSON file; existing paths are never overwritten")
    parser.add_argument("--window-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        bounded_integer(args.frames, "frames", MAX_FRAMES)
        bounded_seconds(args.lifetime, "lifetime", MAX_LIFETIME)
        bounded_integer(args.scale, "scale", MAX_SCALE)
        if args.window_worker:
            if not re.fullmatch(r"(?:unix)?:[0-9]{1,5}(?:\.[0-9]{1,2})?", os.environ.get("DISPLAY", "")):
                raise ValueError("window worker requires an explicit local X11 DISPLAY")
            contents = sys.stdin.buffer.read(MAX_REPORT_BYTES + 1)
            if len(contents) > MAX_REPORT_BYTES:
                raise ValueError("window worker input exceeds bounded report bytes")
            payload = parse_json(contents)
            frame = Frame(payload["width"], payload["height"], base64.b64decode(payload["rgba"], validate=True))
            bounded_integer(payload["frames"], "frames", MAX_FRAMES)
            bounded_seconds(payload["lifetime"], "lifetime", MAX_LIFETIME)
            bounded_integer(payload["scale"], "scale", MAX_SCALE)
            result = _present_window(frame, frames=payload["frames"], lifetime=payload["lifetime"], scale=payload["scale"])
        else:
            if args.output is not None and os.path.lexists(args.output):
                raise ValueError("presentation output must be a new path")
            if args.report_raw is not None and args.report is None:
                raise ValueError("--report-raw requires --report")
            if args.capture or args.shader_directory:
                if args.origin is not None or args.report_raw is not None:
                    raise ValueError("origin/raw overrides are unsupported for generated frames")
                options = {"width": 16 if args.width is None else args.width,
                           "height": 16 if args.height is None else args.height, "timeout": args.render_timeout}
                frame = (render_capture(args.capture, **options) if args.capture else
                         render_shader(args.shader_directory, build=args.build, target=args.target, **options))
            else:
                frame = load_frame(report=args.report, raw=args.report_raw if args.report else args.raw,
                                   width=args.width, height=args.height, origin=args.origin)
            result = present_frame(frame, headless=args.headless, frames=args.frames,
                                   lifetime=args.lifetime, scale=args.scale)
            if args.output is not None:
                with args.output.open("x") as destination:
                    destination.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"schema": "mgc-host-presentation-v1", "passed": False, "displayed": False,
                          "scope": SCOPE, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
