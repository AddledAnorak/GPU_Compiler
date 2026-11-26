"""Parsing/headless contracts and optional real Xvfb drawable/close checks."""

import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from tools import host_present as present


RGBA = bytes((255, 0, 1, 0, 2, 255, 3, 255, 4, 5, 255, 17, 250, 240, 230, 128))


def managed_report(directory, rgba=RGBA, **overrides):
    report = {"pipeline": "managed VS -> CPU raster -> managed FS -> RGBA8", "fpga_execution": False,
              "width": 2, "height": 2, "pixel_origin": "exported top-left; raster coordinates bottom-left",
              "rgba_sha256": hashlib.sha256(rgba).hexdigest(), **overrides}
    path = directory / "render.json"
    path.write_text(json.dumps(report))
    (directory / "frame.rgba8").write_bytes(rgba)
    return path


def test_top_left_rgba_and_opaque_ppm_preserve_channel_order():
    frame = present.Frame(2, 2, RGBA)
    expected = bytes((255, 0, 1, 2, 255, 3, 4, 5, 255, 250, 240, 230))
    assert frame.rgb == expected
    assert frame.ppm() == b"P6\n2 2\n255\n" + expected
    assert frame.rgba_sha256 == hashlib.sha256(RGBA).hexdigest()


@pytest.mark.parametrize("width,height", ((0, 1), (-1, 1), (129, 1), (1, 129), (True, 1), (1, 1.0), ("2", 2)))
def test_frame_rejects_unbounded_or_coerced_dimensions(width, height):
    with pytest.raises(ValueError, match="width|height"):
        present.Frame(width, height, RGBA)


@pytest.mark.parametrize("rgba", (b"", RGBA[:-1], RGBA + b"x", bytearray(RGBA), [0] * 16))
def test_frame_requires_exact_byte_storage(rgba):
    with pytest.raises(ValueError, match="RGBA8"):
        present.Frame(2, 2, rgba)


def test_saved_managed_digest_and_provenance_are_not_fresh(tmp_path):
    report = managed_report(tmp_path)
    frame = present.load_frame(report=report)
    assert frame.rgba == RGBA
    assert frame.source["kind"] == "saved-managed-render"
    assert frame.source["fresh_gl_draw"] is False
    assert frame.source["fresh_host_render"] is False
    assert frame.source["raw"]["sha256"] == frame.rgba_sha256
    assert frame.source["report"]["path"] == str(report)


def test_native_bottom_left_raw_is_flipped_exactly_once(tmp_path):
    raw = tmp_path / "native.rgba8"
    raw.write_bytes(RGBA[8:] + RGBA[:8])
    report = tmp_path / "native-report.json"
    report.write_text(json.dumps({"fpga_execution": False, "native_gl_readback": True,
                                 "host_executor_committed": True, "width": 2, "height": 2,
                                 "format": "R8G8B8A8_UNORM", "pixel_origin": "top-left",
                                 "raw_pixel_origin": "bottom-left", "raw_path": raw.name}))
    frame = present.load_frame(report=report)
    assert frame.rgba == RGBA
    assert frame.source["kind"] == "saved-native-readback"
    assert frame.source["fresh_gl_draw"] is False
    assert frame.source["raw"]["sha256"] != frame.rgba_sha256
    with pytest.raises(ValueError, match="origin conflicts"):
        present.load_frame(report=report, origin="top-left")


@pytest.mark.parametrize("overrides", ({"fpga_execution": True}, {"passed": False}, {"status": "running"},
                                      {"pixel_origin": "unknown"}, {"rgba_sha256": "0" * 64},
                                      {"rgba_sha256": None}, {"width": True}, {"pipeline": "unidentified"},
                                      {"pixel_origin": "bottom-left"}, {"physical_gpu_execution": True},
                                      {"softpipe_comparison": {"passed": False}}))
def test_failed_ambiguous_or_mismatched_reports_fail_closed(tmp_path, overrides):
    report = managed_report(tmp_path, **overrides)
    with pytest.raises(ValueError):
        present.load_frame(report=report)


@pytest.mark.parametrize("options", ({"width": 3}, {"height": 3}, {"width": True}, {"origin": "bottom-left"}))
def test_report_overrides_cannot_silently_change_geometry_or_origin(tmp_path, options):
    report = managed_report(tmp_path)
    with pytest.raises(ValueError):
        present.load_frame(report=report, **options)


def test_raw_needs_geometry_and_exact_regular_file(tmp_path):
    raw = tmp_path / "input.rgba8"
    raw.write_bytes(RGBA)
    with pytest.raises(ValueError, match="width"):
        present.load_frame(raw=raw)
    assert present.load_frame(raw=raw, width=2, height=2).rgba == RGBA
    raw.write_bytes(RGBA[:-1])
    with pytest.raises(ValueError, match="RGBA8"):
        present.load_frame(raw=raw, width=2, height=2)
    raw.write_bytes(RGBA + b"x")
    with pytest.raises(ValueError, match="bounded file"):
        present.load_frame(raw=raw, width=2, height=2)


def test_symlink_fifo_and_nonfinite_json_are_rejected_without_blocking(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(RGBA)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        present.read_bounded(link, 16)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="ordinary bounded"):
        present.read_bounded(fifo, 16)
    with pytest.raises(ValueError, match="nonfinite"):
        present.parse_json('{"duration": NaN}')


@pytest.mark.parametrize("contents", ('{"duration": 1e309}', '{"passed": false, "passed": true}'))
def test_overflow_or_duplicate_json_fields_fail_closed(contents):
    with pytest.raises(ValueError, match="nonfinite|duplicate"):
        present.parse_json(contents)


def test_headless_never_imports_tk_or_claims_a_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setitem(sys.modules, "tkinter", None)
    report = present.present_frame(present.Frame(2, 2, RGBA), headless=True)
    assert report["passed"] is True
    assert report["displayed"] is False and report["readback"] is None
    assert report["frames_presented"] == 0 and report["window_closed"] is None
    assert report["fresh_gl_draw"] is False and report["fpga_execution"] is False


@pytest.mark.parametrize("display", (None, "", "remote.example:0", "localhost:0", ":1;touch /tmp/unsafe"))
def test_missing_or_nonlocal_display_fails_closed(monkeypatch, display):
    if display is None:
        monkeypatch.delenv("DISPLAY", raising=False)
    else:
        monkeypatch.setenv("DISPLAY", display)
    with pytest.raises(ValueError, match="local X11 DISPLAY"):
        present.present_frame(present.Frame(2, 2, RGBA))


@pytest.mark.parametrize("options", ({"frames": 0}, {"frames": 121}, {"frames": True},
                                     {"lifetime": float("nan")}, {"lifetime": float("inf")},
                                     {"lifetime": 0}, {"lifetime": 31}, {"lifetime": True},
                                     {"scale": 0}, {"scale": 9}, {"headless": 1}))
def test_presentation_bounds_apply_in_headless_mode(options):
    with pytest.raises(ValueError):
        present.present_frame(present.Frame(2, 2, RGBA), **{"headless": True, **options})


def test_maximum_headless_bounds_are_explicit_and_finite():
    frame = present.Frame(128, 128, bytes(128 * 128 * 4))
    report = present.present_frame(frame, headless=True, frames=120, lifetime=30, scale=8)
    assert report["frames_requested"] == 120 and report["lifetime_seconds"] == 30
    assert report["displayed"] is False


def test_render_capture_uses_existing_deadline_api_without_fresh_gl_claim(tmp_path, monkeypatch):
    capture = tmp_path / "draw.json"
    capture.write_text('{"unit_test_capture": true}')
    calls = []

    def render_artifact(artifact, **options):
        calls.append((artifact, options))
        return np.frombuffer(RGBA, dtype=np.uint8).reshape(2, 2, 4), None, {"unit_test_renderer": True}

    monkeypatch.setattr("tools.host_render.render_artifact", render_artifact)
    frame = present.render_capture(capture, width=2, height=2, timeout=0.5)
    assert calls == [({"unit_test_capture": True}, {"width": 2, "height": 2, "execution_timeout": 0.5})]
    assert frame.rgba == RGBA
    assert frame.source["fresh_host_render"] is True and frame.source["fresh_gl_draw"] is False


def test_fresh_shader_uses_new_temporary_renderer_output(tmp_path, monkeypatch):
    calls = []

    def run(command, **options):
        directory = Path(command[command.index("--output") + 1])
        assert not (directory / "render.json").exists()
        managed_report(directory)
        calls.append((command, options))
        return b"unit-test-only renderer output"

    monkeypatch.setattr(present, "_subprocess", run)
    frame = present.render_shader(tmp_path, width=2, height=2, timeout=0.5)
    assert calls[0][0][1:3] == ["-m", "tools.run_render"]
    assert calls[0][1] == {"timeout": 0.5}
    assert frame.source["fresh_gl_draw"] is True
    assert frame.source["temporary_renderer_files_retained"] is False
    assert not Path(frame.source["raw"]["path"]).exists()


def test_window_subprocess_has_a_hard_timeout():
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="hard deadline"):
        present._subprocess([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1)
    assert time.monotonic() - started < 3


def test_worker_success_booleans_without_matching_drawable_readback_are_rejected(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(present, "_subprocess", lambda *args, **options: b'{"displayed": true, "window_closed": true}')
    with pytest.raises(ValueError, match="actual X11 RGB readback"):
        present.present_frame(present.Frame(2, 2, RGBA))


def test_unreachable_local_display_is_bounded_and_not_claimed_displayed(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":65534")
    started = time.monotonic()
    with pytest.raises(ValueError, match="Tk/X11 display"):
        present.present_frame(present.Frame(2, 2, RGBA), lifetime=0.1)
    assert time.monotonic() - started < 6


def test_cli_headless_and_existing_output_preservation(tmp_path, capsys):
    report = managed_report(tmp_path)
    output = tmp_path / "presentation.json"
    assert present.main(["--report", str(report), "--headless", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["displayed"] is False
    original = output.read_bytes()
    assert present.main(["--report", str(report), "--headless", "--output", str(output)]) == 1
    assert output.read_bytes() == original
    assert json.loads(capsys.readouterr().err)["passed"] is False


def test_cli_missing_display_reports_failure_not_headless_success(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DISPLAY", raising=False)
    report = managed_report(tmp_path)
    assert present.main(["--report", str(report)]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["passed"] is False and error["displayed"] is False


@pytest.mark.parametrize("mask,pixel,expected", ((0xFF0000, 0xAB1203, 171), (0xFF00, 0xAB1203, 18),
                                               (0xFF, 0xAB1203, 3)))
def test_x11_channel_masks_are_not_native_byte_order(mask, pixel, expected):
    assert present._channel(pixel, mask) == expected


@pytest.fixture(scope="module")
def xvfb_display(tmp_path_factory):
    executable = shutil.which("Xvfb")
    if executable is None:
        pytest.skip("real window validation unavailable: Xvfb executable is not installed; no display claim")
    pytest.importorskip("tkinter", reason="real window validation requires Tk")
    directory = tmp_path_factory.mktemp("host-present-xvfb")
    read_fd, write_fd = os.pipe()
    with (directory / "xvfb.log").open("wb") as log:
        process = subprocess.Popen([executable, "-displayfd", str(write_fd), "-screen", "0", "1024x1024x24",
                                    "-nolisten", "tcp"], pass_fds=(write_fd,), stdout=log, stderr=log)
        os.close(write_fd)
        try:
            if not select.select([read_fd], [], [], 10)[0]:
                pytest.fail("installed Xvfb failed to report a display within 10s")
            number = os.read(read_fd, 32).decode("ascii").strip()
            if not number.isdigit():
                pytest.fail("installed Xvfb returned no valid display")
            yield f":{number}"
        finally:
            os.close(read_fd)
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def test_actual_xvfb_readback_pixel_order_and_lifetime(xvfb_display, monkeypatch):
    monkeypatch.setenv("DISPLAY", xvfb_display)
    frame = present.Frame(2, 2, RGBA)
    report = present.present_frame(frame, frames=3, lifetime=0.3, scale=2)
    assert report["displayed"] is True and report["window_closed"] is True
    assert report["close_reason"] == "lifetime"
    assert 1 <= report["frames_presented"] <= 3
    assert report["readback"]["mismatch_pixels"] == 0
    assert report["readback"]["rgb_sha256"] == hashlib.sha256(frame.rgb).hexdigest()
    assert report["readback"]["backend"] == "XGetImage/XGetPixel"


def test_actual_xvfb_wm_delete_protocol_closes_early(xvfb_display, monkeypatch):
    import tkinter

    monkeypatch.setenv("DISPLAY", xvfb_display)
    factory = tkinter.Tk

    def closeable_root():
        root = factory()
        root.after(150, lambda: root.tk.eval(root.protocol("WM_DELETE_WINDOW")))
        return root

    monkeypatch.setattr(tkinter, "Tk", closeable_root)
    report = present._present_window(present.Frame(2, 2, RGBA), frames=20, lifetime=1, scale=1)
    assert report["displayed"] is True and report["window_closed"] is True
    assert report["close_reason"] == "wm-close"
    assert 1 <= report["frames_presented"] < 20 and report["elapsed_seconds"] < 1


def test_actual_xvfb_readback_mismatch_fails_closed(xvfb_display, monkeypatch):
    monkeypatch.setenv("DISPLAY", xvfb_display)
    monkeypatch.setattr(present, "readback_x11", lambda *args: bytes(12))
    with pytest.raises(ValueError, match="actual X11 readback differs"):
        present._present_window(present.Frame(2, 2, RGBA), frames=1, lifetime=0.2, scale=1)
