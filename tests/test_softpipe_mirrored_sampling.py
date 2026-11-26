"""Real softpipe-only mirrored sampling, independent of the MGC compiler/CPU."""

import os
from pathlib import Path
import struct

import numpy as np
import pytest

from tools import mirrored_texture_acceptance as acceptance
from tools.mgcdriver import Driver
from tools.run_render import softpipe_reference
from tools.run_shader import ROOT


BUILD = Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
NAMES = {
    "linear-mirrored-repeat-seams-4x4",
    "linear-mirrored-repeat-negative-seams-3x5",
    "linear-mirrored-repeat-even-odd-seams-5x3",
    "linear-mirrored-repeat-multiple-periods-3x5",
    "linear-mirrored-repeat-early-kill-4x4",
    "linear-mirrored-repeat-2x2",
    "linear-mirrored-repeat-zero-alpha-1x1",
    "nearest-mirrored-repeat-seams-4x4",
    "nearest-mirrored-repeat-multiple-periods-3x5",
}
CASES = tuple(case for case in acceptance.CASES if case.name in NAMES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_real_softpipe_reverse_weights_include_exact_texel_centers(tmp_path, monkeypatch, case):
    if not (BUILD / "build.ninja").is_file():
        pytest.skip("configured Mesa required for actual softpipe GL sampling")

    def unexpected(*args, **kwargs):
        raise AssertionError("independent softpipe regression must never execute MGC programs")

    monkeypatch.setattr(Driver, "submit", unexpected)
    shader = acceptance._shader(case, tmp_path)
    mesh = tmp_path / "vertices.bin"
    mesh.write_bytes(acceptance.case_mesh(case))
    texels = acceptance.input_texels(case.width, case.height)
    texture = tmp_path / "texture.bin"
    texture.write_bytes(struct.pack("<II", case.width, case.height) + texels.tobytes())
    reference = softpipe_reference(shader, BUILD, tmp_path / "reference", mesh=mesh, texture=texture,
                                   texture_filter=case.texture_filter, texture_wrap="mirrored-repeat")
    expected, coverage = acceptance.rational_oracle(texels, phase=case.phase,
                                                   texture_filter=case.texture_filter,
                                                   discard=case.variant != "sample")
    difference = np.abs(reference.astype(np.int16) - expected.astype(np.int16))
    assert int(difference.max()) <= (1 if case.texture_filter == "linear" else 0)
    assert "GL_RENDERER: softpipe" in (tmp_path / "reference/softpipe.log").read_text()
    assert np.all(coverage[:, 8:] == 1)
