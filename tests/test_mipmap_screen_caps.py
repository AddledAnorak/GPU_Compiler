import os
import subprocess

import pytest

from test_gallium_driver import runtime_helper


@pytest.mark.parametrize("version", (None, "0", "1", "6", "7", "8", "9", "10", "11",
                                      "", "09", "9.0", "+9", " 9", "9 "))
@pytest.mark.parametrize("target", (None, "legacy-mgcsim-v1", "fp16-hardware-v1", "unknown"))
def test_fragment_lod_cap_is_explicit_legacy_mipmap_only(runtime_helper, version, target):
    environment = os.environ.copy()
    for name in ("MGC_HOST_TEXTURE", "MGC_COMPILE_TARGET", "MGC_HOST_INSTANCING",
                 "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T"):
        environment.pop(name, None)
    if version is not None:
        environment["MGC_HOST_TEXTURE"] = version
    if target is not None:
        environment["MGC_COMPILE_TARGET"] = target
    result = subprocess.run([str(runtime_helper), "display_formats"], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
