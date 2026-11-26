import os
import subprocess

import pytest

from test_gallium_driver import runtime_helper


@pytest.mark.parametrize("version", (None, "0", "1", "2", "01", "1.0", "+1", " 1", "1 ", ""))
@pytest.mark.parametrize("target", (None, "legacy-mgcsim-v1", "fp16-hardware-v1", "unknown"))
def test_instance_caps_are_explicit_legacy_host_only(runtime_helper, version, target):
    environment = os.environ.copy()
    for name in ("MGC_HOST_INSTANCING", "MGC_COMPILE_TARGET", "MGC_HOST_TEXTURE",
                 "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T"):
        environment.pop(name, None)
    if version is not None:
        environment["MGC_HOST_INSTANCING"] = version
    if target is not None:
        environment["MGC_COMPILE_TARGET"] = target
    result = subprocess.run([str(runtime_helper), "display_formats"], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
