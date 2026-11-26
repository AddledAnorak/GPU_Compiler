from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def local_hardware_examples():
    examples = Path(__file__).resolve().parents[1] / "hardware-examples/aakash-2026-09-28"
    if not examples.is_dir():
        pytest.skip("original hardware examples are not included in the source checkout")
    return examples
