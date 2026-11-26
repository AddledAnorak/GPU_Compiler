"""Direct-script entry point; equivalent to PYTHONPATH=tools python -m mgcdriver."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mgcdriver.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
