"""Central data directory configuration for Mobile3M pipeline."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "Mobile3M_data"

_default_mobile3m = (PROJECT_ROOT / "../../datasets/Mobile3M/datasets").resolve()
MOBILE3M_SRC = Path(os.environ.get("MOBILE3M_SRC", str(_default_mobile3m))).resolve()
