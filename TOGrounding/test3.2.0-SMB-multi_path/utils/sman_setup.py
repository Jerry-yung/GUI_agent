"""Load SMAN-Bench/utils.py without shadowing project utils package."""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from types import ModuleType

from utils.paths import SMAN_BENCH_DIR

_sman_module: ModuleType | None = None


@lru_cache(maxsize=1)
def get_sman_utils() -> ModuleType:
    global _sman_module
    if _sman_module is not None:
        return _sman_module

    path = SMAN_BENCH_DIR / "utils.py"
    if not path.is_file():
        raise FileNotFoundError(f"SMAN utils not found: {path}")

    sman_dir = str(SMAN_BENCH_DIR)
    if sman_dir not in sys.path:
        sys.path.insert(0, sman_dir)

    spec = importlib.util.spec_from_file_location("sman_bench_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SMAN utils from {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["sman_bench_utils"] = mod
    spec.loader.exec_module(mod)
    _sman_module = mod
    return mod


def ensure_sman_path() -> None:
    """Backward-compatible no-op; SMAN utils loaded via importlib."""
    get_sman_utils()
