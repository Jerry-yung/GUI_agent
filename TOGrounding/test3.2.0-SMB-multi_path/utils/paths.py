"""Path helpers for test3.2.0-SMB multi-path."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMAN_BENCH_DIR = ROOT / "SMAN-Bench"
DEFAULT_DATA_DIR = (ROOT / "../../datasets/Mobile3M/datasets").resolve()

CACHE_DIR = ROOT / "cache"
CACHE_LABELED = CACHE_DIR / "labeled"
CACHE_NODES = CACHE_DIR / "nodes"
CACHE_EMBEDDINGS = CACHE_DIR / "embeddings"
CACHE_RETRIEVAL = CACHE_DIR / "retrieval"

RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"


def ensure_cache_dirs() -> None:
    for d in (CACHE_LABELED, CACHE_NODES, CACHE_EMBEDDINGS, CACHE_RETRIEVAL):
        d.mkdir(parents=True, exist_ok=True)


def instruction_query_digest(query_text: str) -> str:
    """Same digest as ``embed_instruction`` cache key (md5 prefix)."""
    text = (query_text or "").strip()
    if not text:
        return "empty"
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is None:
        return DEFAULT_DATA_DIR
    p = Path(data_dir)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def result_dir_name(task_type: str, agent: str, model_slug: str) -> str:
    return f"{task_type}_{agent}_{model_slug}"


def result_dir(task_type: str, agent: str, model_slug: str) -> Path:
    return RESULTS_DIR / result_dir_name(task_type, agent, model_slug)


def typed_result_dir(
    task_type: str,
    agent: str,
    model_slug: str,
    gt_type: str,
) -> Path:
    """``results/{task_type}_{agent}_{model}/{gt_type}/``"""
    return result_dir(task_type, agent, model_slug) / gt_type.strip().lower()


def page_paths(graph_dir: Path, page_name: str) -> dict[str, Path]:
    page_dir = graph_dir / page_name
    return {
        "dir": page_dir,
        "screenshot": page_dir / f"{page_name}-screen.png",
        "html": page_dir / f"{page_name}-html.txt",
        "xml": page_dir / f"{page_name}-xml.txt",
        "json": page_dir / f"{page_name}.json",
    }


def cache_labeled_file(
    task_name: str, page_name: str, *, top_k: int, suffix: str = ".png"
) -> Path:
    """cache/labeled/top_{top_k}/{task_name}/{page_name}.png"""
    return CACHE_LABELED / f"top_{top_k}" / task_name / f"{page_name}{suffix}"


def cache_nodes_file(task_name: str, page_name: str, *, top_k: int) -> Path:
    """cache/nodes/top_{top_k}/{task_name}/{page_name}.json"""
    return CACHE_NODES / f"top_{top_k}" / task_name / f"{page_name}.json"


def cache_retrieval_file(
    task_name: str,
    page_name: str,
    *,
    top_k: int,
    query_text: str,
) -> Path:
    """cache/retrieval/top_{top_k}/{task_name}/{page_name}/{to_digest}.json"""
    digest = instruction_query_digest(query_text)
    return CACHE_RETRIEVAL / f"top_{top_k}" / task_name / page_name / f"{digest}.json"


def nodes_emb_step_dir(task_name: str, page_name: str) -> Path:
    """cache/embeddings/{task_name}/{page_name}/"""
    return CACHE_EMBEDDINGS / task_name / page_name
