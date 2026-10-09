"""BugBench loader (self-contained) — cchoi1 BugSourceBench (HF/parquet) -> records.

Stripped to what THIS system needs: no Task/Anchor classes, no anchor-pool
machinery (memory replaces anchoring here).

What it emits: plain RECORD dicts. A record is simultaneously
  - "the task"  — carries test / entry_point / code_prompt, everything the
    verifier and fixer read; and
  - the input to seed.py, which extracts the compact key + the (s,a,r) payload.
seed.py already consumes plain dicts, so no wrapper type is needed.

Two entry points:
  load_seed_pool(split)        -> clean tasks to break (bigcodebench_ours; buggy="")
  load_bugbench_records(src, split) -> real published bugs (for warm-start; has buggy)

Bodies: stored solutions are turned into 4-space-normalised FUNCTION BODIES via a
self-contained AST extractor, so `code_prompt + body` reconstructs the function
(the convention seed.py / the verifier rely on). Unrecoverable rows are dropped.

Data source: prefers a local parquet mirror
  <memetic_DATA_DIR or ./asp_datasets>/<repo_name>/<split>-00000-of-00001.parquet
and falls back to the HF Hub. Both paths yield identical row dicts.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap
from typing import Any, Dict, Iterable, List, Optional

from memetic.types import Record


# ---- repos ---------------------------------------------------------------
# The four paper bug sources (warm-start material) + the clean training pool
# the generator breaks. Namespace cchoi1.
BUGBENCH_REPOS: Dict[str, str] = {
    "human":            "cchoi1/bugbench_v2",
    "human_edited_lm":  "cchoi1/bugs_human_edited_lm",
    "qwen7b":           "cchoi1/bugs_lm_qwen7b",
    "gpt_oss_20b":      "cchoi1/bugs_lm_gpt_oss_20b",
    "adversarial":      "cchoi1/bugbench_adversarial_new",
}
SEED_POOL_REPO = "cchoi1/bigcodebench_ours"

DEFAULT_SOURCES = ("human", "human_edited_lm", "qwen7b", "gpt_oss_20b")

_DATA_DIR = os.environ.get(
    "memetic_DATA_DIR", os.path.join(os.getcwd(), "asp_datasets")
)


def _local_parquet(repo_id: str, split: str) -> Optional[str]:
    name = repo_id.split("/")[-1]
    cand = os.path.join(_DATA_DIR, name, f"{split}-00000-of-00001.parquet")
    return cand if os.path.exists(cand) else None


def _read_rows(repo_id: str, split: str, hf_token: Optional[str] = None) -> List[dict]:
    """Local parquet if present, else HF Hub. Returns list of row dicts."""
    local = _local_parquet(repo_id, split)
    if local is not None:
        import pyarrow.parquet as pq
        return pq.read_table(local).to_pylist()
    from datasets import load_dataset
    return list(load_dataset(repo_id, split=split, token=hf_token))


# ---- body extraction (shared) -------------------------------------------
from memetic.codeutils import (
    strip_fence as _strip_fence,
    reindent as _reindent,
    extract_function_body,
)


def _to_body(stored_solution: str, entry_point: str) -> Optional[str]:
    """Stored full/partial solution -> 4-space-normalised body. Handles fenced
    code and bare bodies (no def line) by wrapping then re-extracting."""
    src = _strip_fence(stored_solution)
    if not src:
        return None
    body = extract_function_body(src, entry_point)
    if body is not None:
        return body
    wrapped = f"def {entry_point}():\n" + _reindent(src, 4) + "\n"
    return extract_function_body(wrapped, entry_point)


# ---- row -> record -------------------------------------------------------
def _row_to_record(row: Dict[str, Any], source: str, split: str,
                   *, need_bug: bool) -> Optional[Record]:
    entry_point = row.get("entry_point") or "task_func"
    canonical = _to_body(row.get("reference_solution", ""), entry_point)
    if canonical is None:
        return None                        # no correct body -> can't use it

    buggy = ""
    if need_bug:
        buggy = _to_body(row.get("buggy_solution", ""), entry_point)
        if buggy is None:
            return None                    # warm-start row must have a real bug

    return {
        "task_id": str(row.get("uid") or row.get("task_id") or ""),
        "entry_point": entry_point,
        "code_prompt": row.get("code_prompt", "") or "",
        "canonical_solution": canonical,   # BODY
        "buggy": buggy,                    # BODY ("" for seed pool)
        "test": row.get("ground_truth", "") or "",
        "instruct_prompt": (row.get("instruct_prompt")
                            or row.get("complete_prompt") or ""),
        "source": source,
        "split": split,
    }


# ---- public API ----------------------------------------------------------
def load_bugbench_records(
    source: str,
    split: str,
    *,
    limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[Record]:
    """Real published bugs from one source/split (warm-start material).
    Records have a real `buggy` body. `source` is a key in BUGBENCH_REPOS or a
    full repo id."""
    repo_id = BUGBENCH_REPOS.get(source, source)
    rows = _read_rows(repo_id, split, hf_token)
    out: List[Record] = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        rec = _row_to_record(row, source, split, need_bug=True)
        if rec is not None:
            out.append(rec)
    return out


def load_seed_pool(
    split: str = "train",
    *,
    limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[Record]:
    """Clean tasks the generator breaks (bigcodebench_ours). No bug (`buggy=""`);
    the generator synthesises it. Same record shape as warm-start records."""
    rows = _read_rows(SEED_POOL_REPO, split, hf_token)
    out: List[Record] = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        rec = _row_to_record(row, "bigcodebench_ours", split, need_bug=False)
        if rec is not None:
            out.append(rec)
    return out


def load_warmstart(
    sources: Iterable[str] = DEFAULT_SOURCES,
    split: str = "train",
    *,
    per_source: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[Record]:
    """All warm-start bug records across the given sources, concatenated. Feed
    each through seed.record_to_warmstart_case to populate a Memory."""
    out: List[Record] = []
    for src in sources:
        out.extend(load_bugbench_records(src, split, limit=per_source, hf_token=hf_token))
    return out
