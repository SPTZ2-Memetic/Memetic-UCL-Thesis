"""Data contract for memetic — the single place that answers
"what flows through this system?"

Two types propagate everywhere:

  Record  — a task/bug as it comes off the loader. It IS the task: it carries
            everything the verifier and fixer need (test, entry_point,
            code_prompt, ...). A plain TypedDict on purpose, so loader / seed /
            verifier stay dumb and pass dicts around with zero ceremony. The
            schema is documented here instead of living implicitly in scattered
            .get("...") calls.

  Case    — one memory episode (s, a, r). What the generator's / fixer's Memory
            stores and retrieves. seed.py builds these from Records.

Record is the input to seed.py; Case is its output. Nothing else is a
first-class type.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Optional, TypedDict
import time
import uuid

import numpy as np


# ---- Record: the task/bug dict (the "Task" of this system) ----------------
class Record(TypedDict):
    task_id: str
    entry_point: str
    code_prompt: str           # signature + docstring scaffold (imports, def, doc)
    canonical_solution: str    # BODY (4-space normalised); reconstruct = code_prompt + this
    buggy: str                 # BODY; "" for the seed pool (generator fills it)
    test: str                  # ground-truth tests the verifier runs
    instruct_prompt: str       # natural-language problem statement
    source: str                # "human" / "qwen7b" / ... or "bigcodebench_ours"
    split: str                 # "train" / "test" / "test_small"


# ---- Case: one memory episode (s, a, r) -----------------------------------
@dataclass
class Case:
    # --- retrieval key (this is what gets embedded) -----------------------
    # A short, canonical descriptor of the SEED situation. Option A: built from
    # the seed, so it's identical in shape at write time and read time.
    key_text: str

    # --- payload (raw, never embedded; fed to the LLM at reuse time) -------
    # s: the seed the generator received.
    question: str
    reference_solution: str
    # a: what the generator did.
    mutation: str            # short description of the bug introduced
    buggy_code: str          # the resulting buggy code
    # r: productivity label in {0, 1}. None until the fixer has scored it.
    reward: Optional[int] = None

    # --- routing / analysis metadata --------------------------------------
    source: str = "unknown"              # data_source tag (human, qwen7b, ...)
    solve_rate: Optional[float] = None   # fraction of K fixer attempts that passed
    task_id: Optional[str] = None
    step: Optional[int] = None
    origin: str = "live"                 # "warm_start" or "live"
    error_signature: str = ""           # compact failure sig of the bug (verifier)
    error_kind: str = ""                 # coarse failure bucket (index/key/assertion/...)
    explanation: str = ""                # short NL sentence describing the bug (LLM, given the diff)
    buggy_line: str = ""                 # the exact diff (before/after) -- from short_diff, deterministic
    from_exploration: bool = False       # made with NO memory shown (Skill-SP exploration stream);
                                          # only written to the bank if it cleared the frontier-reward gate
    construct_key: str = ""              # the specific canonical statement this bug targeted (from
                                          # constructs.changed_construct), for CROSS-TASK construct
                                          # matching -- empty if the bug didn't align as a clean local
                                          # edit, or if this was an INVALID bug (no real consequence
                                          # to transfer). Embedded on its own channel (see
                                          # Memory.read_by_constructs), separate from key_text.
    n_attempts: int = 0                  # MOVING-TARGET HARDNESS (Skill-SP Appendix B): times this
                                          # construct has been MATCHED and actually used for a new
                                          # generation. Updated PASSIVELY, as a byproduct of normal
                                          # construct-matching retrieval -- NOT from a separate re-test
                                          # step (see Memory.update_construct_stats). A frozen
                                          # solve_rate from creation time never reflects whether the
                                          # solver has since gotten better at this construct; a running
                                          # frontier-rate does.
    n_frontier: int = 0                  # of those n_attempts, how many resulted in a genuinely
                                          # PRODUCTIVE (0 < solve_rate < 1) bug, not a trivial or
                                          # unsolvable one -- expected_frontier_rate = n_frontier /
                                          # n_attempts is the Skill-SP-style moving target: a construct
                                          # that was hard when discovered but has since become
                                          # consistently trivial will visibly decay here as training
                                          # progresses, distinct from its original, frozen solve_rate.

    # --- system fields ----------------------------------------------------
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    # cached embedding of key_text (set on write); not serialized to JSON.
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # cached embedding of construct_key (set on write, if construct_key is
    # non-empty); not serialized to JSON.
    construct_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)   # embeddings persist separately (npy)
        d.pop("construct_embedding", None)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Case":
        d = dict(d)
        d.pop("embedding", None)
        d.pop("construct_embedding", None)
        # tolerate SAVED banks from an older schema (e.g. a removed field like
        # the old 'scope') -- drop anything that isn't a real field of the
        # CURRENT dataclass, rather than crashing. A field the old data lacks
        # simply uses its default; a field the old data has that we've since
        # removed is silently dropped.
        valid = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)

    def prompt_block(self) -> str:
        """How a retrieved case is rendered into a role's context."""
        verdict = {1: "PRODUCTIVE", 0: "UNPRODUCTIVE"}.get(self.reward, "UNSCORED")
        sr = "" if self.solve_rate is None else f" (solve_rate={self.solve_rate:.2f})"
        return (
            f"[past bug — source={self.source} — {verdict}{sr}]\n"
            f"mutation: {self.mutation}\n"
            f"buggy_code:\n{self.buggy_code}\n"
        )


# ---------------------------------------------------------------------------
# FixerCase — the fixer's memory unit. Deliberately a DIFFERENT shape from
# the generator's Case: the fixer's state (task+buggy code+CURRENT error),
# action (the fix), and reward semantics (binary pass/fail, no SESA bell --
# there is no "too easy/too hard" tension for fixing) are all different from
# the generator's. Two roles, two case shapes, coupled only via solve_rate.
# ---------------------------------------------------------------------------
@dataclass
class FixerCase:
    # --- retrieval keys (embedded): TWO SEPARATE channels ------------------
    # key_text = task description + buggy code (the SITUATION).
    # error_text = the compact error signature ONLY (the DEFECT TYPE).
    # Kept and embedded SEPARATELY (not concatenated) because the same code
    # can fail with different errors needing different fixes, and vice versa
    # -- a single combined embedding lets one channel's length dominate the
    # other in mean-pooling. Retrieval combines both similarity scores with
    # an explicit weight (see Memory.read's error_weight), so neither can
    # silently swamp the other the way concatenation did (see project notes
    # on the raw-traceback embedding bug this replaces).
    key_text: str
    error_text: str = ""
    buggy_construct_key: str = ""   # the SPECIFIC buggy statement(s) this fix
                                    # touched -- the "-" lines of fix_diff,
                                    # extracted directly (fix_diff is already
                                    # "-buggy_line +fixed_line", so no new
                                    # reconstruction or LLM call is needed).
                                    # A THIRD, EXPERIMENTAL retrieval channel
                                    # (see Memory.read_by_buggy_constructs):
                                    # matches the specific broken CONSTRUCT
                                    # instead of the whole buggy function,
                                    # the same principle already validated
                                    # for the generator (see project notes:
                                    # ~0.93 cosine for genuine cross-task
                                    # tactic matches vs ~0.20 unrelated, a
                                    # sharper split than whole-function
                                    # similarity gives).

    # --- payload (raw, shown to the model at reuse time) --------------------
    question: str = ""           # task description
    buggy_code: str = ""         # the buggy function as it stood for this attempt
    error: str = ""              # the RAW error this attempt was responding to (for
                                 # trace/display; error_text above is the compact
                                 # embeddable version, may differ)
    fix_diff: str = ""           # the EXACT fix made (deterministic, short_diff)
    explanation: str = ""        # short NL line, LLM-given-the-diff, SUCCESS attempts only
    passed: bool = False         # did this fix attempt pass verification
    reward: float = 0.0          # 1.0 if passed else 0.0 (binary, no bell)

    # --- routing / analysis metadata ----------------------------------------
    bug_case_id: Optional[str] = None   # links back to the generator Case for this bug
    task_id: Optional[str] = None
    attempt_idx: Optional[int] = None
    step: Optional[int] = None

    # --- system fields --------------------------------------------------------
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)      # key_text channel
    error_embedding: Optional[np.ndarray] = field(default=None, repr=False)  # error_text channel
    buggy_construct_embedding: Optional[np.ndarray] = field(default=None, repr=False)  # buggy_construct_key channel

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        d.pop("error_embedding", None)
        d.pop("buggy_construct_embedding", None)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "FixerCase":
        d = dict(d)
        d.pop("embedding", None)
        valid = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)
