"""Seed-descriptor builder: BugBench payload -> Case (key + payload).

Bridges the body-form Task payload to our Memory. Locked-in decisions:

  - Payload field names: entry_point, code_prompt,
    canonical_solution (BODY), buggy (BODY), instruct_prompt, test, source, split.
  - reference_solution the generator mutates = code_prompt + canonical_solution
    (the loader's own reconstruction convention).
  - question the generator reads = instruct_prompt (falls back to code_prompt).
  - Option A retrieval: key_text is built from the SEED only (signature + short
    spec + target-source tag), so read-time and write-time keys share one space.
    We embed this compact string, NEVER the raw code (no truncation / washout).
  - target_source is the difficulty lever. For LIVE generation it's the source we
    ask the generator to imitate. For WARM-START it's the bug's real source.

Standalone: operates on plain dicts. Accepts either a Task object (uses
.payload / .id) or a bare payload dict.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Optional

from memetic.types import Case, Record


# ---- payload coercion ----------------------------------------------------
def _as_payload(task_or_payload: Any) -> dict:
    p = getattr(task_or_payload, "payload", None)
    return dict(p) if isinstance(p, dict) else dict(task_or_payload)


def _task_id(task_or_payload: Any, payload: dict) -> Optional[str]:
    tid = getattr(task_or_payload, "id", None)
    return tid or payload.get("task_id") or payload.get("uid")


# ---- field extraction ----------------------------------------------------
def reconstruct_function(payload: dict, body_key: str = "canonical_solution") -> str:
    """code_prompt + body -> the full runnable function (loader convention)."""
    cp = (payload.get("code_prompt") or "").rstrip("\n")
    body = payload.get(body_key) or ""
    if not cp:
        return body.strip("\n")
    return (cp + "\n" + body).strip("\n")


def seed_question(payload: dict) -> str:
    return (payload.get("instruct_prompt") or payload.get("code_prompt") or "").strip()


def seed_reference(payload: dict) -> str:
    return reconstruct_function(payload, "canonical_solution")


def _signature_line(code_prompt: str, entry_point: str) -> str:
    for ln in (code_prompt or "").splitlines():
        s = ln.strip()
        if s.startswith(f"def {entry_point}") or (s.startswith("def ") and entry_point in s):
            return s.rstrip(":")
    return f"def {entry_point}(...)"


def _first_sentence(text: str, cap: int = 160) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    return s[:cap].strip()


def _short_spec(payload: dict) -> str:
    ip = payload.get("instruct_prompt") or ""
    if ip.strip():
        return _first_sentence(ip)
    # else pull the first docstring line out of code_prompt
    cp = payload.get("code_prompt") or ""
    m = re.search(r'"""(.*?)(?:"""|$)', cp, re.DOTALL)
    if m:
        return _first_sentence(m.group(1))
    return ""


def build_key_text(task_or_payload: Any, steer_source: Optional[str] = None) -> str:
    """The STATE, embedded for retrieval -- Memento-faithful: embed the whole
    situation (task description + canonical solution), not a compact
    signature. Same builder at read and write time, so cosine(query_state,
    stored_state) is meaningful. This REPLACES the earlier compact
    signature+spec key (which under-represented the code) and the AST
    construct-matching approach (shelved -- see project notes): retrieval is
    now simple whole-state similarity, ranked by reward among near neighbors,
    exactly as Memento does for its (text) tasks, adapted to code by embedding
    the full canonical instead of a short summary.

    steer_source is OFF by default -> the key is source-less, so warm-start and
    live cases share one key space and there is no self-preference bias in
    retrieval. Passing steer_source prepends a source tag (steering hook,
    unused in the plain loop)."""
    p = _as_payload(task_or_payload)
    question = seed_question(p)
    reference = seed_reference(p)
    base = f"{question}\n{reference}".strip()
    return f"source:{steer_source}\n{base}" if steer_source else base


# ---- diff for the mutation field ----------------------------------------
def short_diff(canonical_body: str, buggy_body: str, max_lines: int = 10) -> str:
    diff = difflib.unified_diff(
        (canonical_body or "").splitlines(),
        (buggy_body or "").splitlines(),
        lineterm="", n=1,
    )
    changed = [l for l in diff
               if l and l[0] in "+-" and not l.startswith(("+++", "---"))]
    if not changed:
        return "no textual change"
    return "\n".join(changed[:max_lines])


# ---- Case builders -------------------------------------------------------
def seed_to_case(
    task_or_payload: Any,
    *,
    buggy_code: str,
    mutation: str,
    source: str,
    steer_source: Optional[str] = None,
    reward: Optional[int] = None,
    solve_rate: Optional[float] = None,
    step: Optional[int] = None,
    origin: str = "live",
) -> Case:
    """Build a generator Case from a seed + the bug that was made from it.

    reward is left None for the live path (fixer scores it later via
    Memory.update_reward). buggy_code is stored raw as payload."""
    p = _as_payload(task_or_payload)
    return Case(
        key_text=build_key_text(task_or_payload, steer_source),
        question=seed_question(p),
        reference_solution=seed_reference(p),
        mutation=mutation,
        buggy_code=buggy_code,
        reward=reward,
        source=source,
        solve_rate=solve_rate,
        task_id=_task_id(task_or_payload, p),
        step=step,
        origin=origin,
    )


def record_to_warmstart_case(record: Record) -> Case:
    """Turn a real BugSourceBench record (from load_bugbench_records) into a
    warm-start Case. It's a published, valid bug, so reward=1 by construction.
    Keyed on its seed with its REAL source as the target tag -> lands in the same
    key space as live cases, and steers the generator toward that source's style.
    """
    buggy_full = reconstruct_function(record, "buggy")
    mutation = short_diff(record.get("canonical_solution", ""), record.get("buggy", ""))
    return seed_to_case(
        record,
        buggy_code=buggy_full,
        mutation=mutation,
        source=record.get("source", "unknown"),
        reward=1,
        origin="warm_start",
    )



# ---- construct-key derivation (decision A) --------------------------------
def _changed_lines(mutation: str, max_lines: int = 6) -> str:
    """The added/removed code lines from a unified-diff mutation, import noise
    dropped -- the 'construct' that was edited. Small, embeddable."""
    lines = []
    for ln in (mutation or "").splitlines():
        if not ln or ln[0] not in "+-":
            continue
        body = ln[1:].strip()
        if not body or body.startswith(("import ", "from ")):
            continue
        lines.append(body)
    return " ; ".join(lines[:max_lines])





# ---- bug explanation (LLM, given the diff -- not asked to find it) --------
def explain_bug_tactic(diff: str, policy) -> str:
    """GENERATOR-scoped generalized skill explanation, distinct from
    explain_bug (which produces a SPECIFIC-instance caption) and from
    memetic.fixer.explain_bug_general (fixer-scoped, differently
    templated). Asks for a STRUCTURED, TRANSFERABLE tactic: an ACTION applied
    to a general CODE PATTERN, producing a general kind of ERROR -- described
    abstractly enough that a completely DIFFERENT task with a similar code
    pattern would recognize it, not tied to this specific function/variables.

    This is the natural-language half of construct-matching: the matched
    CONSTRUCT localizes WHERE a similar pattern exists in a new task; this
    explanation supplies WHAT tactic to apply there, generalized so it
    transfers across genuinely different tasks that happen to share a
    similar code shape (see project notes: this is the direct fix for
    self-play repeatedly rediscovering the SAME bug on the SAME task rather
    than a tactic that generalizes ACROSS tasks)."""
    if not diff.strip():
        return ""
    prompt = (
        "A bug was introduced into a function by this exact change (unified "
        "diff, '-' = removed, '+' = added):\n\n"
        f"{diff}\n\n"
        "Describe this bug as a GENERAL, TRANSFERABLE tactic, in the form:\n"
        "\"[the general action/transformation] applied to [a general code "
        "pattern or operation, described ABSTRACTLY -- not this function's "
        "specific names] introduces [the general kind of error/consequence].\"\n\n"
        "Phrase it so it would apply to a DIFFERENT task that happens to use "
        "a SIMILAR code pattern or library call, not just this exact code. "
        "Do NOT mention this function's specific variable names, or restate "
        "the code verbatim. Reply with only the one sentence."
    )
    try:
        out = policy.generate(prompt).strip().splitlines()
        line = next((l.strip() for l in out if l.strip()), "")
        return line[:220]
    except Exception:
        return ""


def explain_bug(diff: str, policy) -> str:
    """One short natural-language sentence describing the bug, given the EXACT
    diff we already computed deterministically (short_diff). The model is never
    asked to find the change -- only to describe it -- so it can't misquote or
    paraphrase the code; it only generates the one thing that needs language."""
    if not diff.strip():
        return ""
    prompt = (
        "A bug was introduced into a function by this exact change "
        "(unified diff, '-' = removed, '+' = added):\n\n"
        f"{diff}\n\n"
        "In ONE short sentence, describe what this bug does or why it is wrong. "
        "Do not restate the code. Reply with only the sentence."
    )
    try:
        out = policy.generate(prompt).strip().splitlines()
        line = next((l.strip() for l in out if l.strip()), "")
        return line[:200]
    except Exception:
        return ""
