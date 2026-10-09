"""Live re-distillation (design note Section 7): turn ONE productive generator
bug back into a skill-bank update, using the SAME localize -> distill ->
resolve pipeline that scripts/probe_distillation.py runs on the warm-start
BugSourceBench data. There is deliberately ONE such pipeline, not two that
could drift -- this module just re-points the existing distill.* + dedup.*
helpers at a generator output instead of a published bug record.

Per Section 7, both generation streams feed the SAME machinery and only the
EXPECTED outcome differs:
  - skilled stream  -> mostly 'same'/'specialization' of the skill that guided
    it (logs a new real manifestation + bumps that skill's transfer stats).
    NOTE we do NOT force-attribute the bug to its guiding skill: resolution is
    decided on the distilled MECHANISM (structural -> text -> LLM), exactly as
    for warm-start data. A skilled bug resolving to its guiding skill is the
    expected, checkable outcome (see scripts/probe_redistill.py), not something
    hard-wired -- if the generator adapted the mechanism into something
    genuinely different, it SHOULD resolve elsewhere or to 'new'.
  - exploration stream -> mostly 'new' (mechanisms the bank doesn't yet hold).

The input bug is assumed ALREADY verified productive (a real, non-degenerate,
behaviour-changing bug) -- this module does not re-verify; the caller's
FunctionVerifier gate is what admits a bug to re-distillation at all.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from memetic.types import Record
from memetic.skill_bank import SkillBank
from memetic.dedup import resolve_skill, DedupDecision
from memetic.distill import (
    find_changed_sites,
    pick_most_specific_site,
    find_causal_site_via_failure,
    distill_skill,
)
from memetic.seed import short_diff


def _skill_id_for(source: str, task_id: str, diff: str) -> str:
    """Stable, unique-enough id for a would-be new skill. Only actually used
    when resolve_skill's outcome is 'new'/'specialization' (a 'same' merge
    reuses the existing skill's id). Deterministic in the diff so re-running
    the same bug produces the same id."""
    safe = str(task_id).replace("/", "_")
    h = hashlib.md5((diff or "").encode("utf-8")).hexdigest()[:8]
    return f"g_{source}_{safe}_{h}"


def redistill_bug(bank: SkillBank, record: Record, buggy_body: str, policy, *,
                  source: str, similarity_fn=None, text_threshold: float = 0.40,
                  max_llm_checks: int = 20, test_failure: str = "",
                  no_failure_fallback: bool = False,
                  skill_id: Optional[str] = None,
                  guiding_skill_id: Optional[str] = None) -> Optional[DedupDecision]:
    """Localize -> distill -> resolve ONE productive generator bug into `bank`.

    Returns the DedupDecision (outcome + full comparison trace + the resulting
    bank entry), or None if the bug can't be localized/distilled (no clean
    changed site AND no usable failure-grounded fallback, or distillation
    itself fails closed).

    source: 'generator_skilled' or 'generator_exploration' -- recorded as the
    skill's provenance; the code path is identical either way.
    similarity_fn / text_threshold / max_llm_checks: passed straight to
    resolve_skill -- use the SAME dedup settings as the warm-start run for a
    consistent bank (embedding metric, 0.40).
    guiding_skill_id (skilled stream): the id of the skill that guided this
    bug's generation. Looked up in the bank and passed to resolve_skill as the
    dedup PRIOR so the loop recognises its own output (see resolve_skill's
    docstring -- confirmed on real data that without it ~half of skilled bugs
    wrongly resolve to 'new' as paraphrases of their guiding skill)."""
    canonical_body = record.get("canonical_solution", "") or ""
    canonical_full = (record.get("code_prompt", "") or "") + canonical_body
    entry_point = record.get("entry_point") or "task_func"

    diff = short_diff(canonical_body, buggy_body, max_lines=40)

    sites = find_changed_sites(canonical_full, canonical_body, buggy_body)
    if sites:
        site = pick_most_specific_site(sites)
        localization = "aligned"
        buggy_snippet = diff
        causal = ""
        tf_for_distill = ""
    elif not no_failure_fallback and test_failure:
        # not a clean local edit (the generator rewrote a larger span) -- but
        # it verified as a real bug, so ground localization in the actual test
        # failure instead of discarding it (same fallback the warm-start pass
        # uses for gpt_oss_20b-style rewrites).
        bug_full = (record.get("code_prompt", "") or "") + buggy_body
        fallback = find_causal_site_via_failure(canonical_full, bug_full, test_failure,
                                                policy, entry_point=entry_point)
        if fallback is None:
            return None
        site, buggy_snippet, causal = fallback
        localization = "failure_grounded"
        tf_for_distill = test_failure
    else:
        return None

    d = distill_skill(canonical_full, site, buggy_snippet, policy,
                      test_failure=tf_for_distill, causal_chain=causal,
                      source_diff=diff, source_task_id=record.get("task_id", ""),
                      source=source)
    if d is None:
        return None
    d.localization = localization

    if skill_id is None:
        skill_id = _skill_id_for(source, record.get("task_id", ""), diff)
    guiding = bank.get(guiding_skill_id) if guiding_skill_id else None
    return resolve_skill(bank, d, site.site_type, skill_id, policy,
                         similarity_fn=similarity_fn, text_threshold=text_threshold,
                         max_llm_checks=max_llm_checks, guiding_skill=guiding)
