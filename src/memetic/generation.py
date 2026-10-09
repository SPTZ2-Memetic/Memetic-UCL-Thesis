"""Generator, redesigned: skilled bug generation driven by OUR skill bank +
matching pipeline, replacing the old memory/Case retrieval.

Difference from the previous generator (generator.py in the earlier project):
  - OLD: retrieved several past-bug Cases by construct-similarity, rendered a
    multi-skill <MEMORY> block, and ran 4 identical skilled generations.
  - NEW: per canonical, run the real retrieval -- extract sites (sites.py) ->
    Stage A structural match (matching.match_sites_to_skills) -> embedding
    pre-rank (matching.rank_candidates_by_embedding) -> Stage B calibrated
    judge (judge_skill.judge_skill_applicability) -- to get applicable
    (site, skill) pairs, then run 4 generations: 3 SKILLED, each conditioned
    on ONE DISTINCT skill (strong verdicts first), + 1 EXPLORATION (no skill,
    novelty prompt for discovering new mechanisms). A thin-supply task with
    fewer than 3 applicable skills fills the remaining slots with exploration
    so the batch is always 4.

Unchanged from the old generator (so the verifier can reconstruct the buggy
function and so generation-time validity is enforced the same way):
  - OUTPUT CONTRACT: one ```python``` block -> AST-extract BODY ->
    code_prompt + body = the runnable buggy function.
  - VALIDITY GATES at generation time: body extracts (valid Python, function
    present, signature unchanged), reconstructed source parses, body differs
    from canonical (no no-op), not self-labeled ('# Bug:' giveaways).

The skill is injected as GUIDANCE / an EXAMPLE of the KIND of mistake -- the
model is told to apply the PRINCIPLE (adapt it to this code), never to copy
the description's wording into the code. Productivity/difficulty is NOT
decided here (that needs the fixer); this only produces candidates and stamps
generation-time validity.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from memetic.types import Record
from memetic.sites import Site
from memetic.skills import Skill
from memetic.skill_bank import SkillBank
from memetic.matching import (
    CandidatePair, match_sites_to_skills, rank_candidates_by_embedding,
)
from memetic.judge_skill import judge_skill_applicability, ApplicabilityVerdict
from memetic.codeutils import parse_and_extract_function, is_self_labeled, strip_self_label
from memetic.seed import reconstruct_function, short_diff


@dataclass
class SkilledPair:
    """One retrieved, judged-applicable (site, skill) the generator can
    condition a skilled generation on -- carries the Stage B verdict so the
    prompt can show the abstract mistake and the role bindings."""
    site: Site
    skill: Skill
    verdict: ApplicabilityVerdict

    @property
    def level(self) -> str:
        return self.verdict.level


@dataclass
class BugCandidate:
    success: bool
    body: Optional[str] = None            # extracted body (4-space normalised)
    reconstructed: Optional[str] = None   # code_prompt + body: the runnable buggy fn
    mutation: str = ""                    # short diff canonical -> buggy
    reason: Optional[str] = None          # failure reason when success=False
    raw_output: str = ""
    prompt: str = ""
    self_labeled: bool = False
    is_exploration: bool = False          # generated with NO skill (novelty stream)
    skill_id: str = ""                    # which skill guided this (skilled stream)
    skill_level: str = ""                 # strong / plausible (skilled stream)
    metadata: dict = field(default_factory=dict)


# ---- retrieval -----------------------------------------------------------
def retrieve_skilled_pairs(
    record: Record,
    skill_bank: SkillBank,
    policy,
    embedder=None,
    *,
    judge_budget: int = 40,
    n_skills: int = 3,
) -> List[SkilledPair]:
    """The full Stage A + rank + Stage B retrieval for ONE canonical, returning
    up to n_skills applicable pairs on DISTINCT skills, strong verdicts first.

    judge_budget caps how many top-ranked candidates get a (paid) Stage B
    judge call -- the embedding pre-rank puts the most-likely-applicable pairs
    at the front, so a modest budget still surfaces the real matches without
    judging the hundreds of loose structural candidates. embedder=None skips
    ranking (judges in Stage A order) -- fine for small pools, wasteful for
    big ones."""
    canonical = reconstruct_function(record, "canonical_solution")
    pairs: List[CandidatePair] = match_sites_to_skills(canonical, skill_bank.all())
    if not pairs:
        return []
    if embedder is not None:
        pairs = rank_candidates_by_embedding(pairs, embedder, top_k=judge_budget)
    else:
        pairs = pairs[:judge_budget]

    applicable: List[SkilledPair] = []
    for p in pairs:
        verdict = judge_skill_applicability(canonical, p.site, p.skill, policy)
        if verdict.applicable:
            applicable.append(SkilledPair(site=p.site, skill=p.skill, verdict=verdict))

    # strong before plausible; keep retrieval (embedding) order within a tier.
    order = {"strong": 0, "plausible": 1}
    applicable.sort(key=lambda sp: order.get(sp.level, 2))

    # COVERAGE: distinct skills (design note Section 4 -- don't let one skill
    # fill all skilled slots). First applicable pair per skill_id wins (it is
    # already the strongest/highest-ranked for that skill).
    chosen: List[SkilledPair] = []
    seen: set = set()
    for sp in applicable:
        if sp.skill.skill_id in seen:
            continue
        seen.add(sp.skill.skill_id)
        chosen.append(sp)
        if len(chosen) >= n_skills:
            break
    return chosen


# ---- prompts -------------------------------------------------------------
_BASE_RULES = (
    "- The bug MUST change the function's OBSERVABLE BEHAVIOUR (its return "
    "value or output on some input) so that at least one unit test fails. A "
    "change that only rewrites or reorganises the code without changing what "
    "it returns is NOT a valid bug.\n"
    "- The resulting code MUST still be valid, COMPILING Python.\n"
    "- Do NOT drastically rewrite the code; keep the overall structure similar.\n"
    "- Do NOT change the function signature, imports, or I/O format.\n"
    "- Do NOT add any comment that describes or labels the bug (no "
    "'# Bug: ...'). It should read like a plausible, unannotated mistake.\n"
    "- Output ONLY the full buggy function inside a single ```python``` block."
)


def _render_bindings(bindings: Dict[str, str]) -> str:
    if not bindings:
        return "    (no specific roles identified)"
    return "\n".join(f"    {role} = {expr}" for role, expr in bindings.items())


def build_skilled_prompt(record: Record, pair: SkilledPair) -> str:
    """Skilled generation: inject the mechanism + the judge's abstract
    restatement + the matched site + the role bindings, as GUIDANCE. The
    model is told to apply the PRINCIPLE, adapted -- never to copy wording."""
    problem = (record.get("instruct_prompt") or record.get("code_prompt") or "").strip()
    canonical = reconstruct_function(record, "canonical_solution")
    v = pair.verdict
    abstract = f"\n  In general terms: {v.abstract_mistake}" if v.abstract_mistake else ""
    site_ctx = ""
    if pair.site.parent_text and pair.site.parent_text != pair.site.source_text:
        site_ctx = f"\n    (within this statement: {pair.site.parent_text})"
    return (
        "You are a *bug generator* for Python solutions to programming "
        "problems. You are given a problem, a CORRECT reference "
        "implementation, and ONE known bug MECHANISM to guide you.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Correct reference implementation:\n{canonical}\n\n"
        "A KNOWN BUG MECHANISM, to use as GUIDANCE and an EXAMPLE of the KIND "
        "of mistake to introduce (do NOT copy any of its wording into the "
        "code -- reason about the underlying principle and make the "
        "ANALOGOUS mistake here):\n"
        f"  Mechanism: {pair.skill.name} -- {pair.skill.mechanism}{abstract}\n"
        f"  Where it fits in this code: {pair.site.source_text}{site_ctx}\n"
        f"  The pieces it would act on here:\n{_render_bindings(v.bindings)}\n\n"
        "Now apply the SAME PRINCIPLE at (or around) that location: introduce "
        "a bug of THIS KIND, adapted to this specific code. It does not need "
        "to match the mechanism verbatim -- make the analogous, legitimate "
        "mistake that fits here.\n"
        f"{_BASE_RULES}\n\n"
        "Return the entire function with the buggy code inside a single "
        "```python``` block:\n"
    )


def build_exploration_prompt(record: Record) -> str:
    """Exploration generation: no skill. Push for an UNCOMMON bug, so the
    system can discover mechanisms the bank does not yet represent."""
    problem = (record.get("instruct_prompt") or record.get("code_prompt") or "").strip()
    canonical = reconstruct_function(record, "canonical_solution")
    return (
        "You are a *bug generator* for Python solutions to programming "
        "problems. You are given a problem and a CORRECT reference "
        "implementation.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Correct reference implementation:\n{canonical}\n\n"
        "Introduce ONE UNCOMMON, non-obvious subtle bug -- the kind that a "
        "catalogue of common mistakes would NOT already contain -- so a new "
        "failure pattern can be discovered. Be inventive but realistic: it "
        "must still look like a genuine mistake a developer could make, not a "
        "contrived edit.\n"
        f"{_BASE_RULES}\n\n"
        "Return the entire function with the buggy code inside a single "
        "```python``` block:\n"
    )


# ---- one generation + validity gates -------------------------------------
def _finish(record: Record, prompt: str, raw: str, *, is_exploration: bool,
            skill_id: str = "", skill_level: str = "") -> BugCandidate:
    """Shared parse + generation-time validity gates (identical contract to
    the old generator, so the verifier reconstructs the same way)."""
    entry_point = record.get("entry_point") or "task_func"
    canonical_body = record.get("canonical_solution", "")
    base = dict(raw_output=raw, prompt=prompt, is_exploration=is_exploration,
                skill_id=skill_id, skill_level=skill_level)

    body, fell_back = parse_and_extract_function(raw, entry_point)
    if body is None:
        return BugCandidate(False, reason="no_body", metadata={"parse_fell_back": fell_back}, **base)
    # RECOVER self-labeled bugs instead of rejecting: strip the giveaway
    # comment and keep the code. The remaining gates (no-op, parse) then judge
    # the stripped body -- a strip that empties the change or breaks syntax is
    # correctly rejected below, so this only ever recovers valid bugs.
    was_self_labeled = is_self_labeled(body)
    if was_self_labeled:
        body = strip_self_label(body)
    if body.strip() == canonical_body.strip():
        return BugCandidate(False, body=body, reason="no_change",
                            self_labeled=was_self_labeled, **base)
    reconstructed = (record.get("code_prompt", "") or "") + body
    try:
        ast.parse(reconstructed)
    except SyntaxError:
        return BugCandidate(False, body=body, reason="syntax_error",
                            self_labeled=was_self_labeled, **base)
    return BugCandidate(
        True, body=body, reconstructed=reconstructed,
        mutation=short_diff(canonical_body, body),
        self_labeled=was_self_labeled,
        metadata={"parse_fell_back": fell_back}, **base)


def generate_skilled(record: Record, policy, pair: SkilledPair) -> BugCandidate:
    prompt = build_skilled_prompt(record, pair)
    raw = policy.generate(prompt)
    return _finish(record, prompt, raw, is_exploration=False,
                   skill_id=pair.skill.skill_id, skill_level=pair.level)


def generate_exploration(record: Record, policy) -> BugCandidate:
    prompt = build_exploration_prompt(record)
    raw = policy.generate(prompt)
    return _finish(record, prompt, raw, is_exploration=True)


# ---- the 4-generation batch ----------------------------------------------
def generate_batch(
    record: Record,
    skill_bank: SkillBank,
    policy,
    embedder=None,
    *,
    n_total: int = 4,
    n_skilled_target: int = 3,
    judge_budget: int = 40,
    parallel: bool = True,
) -> List[BugCandidate]:
    """The redesign's per-task batch: retrieve applicable skills, then run
    n_total generations = (up to n_skilled_target distinct-skill SKILLED) +
    (the rest EXPLORATION). A task with fewer than n_skilled_target applicable
    skills fills the remaining slots with exploration, so the batch is always
    n_total and thin-supply tasks lean on discovery -- exactly the intended
    behaviour, not a failure.

    Returns the skilled candidates first (in strong->plausible order), then
    the exploration candidates."""
    pairs = retrieve_skilled_pairs(record, skill_bank, policy, embedder,
                                   judge_budget=judge_budget, n_skills=n_skilled_target)
    n_skilled = min(len(pairs), n_skilled_target, n_total)
    n_explore = n_total - n_skilled

    jobs = [("skilled", pairs[i]) for i in range(n_skilled)] + \
           [("explore", None)] * n_explore

    def _run(job):
        kind, pair = job
        if kind == "skilled":
            return generate_skilled(record, policy, pair)
        return generate_exploration(record, policy)

    if not parallel or len(jobs) <= 1:
        return [_run(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        return list(ex.map(_run, jobs))
