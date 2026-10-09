"""Generator redesign: skill-bank dedup/merge -- implements
generator-redesign-note.md section 7 ("Skill discovery and bank updates")
exactly. Every distilled candidate gets ONE of three outcomes:

  "same"           -- not a new skill; logged as another manifestation on
                       the matching existing skill, stats bumped.
  "specialization" -- a new Skill IS created (the distinction is reusable
                       enough to keep separate), but linked to its parent
                       via a shared `family` (section 2: "family: broader
                       grouping -- several skills can share one").
  "new"            -- a genuinely new mechanism; new skill, its own family
                       (it becomes the head of a new family).

CONFIRMED needed on real data: a real distillation run (per-source=50,
qwen-coder) produced 200 skills from 200 kept cases -- a 0% merge rate,
because bank.add_distilled() previously just inserted every case
unconditionally with no comparison against the bank at all. Given how
narrow individual bug instances are, that is almost certainly heavy
duplication, not 200 genuinely distinct mechanisms.

WHY THIS IS TWO CHEAP FILTERS BEFORE ANY LLM CALL, NOT "ask an LLM to
compare against every skill in the bank": that's O(bank size) LLM calls
PER newly distilled skill, i.e. O(n^2) LLM calls over a whole distillation
run -- confirmed not viable at n=200, let alone at the ~thousands of skills
a live, continuously-distilling training loop (section 7 again -- this
same machinery runs every round) would accumulate over time.

  1. STRUCTURAL pre-filter (free, deterministic, sites.py/skills.py
     already give us this for free via StructuralPrecondition.matches-style
     overlap): any bank skill sharing >=1 structural signal (site_type, an
     operator, a keyword, a parent type) with the new distilled skill
     becomes a CANDIDATE, nothing more. This is necessarily loose in BOTH
     directions and must not be trusted as a decision:
       - it can OVER-include: identical structural preconditions do NOT
         imply the same mechanism. Structural fields only describe WHERE a
         mutation could apply (site type / operator class / parent
         construct) -- sites.py's whole taxonomy is only ~10-15 categories,
         so two genuinely unrelated mechanisms (e.g. "boundary flip" vs
         "wrong comparison direction for min/max selection") can share the
         exact same structural signature: site_types=[comparison],
         operators=[Lt, LtE]. Confirmed by design, not a hypothetical.
       - it can UNDER-include: two independently-distilled instances of
         the SAME mechanism can get slightly different structural fields
         (distill_skill runs per-case with no visibility into the rest of
         the bank -- one call might set operators=None where another sets
         operators=[Eq] for what's actually the same underlying mistake).
     This stage exists ONLY to avoid comparing against the whole bank.
  2. MECHANISM-TEXT pre-filter (name+mechanism text; metric is SWAPPABLE --
     see below). This is where "is this actually the same bug being
     introduced" starts to be assessed at all -- structural fields never
     carry that signal, only free text does (see point above).
  3. LLM confirmation (paid, but now only on a handful of genuine
     near-neighbors per new skill, not the whole bank) -- a 3-way judge
     (same/specialization/different), mirroring section 7's three outcomes
     directly rather than a binary duplicate check.

STAGE 2 METRIC: defaults to stdlib difflib.SequenceMatcher (see
mechanism_text_similarity below) -- plain character/word overlap, no
embedding model. This was a real design choice for Stage A/B (this whole
matching pipeline was built specifically to replace embedding-similarity
matching with structural+semantic judgment there), but Stage 2 here is a
DIFFERENT job (mechanism-text clustering, not applicability judgment), and
SequenceMatcher was never actually a considered choice for it -- just what
stdlib offered when this was first written. It has a known weakness for
this specific job: distilled skills come from independent sources (human,
qwen7b, gpt_oss_20b) that describe the same mechanism in different words,
and paraphrases can share little literal text while meaning the same thing.
mechanism_candidates() below therefore takes a `similarity_fn` parameter --
memetic.embeddings.MechanismEmbeddingSimilarity is a drop-in alternative
(same call signature) -- so the metric can be swapped per-call without
touching this module, and scripts/calibrate_dedup.py exists specifically to
test which metric actually separates real same/different verdicts better,
rather than assuming.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from memetic.distill import DistilledSkill, to_skill
from memetic.skill_bank import SkillBank
from memetic.skills import Skill, SkillExample


@dataclass
class DedupComparison:
    """ONE LLM comparison made while resolving a distilled case -- kept
    even when the verdict is "different", not just the winning one. This
    is the full audit trail: "which bugs seen were decided different,
    specialization, or same" (and why), not just the final outcome."""
    candidate_skill_id: str
    text_similarity: float
    relation: str    # "same" | "specialization" | "different"
    reason: str


@dataclass
class DedupDecision:
    """The complete record of how ONE distilled case was resolved --
    everything needed to review/diagnose the decision by hand later,
    the same way judge_skill_applicability's verdicts were hand-validated
    (13 real cases, 92% agreement)."""
    outcome: str                     # "same" | "specialization" | "new"
    skill: Skill                     # the bank entry this case ended up on
    structural_candidate_count: int  # Stage-1 filter output size
    text_candidate_count: int        # Stage-2 filter output size (<= structural)
    comparisons: List[DedupComparison] = field(default_factory=list)
    candidates_capped: bool = False  # True iff more candidates cleared
                                     # text_threshold than max_llm_checks
                                     # allowed through -- see
                                     # mechanism_candidates()'s docstring.
                                     # A case worth reviewing by hand: it
                                     # means text_candidate_count was
                                     # truncated, not that nothing passed.


def _structural_signals(operators, keywords_any, parent_types) -> set:
    """Flattens one skill's (or one distilled candidate's) structural
    fields into a single set of tokens for cheap overlap checking."""
    out = set()
    for v in (operators or []):
        out.add(f"op:{v}")
    for v in (keywords_any or []):
        out.add(f"kw:{v}")
    for v in (parent_types or []):
        out.add(f"parent:{v}")
    return out


def skills_share_structural_signal(a: Skill, b: Skill) -> bool:
    """Same overlap rule as structural_candidates(), but skill-vs-skill
    instead of distilled-vs-bank -- used by
    scripts/analyze_skill_bank_similarity.py's whole-bank post-hoc check
    (structural_candidates() itself only compares a NEW DistilledSkill
    against the bank as it stood at that moment; this lets us re-check the
    FINAL bank for pairs that structurally could have matched at all,
    independent of when either was created)."""
    if not (set(a.structural.site_types) & set(b.structural.site_types)):
        return False
    sig_a = _structural_signals(a.structural.operators, a.structural.keywords_any,
                                a.structural.parent_types)
    sig_b = _structural_signals(b.structural.operators, b.structural.keywords_any,
                                b.structural.parent_types)
    if not sig_a and not sig_b:
        return True
    return bool(sig_a & sig_b)


def structural_candidates(distilled: DistilledSkill, site_type: str,
                          bank: SkillBank) -> List[Skill]:
    """Stage 1 filter (see module docstring): every bank skill that shares
    the same site_type AND at least one structural signal (operator,
    keyword, or parent type) with `distilled` -- or shares just the
    site_type when NEITHER has any finer-grained signal at all (both left
    everything else null, i.e. "any site of this type"). A pure candidate
    list, not a decision."""
    new_signals = _structural_signals(distilled.structural_operators,
                                      distilled.structural_keywords_any,
                                      distilled.structural_parent_types)
    out = []
    for skill in bank:
        if site_type not in skill.structural.site_types:
            continue
        existing_signals = _structural_signals(
            skill.structural.operators, skill.structural.keywords_any,
            skill.structural.parent_types)
        if not new_signals and not existing_signals:
            out.append(skill)          # both maximally loose on this site_type
        elif new_signals & existing_signals:
            out.append(skill)
    return out


def mechanism_text_similarity(a_name: str, a_mechanism: str,
                              b_name: str, b_mechanism: str) -> float:
    """Stage 2 filter (see module docstring): plain stdlib text similarity
    on name+mechanism, no embedding model. difflib.SequenceMatcher.ratio()
    is a simple, dependency-free choice, not claimed to be the best
    possible text-similarity metric -- good enough to narrow a handful of
    structural candidates down to genuine near-neighbors before spending
    any LLM call, which is all this stage needs to do."""
    a = f"{a_name} {a_mechanism}".strip().lower()
    b = f"{b_name} {b_mechanism}".strip().lower()
    return difflib.SequenceMatcher(None, a, b).ratio()


def mechanism_candidates(distilled: DistilledSkill, candidates: List[Skill],
                         similarity_fn: Optional[Callable] = None,
                         threshold: float = 0.45,
                         max_llm_checks: int = 20) -> Tuple[List[Tuple[Skill, float]], bool]:
    """Ranks Stage-1 candidates by mechanism-text similarity, keeps those
    above `threshold`, WITH their similarity scores -- kept (not discarded)
    so the full decision trace (DedupComparison) can record what actually
    drove each comparison.

    `similarity_fn`: the Stage-2 metric, defaulting to
    mechanism_text_similarity (difflib.SequenceMatcher). Pass an instance of
    memetic.embeddings.MechanismEmbeddingSimilarity (same call
    signature) to use local embeddings instead -- see this module's
    docstring for why that's worth testing, not assumed.

    `max_llm_checks` is now a SAFETY CEILING on cost, NOT a tuning knob for
    quality -- every candidate that clears `threshold` is returned (and
    gets an LLM check) by default, since `threshold` is what's supposed to
    be doing the actual filtering. This only truncates -- and returns
    capped=True when it does -- if an unusually large number of candidates
    clear the threshold at once (a "hub skill" smell -- see
    scripts/analyze_skill_bank_similarity.py's d_qwen7b_..._0035 finding).
    CONFIRMED on real data that the previous behavior (max_candidates=3,
    silently dropping anything past the top 3 with no record of it) was a
    real bug, not a conservative default: a genuine 0.64-similarity
    duplicate lost its only LLM-check slot to unrelated candidates that
    happened to momentarily rank higher, and was never compared at all.
    Silently dropping candidates hides exactly the cases worth reviewing;
    returning `capped=True` (recorded in DedupDecision) surfaces them
    instead.

    Both `threshold` and `max_llm_checks` are still unvalidated defaults --
    see scripts/calibrate_dedup.py, built specifically to set both from
    real LLM verdicts rather than guesses, the same way
    judge_skill_applicability's cutoffs were validated."""
    similarity_fn = similarity_fn or mechanism_text_similarity
    scored = [
        (skill, similarity_fn(distilled.name, distilled.mechanism,
                              skill.name, skill.mechanism))
        for skill in candidates
    ]
    scored = [(s, score) for s, score in scored if score >= threshold]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    capped = len(scored) > max_llm_checks
    return scored[:max_llm_checks], capped


def _format_structural_signal(site_types, operators, keywords_any, parent_types) -> str:
    """Renders a Skill's (or a distilled candidate's) structural fields as a
    short human-readable summary for the judge prompt -- context only, see
    judge_skill_relation's docstring for why this is never decisive on its
    own."""
    parts = []
    if site_types:
        parts.append(f"site_types={list(site_types)}")
    if operators:
        parts.append(f"operators={list(operators)}")
    if keywords_any:
        parts.append(f"keywords_any={list(keywords_any)}")
    if parent_types:
        parts.append(f"parent_types={list(parent_types)}")
    return "; ".join(parts) if parts else "(no structural constraints -- matches broadly)"


def judge_skill_relation(existing: Skill, distilled: DistilledSkill,
                         policy, *, distilled_site_type: str = "") -> Tuple[str, str]:
    """ONE LLM call: is `distilled` the SAME mechanism as `existing`, a
    SPECIALIZATION/variant of it, or a genuinely DIFFERENT mechanism that
    merely looked like a text/structural near-neighbor? Mirrors section 7's
    three outcomes directly (this function only ever needs to distinguish
    same/specialization/different -- "genuinely new mechanism" with no
    existing neighbor at all is decided by the caller before this is even
    invoked, when Stage 1/2 return no candidates).

    Includes each side's structural signal (site type / operator / keyword /
    parent type) in the prompt as CONTEXT, not as a decision input on its
    own -- explicitly instructed below not to be treated as evidence either
    way, since structural fields only describe WHERE a mutation could apply,
    never WHAT the mistake is (confirmed by design -- see module docstring,
    and the SEED_SKILLS boundary_condition family in skills.py, where the
    SAME mechanism deliberately spans three different site_types:
    comparison, loop_header, subscript). The reason to show it at all is
    that it can still help the judge understand what it's looking at when
    the mechanism text alone is vague (e.g. the hub-skill pattern found on
    real data: "Incorrect Data Structure Transformation" reads as similar to
    many unrelated mechanisms by text alone; seeing that one case is a
    boundary comparison and the other a dict-key lookup can help the judge
    notice they're not the same kind of operation at all, without the
    prompt asserting that conclusion for it).

    `distilled_site_type` (optional): DistilledSkill itself carries no
    site_types field (that's supplied by the caller alongside it, same as
    to_skill()) -- passed through here only for display in the prompt, not
    used for anything else.

    FAILS CLOSED to "different" on any parsing/API error -- same
    fail-closed philosophy as judge_skill_applicability (see that module's
    docstring): an unreliable comparison should never silently MERGE two
    skills (which destroys information -- a wrongly-merged skill's distinct
    identity can't be recovered later), so the safe default on failure is
    to let the bank grow by one extra skill rather than risk erasing a real
    distinction."""
    existing_structural = _format_structural_signal(
        existing.structural.site_types, existing.structural.operators,
        existing.structural.keywords_any, existing.structural.parent_types)
    distilled_structural = _format_structural_signal(
        [distilled_site_type] if distilled_site_type else [],
        distilled.structural_operators, distilled.structural_keywords_any,
        distilled.structural_parent_types)
    prompt = (
        "You are checking whether two BUG MECHANISMS are actually the "
        "SAME underlying mistake, a SPECIALIZATION/variant of one shared "
        "mechanism, or genuinely DIFFERENT mechanisms that merely look "
        "textually or structurally similar.\n\n"
        f"EXISTING SKILL: {existing.name}\n"
        f"Mechanism: {existing.mechanism}\n"
        f"Semantic preconditions: {existing.semantic_preconditions}\n"
        f"Structural signal (context only, see note below): {existing_structural}\n\n"
        f"NEW CANDIDATE: {distilled.name}\n"
        f"Mechanism: {distilled.mechanism}\n"
        f"Semantic preconditions: {distilled.semantic_preconditions}\n"
        f"Structural signal (context only, see note below): {distilled_structural}\n\n"
        "IMPORTANT NOTE on the structural signals above: these describe "
        "only WHERE in the code a mutation could apply (site type / "
        "operator class / keyword argument name) -- NEVER what the "
        "mistake actually is. Do NOT treat matching structural signals as "
        "evidence the mechanisms are the same, and do NOT treat differing "
        "structural signals as evidence they are different. The exact "
        "same mechanism can legitimately show up at different structural "
        "sites (e.g. an off-by-one boundary mistake can appear as a "
        "comparison, a loop bound, or a slice index -- all the same "
        "family despite different site types), and two genuinely "
        "unrelated mechanisms can share an identical structural "
        "signature. Base your judgment on the mechanism and semantic "
        "preconditions text; use the structural signal only as "
        "background context for understanding what each case actually "
        "involves.\n\n"
        "Answer:\n"
        "- \"same\": these describe the identical underlying mistake -- the "
        "new one should NOT become a separate skill, just another example "
        "of the existing one.\n"
        "- \"specialization\": the new one is a genuine, reusable variant "
        "of the same broad family (e.g. the same kind of boundary mistake "
        "but in a meaningfully different form) -- worth keeping as its own "
        "skill, but grouped with the existing one.\n"
        "- \"different\": these are actually unrelated mechanisms that "
        "merely happened to look similar (e.g. same site type/operator "
        "class, but a genuinely different kind of mistake).\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"relation": "same" or "specialization" or "different", '
        '"reason": "one short sentence"}'
    )
    try:
        out = policy.generate(prompt).strip()
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", out, re.DOTALL)
        data = json.loads(m.group(1) if m else out)
        relation = str(data.get("relation", "different")).strip().lower()
        if relation not in ("same", "specialization", "different"):
            relation = "different"
        reason = str(data.get("reason", ""))
        return relation, reason
    except Exception:
        return "different", "judge_error"


def resolve_skill(bank: SkillBank, distilled: DistilledSkill, site_type: str,
                  skill_id: str, policy, *,
                  similarity_fn: Optional[Callable] = None,
                  text_threshold: float = 0.45,
                  max_llm_checks: int = 20,
                  guiding_skill=None) -> DedupDecision:
    """The full section-7 decision for ONE newly distilled case: runs the
    structural -> text -> LLM filter chain, applies whichever of the three
    outcomes the LLM (or the absence of any candidate at all) settles on,
    and returns a DedupDecision -- the outcome, the resulting bank entry,
    AND the full comparison trace (every candidate checked, including ones
    judged "different", with their similarity scores and reasons) so the
    whole decision can be reviewed and diagnosed later, not just the final
    label.

    `similarity_fn` / `text_threshold` / `max_llm_checks` pass straight
    through to mechanism_candidates() -- see that function's docstring for
    what each actually controls (max_llm_checks is a cost ceiling, not a
    quality knob; see also scripts/calibrate_dedup.py for how these should
    actually be set).

    This is the single function scripts/probe_distillation.py (and, later,
    the live self-play training loop) should call in place of a bare
    bank.add_distilled() -- see that function's own docstring for why a
    bare add was wrong (it never checked the bank at all).

    guiding_skill (live skilled re-distillation only): the Skill that GUIDED
    the generator to make this bug. When given, it is judged FIRST, bypassing
    the structural/text prefilter. CONFIRMED needed on real data
    (scripts/probe_redistill.py): ~half of skilled bugs' re-distilled skills
    were paraphrases of their guiding skill that the structural->text prefilter
    failed to connect, so they wrongly resolved to 'new' and inflated the bank
    with near-duplicates. Judging the guiding skill directly recovers those as
    same/specialization; a genuinely drifted bug (the generator made a
    different mistake than assigned) is still judged 'different' here and falls
    through to the normal dedup below -- so this only recovers real
    loop-closure, never forces a bad merge."""
    comparisons: List[DedupComparison] = []

    # GUIDING-SKILL PRIOR: first right of refusal for the skill that produced
    # this bug, if known and still in the bank.
    if guiding_skill is not None and guiding_skill.skill_id in bank:
        relation, reason = judge_skill_relation(guiding_skill, distilled, policy,
                                                distilled_site_type=site_type)
        comparisons.append(DedupComparison(candidate_skill_id=guiding_skill.skill_id,
                                           text_similarity=-1.0, relation=relation,
                                           reason=reason))
        if relation == "same":
            example = SkillExample(source=distilled.source, task_id=distilled.source_task_id,
                                   diff=distilled.source_diff, localization=distilled.localization)
            merged = bank.merge_manifestation(guiding_skill.skill_id, example)
            return DedupDecision(outcome="same", skill=merged,
                                 structural_candidate_count=0, text_candidate_count=0,
                                 comparisons=comparisons)
        if relation == "specialization":
            new_skill = to_skill(distilled, site_type, skill_id)
            new_skill.family = guiding_skill.family
            bank.add(new_skill)
            return DedupDecision(outcome="specialization", skill=new_skill,
                                 structural_candidate_count=0, text_candidate_count=0,
                                 comparisons=comparisons)
        # "different" -> fall through to the normal structural/text/LLM dedup

    structural = structural_candidates(distilled, site_type, bank)
    if not structural:
        skill = bank.add_distilled(distilled, site_type, skill_id)
        return DedupDecision(outcome="new", skill=skill,
                             structural_candidate_count=0, text_candidate_count=0,
                             comparisons=comparisons)

    ranked, capped = mechanism_candidates(distilled, structural,
                                          similarity_fn=similarity_fn,
                                          threshold=text_threshold,
                                          max_llm_checks=max_llm_checks)
    if not ranked:
        skill = bank.add_distilled(distilled, site_type, skill_id)
        return DedupDecision(outcome="new", skill=skill,
                             structural_candidate_count=len(structural),
                             text_candidate_count=0, comparisons=comparisons,
                             candidates_capped=capped)

    for candidate, score in ranked:
        if guiding_skill is not None and candidate.skill_id == guiding_skill.skill_id:
            continue   # already judged above as the guiding prior
        relation, reason = judge_skill_relation(candidate, distilled, policy,
                                                distilled_site_type=site_type)
        comparisons.append(DedupComparison(candidate_skill_id=candidate.skill_id,
                                           text_similarity=score, relation=relation,
                                           reason=reason))
        if relation == "same":
            example = SkillExample(source=distilled.source, task_id=distilled.source_task_id,
                                   diff=distilled.source_diff, localization=distilled.localization)
            merged = bank.merge_manifestation(candidate.skill_id, example)
            return DedupDecision(outcome="same", skill=merged,
                                 structural_candidate_count=len(structural),
                                 text_candidate_count=len(ranked), comparisons=comparisons,
                                 candidates_capped=capped)
        if relation == "specialization":
            new_skill = to_skill(distilled, site_type, skill_id)
            new_skill.family = candidate.family   # section 2: shared family = the grouping mechanism
            bank.add(new_skill)
            return DedupDecision(outcome="specialization", skill=new_skill,
                                 structural_candidate_count=len(structural),
                                 text_candidate_count=len(ranked), comparisons=comparisons,
                                 candidates_capped=capped)
        # "different" -- logged in comparisons above, keep checking the rest

    skill = bank.add_distilled(distilled, site_type, skill_id)
    return DedupDecision(outcome="new", skill=skill,
                         structural_candidate_count=len(structural),
                         text_candidate_count=len(ranked), comparisons=comparisons,
                         candidates_capped=capped)
