"""Generator redesign, Phase 3, Stage B: semantic applicability judge.

Stage A (matching.py) is deliberately loose for several skills (no cheap
structural feature exists to discriminate them further -- see skills.py's
own comments on wrong_statistic/wrong_variable_reference). This means Stage
B carries most of the REAL applicability burden, unlike the earlier
construct-matching judge (generator.judge_construct_relevance), which was a
secondary filter on top of already similarity-vetted candidates.

CONSEQUENCE, worth being explicit about: this judge FAILS CLOSED (treat as
NOT applicable on any parsing/API error), the opposite of the earlier
judge's fail-open design. Failing open here would mean showing the
generator an UNVALIDATED (site, skill) pair whenever the judge call itself
breaks -- exactly the kind of confidently-wrong guidance this whole
redesign exists to prevent. A judge outage here should degrade to "less
guidance shown" (closer to the exploration/unguided case), never to
"trust an unchecked match."

Per the design note's own worked example, this also asks for VARIABLE-ROLE
BINDINGS (e.g. INDEX=i, BOUND=len(items), SEQUENCE=items) when applicable --
not just yes/no -- since a later generation step can use these bindings to
ground the instruction concretely, rather than re-deriving them itself.

IMPORTANT: manifestations are DELIBERATELY NOT shown to this judge (see
skills.py's own invariant comment) -- only mechanism, semantic_preconditions
and negative_conditions. Showing manifestations risks the judge anchoring
on "does this look like the example" rather than genuinely evaluating the
mechanism, which would silently re-create construct-matching's exact
problem one layer up.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from memetic.sites import Site
from memetic.skills import Skill


@dataclass
class ApplicabilityVerdict:
    applicable: bool                       # convenience: True iff level in
                                           # {"strong","plausible"} -- kept so
                                           # existing callers (probe scripts,
                                           # coverage analysis) that read
                                           # .applicable keep working unchanged.
    bindings: Dict[str, str] = field(default_factory=dict)
    reason: str = ""
    level: str = "no"                      # "strong" | "plausible" | "no" --
                                           # the graded verdict. Generation
                                           # prioritises "strong" pairs, falls
                                           # back to "plausible" to fill slots.
    abstract_mistake: str = ""             # the mechanism stated abstractly,
                                           # stripped of its origin domain --
                                           # the judge's own generalisation,
                                           # useful in the generator prompt and
                                           # for audit.


def judge_skill_applicability(canonical_source: str, site: Site, skill: Skill,
                              policy) -> ApplicabilityVerdict:
    """Stage B: ONE LLM call per (site, skill) candidate pair. Shows the
    FULL canonical (per the design note: semantic preconditions must be
    evaluated "in the context of the full function", not the site in
    isolation -- e.g. "is this call's result used in a per-row computation"
    requires seeing how the result is used elsewhere), the specific site,
    and the skill's mechanism/semantic_preconditions/negative_conditions.

    Returns an ApplicabilityVerdict with a GRADED level (strong/plausible/no).
    FAILS CLOSED (level="no", applicable=False) on any parsing or API failure.

    CALIBRATED ANALOGY JUDGE (validated in scripts/ab_judge_looseness.py, two
    A/B rounds on the clean held-out coverage trace). The earlier STRICT
    verifier asked "is a near-exact instance of this mechanism already
    present?" and, on real data, rejected ~1/3 of genuinely-applicable pairs
    (confirmed realistic bugs like os.path.exists->isdir, `<`->`<=`, raw-vs-
    standardised data) because they were adaptations, not verbatim matches --
    starving the generator's skilled stream. A skill is an ABSTRACT kind of
    mistake and the generator ADAPTS it, so this judge instead asks: does THIS
    location have the ingredients for that KIND of mistake, using expressions
    actually present here? The naive-loose version of that question overshot
    (fabricated bindings, no-ops, self-contradictions); the two guards kept
    below are exactly the ones the A/B showed were load-bearing."""
    prompt = (
        "You are deciding whether a learned BUG MECHANISM can be used to "
        "introduce a realistic bug into a new function. A skill describes an "
        "ABSTRACT KIND of developer mistake, learned from past examples -- "
        "NOT a specific edit to a specific function. The bug generator will "
        "ADAPT the mechanism to fit new code, so you are NOT checking whether "
        "this function resembles where the mechanism was first seen. You are "
        "checking whether THIS location has the ingredients for that KIND of "
        "mistake.\n\n"
        "IMPORTANT: the function below is the CORRECT version -- never "
        "currently buggy -- so 'this is already correct' is never a reason "
        "to say no (that is true everywhere).\n\n"
        f"FULL FUNCTION:\n{canonical_source}\n\n"
        f"SPECIFIC LOCATION:\n{site.source_text}\n"
        f"ENCLOSING STATEMENT (may already contain a guard that makes the bug "
        f"a no-op):\n{site.parent_text}\n\n"
        f"BUG MECHANISM (an abstract kind of mistake): {skill.name}\n"
        f"What the mistake is: {skill.mechanism}\n"
        f"Where it tends to apply: {skill.semantic_preconditions}\n"
        f"Genuinely does NOT apply if: {skill.negative_conditions}\n\n"
        "Reason in three steps, then answer:\n"
        "1. ABSTRACT MISTAKE: in one phrase, the underlying kind of mistake "
        "here, stripped of the specific library or domain it was first seen "
        "in (e.g. 'use a raw value where a transformed one was needed', not "
        "'base64 vs ascii').\n"
        "2. THIS LOCATION: what the code here actually does, and what real "
        "expressions are present.\n"
        "3. FIT: could a developer introduce THIS KIND of mistake at THIS "
        "location -- adapting it as needed -- using expressions ACTUALLY "
        "present here, so that behavior genuinely changes? Describe the "
        "concrete mutation.\n\n"
        "Rules:\n"
        "- A reasonable ADAPTATION counts. Different library or surface "
        "syntax but the SAME underlying mistake = it applies. It does NOT "
        "have to be verbatim what the mechanism was learned from.\n"
        "- BUT the bug must be introducible AT this location using what is "
        "ACTUALLY here. If step 3 needs you to invent variables, calls, or "
        "code that are not present, answer 'no' -- that is forcing a fit.\n"
        "- If the mutation would leave behavior identical (a no-op), 'no'.\n"
        "- If the 'does NOT apply' condition is clearly met, 'no'.\n\n"
        "Grade the fit:\n"
        "  'strong'   = this location naturally hosts this kind of mistake; "
        "the adaptation is small and obvious.\n"
        "  'plausible'= a reasonable adaptation fits here and would make a "
        "realistic bug.\n"
        "  'no'       = essentially unrelated, or a no-op, or it requires "
        "inventing code that isn't here.\n\n"
        "For bindings: give the concrete expression for each role. If two "
        "roles must be DIFFERENT things, bind them to DIFFERENT expressions "
        "that actually appear above.\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"abstract_mistake": "...", "applicability": "strong" or '
        '"plausible" or "no", "bindings": {"ROLE": "expression", ...}, '
        '"reason": "one sentence naming the concrete mutation"}'
    )
    try:
        out = policy.generate(prompt).strip()
        # strip a markdown code fence if present, same tolerance as
        # elsewhere in this codebase for LLM output that ignores
        # "ONLY a JSON object" and wraps it anyway
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", out, re.DOTALL)
        json_text = m.group(1) if m else out
        data = json.loads(json_text)
        level = str(data.get("applicability", "no")).strip().lower()
        if level not in ("strong", "plausible", "no"):
            level = "no"
        bindings = data.get("bindings", {}) or {}
        if not isinstance(bindings, dict):
            bindings = {}
        bindings = {str(k): str(v) for k, v in bindings.items()}
        reason = str(data.get("reason", ""))
        abstract = str(data.get("abstract_mistake", ""))

        # DISTINCT-BINDING GUARD -- self-contradictory bindings (2+ distinct
        # roles bound to the SAME expression) are WEAK evidence of a forced
        # fit, so DOWNGRADE strong->plausible rather than reject. Confirmed on
        # real data: a hard reject killed 2 of 6 genuine matches (a size-tuple
        # off-by-one and a stricter column check) where the mechanism
        # legitimately operates on one expression -- a 33% regression.
        # Demoting denies these the "strong" priority without dropping them;
        # post-generation validity is the real filter for the rest.
        if level == "strong" and len(bindings) > 1 \
                and len(set(bindings.values())) < len(bindings):
            level = "plausible"
            reason = (reason + " [downgraded strong->plausible: distinct roles "
                     "bound to the same expression]").strip()

        return ApplicabilityVerdict(applicable=(level in ("strong", "plausible")),
                                    bindings=bindings, reason=reason,
                                    level=level, abstract_mistake=abstract)
    except Exception:
        return ApplicabilityVerdict(applicable=False, bindings={}, reason="judge_error",
                                    level="no", abstract_mistake="")
