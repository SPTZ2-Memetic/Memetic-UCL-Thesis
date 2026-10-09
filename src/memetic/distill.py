"""Generator redesign, Phase 4: distilling real skills from BugSourceBench
canonical/buggy pairs, replacing the hand-authored SEED_SKILLS (which step
back into their intended role as a debugging/proof-of-concept baseline --
see project notes).

Pipeline (per the design note):
  BugSourceBench reference bugs
        |
  for each canonical -> buggy pair:
      identify actual edit/site      <- this module: find_changed_sites()
                                         (falls back to
                                         find_causal_site_via_failure() when
                                         the two don't statically align --
                                         see that function's docstring)
      describe reusable bug mechanism       )
      infer structural preconditions        ) <- this module: distill_skill()
      infer semantic preconditions          )
      infer negative conditions             )
        |
  candidate skills
        |
  merge / cluster equivalent mechanisms   <- NOT YET BUILT, needs its own
        |                                    validated ground truth first
  INITIAL SKILL BANK                         (see project notes)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from memetic.constructs import changed_construct
from memetic.sites import Site, extract_meaningful_sites, _COMPARE_OPS, _BINOP_OPS, _BOOLOP_OPS
from memetic.skills import Skill, SkillExample, StructuralPrecondition
from memetic.codeutils import error_kind, CRASH_KINDS, extract_traceback_line


def is_docstring_stub(body_bare: str) -> bool:
    """CONFIRMED NEEDED on real data: a real distillation probe run showed
    gpt_oss_20b's "buggy" outputs are frequently just the ENTIRE
    implementation replaced by a bare docstring, no real logic at all --
    e.g. a function that raises/returns None because it never does
    anything. These technically pass the "is_valid_bug" check (they parse
    and fail a test) but are NOT a real, subtle mistake worth distilling
    into a skill -- naively distilling them produces a plausible-sounding
    but actively harmful "skill" (e.g. "Documentation Drift") that would
    teach the generator "delete the implementation" is a legitimate bug
    tactic.

    Returns True if, after skipping any leading docstring, the body has NO
    real statement left (or only `pass`) -- i.e. it's a stub, not a bug."""
    import ast
    try:
        wrapped = body_bare if body_bare.lstrip().startswith("def ") else "def _f():\n" + body_bare
        tree = ast.parse(wrapped)
        fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except Exception:
        return False   # can't parse -> let the normal verifier catch it, not this filter
    stmts = fn.body
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(
            getattr(stmts[0], "value", None), ast.Constant) and isinstance(stmts[0].value.value, str):
        stmts = stmts[1:]   # skip the docstring itself
    real = [s for s in stmts if not isinstance(s, ast.Pass)]
    return len(real) == 0


def find_changed_sites(canonical_full_source: str, canonical_body: str,
                       buggy_body: str) -> List[Site]:
    """Locates WHICH extracted Site(s) correspond to what actually changed
    between a canonical and its buggy version -- the foundation for
    distillation (you can't describe a reusable mechanism without first
    knowing exactly where the real edit happened).

    Reuses changed_construct's ALIGNMENT detection unchanged (same
    definition of "a clean local edit" as construct-matching used all
    along) -- this function only adds the NEW step of mapping that changed
    STATEMENT down to the finer-grained Site(s) within it, so distillation
    gets a concrete site_type/operator/keywords to generalize from, not
    just raw diff text.

    canonical_full_source: the FULL function ('def f(...):\\n    ...') --
    needed by extract_meaningful_sites for real line numbers/ast.parse.
    canonical_body / buggy_body: BARE indented bodies (no 'def' line) --
    what changed_construct expects, matching how Records store them.

    Returns [] if the two don't align (a genuine rewrite, not a local
    edit) -- these cases are NOT distillable THIS way (see
    find_causal_site_via_failure for the fallback path a caller should try
    instead of simply discarding them), same exclusion construct-matching
    always applied for the static/cheap path. Returns [] (not a crash) if
    the changed text can't be matched to any extracted site either -- e.g.
    the change is purely in something extract_meaningful_sites doesn't
    track (a literal constant, a variable rename with no other change).

    SIZE-RATIO OVERRIDE -- CONFIRMED on real data (distillation_probe_v10,
    human/BigCodeBench/2 among others): changed_construct's `aligned` flag
    can be TRUE even for a genuine whole-function rewrite, when a SHORT
    canonical statement happens to have real (not coincidental -- see
    constructs._call_overlap's own corroboration fix) multi-call overlap
    against SOME statement buried in a much LONGER buggy rewrite. This is
    a distinct failure mode from the single-call-name loophole
    _call_overlap already guards against: here the overlap genuinely
    isn't a false positive by _call_overlap's own definition, it's just
    matching the WRONG statement because so much of the function changed
    around it that "aligned" stopped meaning "a local edit" at all. The
    result (confirmed on real output) is a vague, generic distilled skill
    ("Incorrect Data Transformation" -- a description so abstract it could
    describe almost any bug) because the LLM is being shown a real local
    site but a barely-related buggy counterpart plucked from a
    much-changed function.

    Detected the same way a human would eyeball it: if the buggy body is
    disproportionately larger than canonical (not just "a bit longer" --
    genuine rewrites in this dataset add many new lines, e.g. docstrings,
    restructured control flow), treat this as NOT a clean local edit after
    all and return [] -- the caller's existing fallback path
    (find_causal_site_via_failure) already exists specifically for this
    situation and does meaningfully better on it (CONFIRMED this session:
    every failure-grounded case reviewed at scale was accurate, while the
    one aligned-path case reviewed was the vague one). This REROUTES,
    it does not discard -- the case is still attempted, just via the path
    that's actually good at it."""
    changed_text, aligned = changed_construct(canonical_body, buggy_body)
    if not aligned or not changed_text:
        return []

    canonical_lines = [l for l in canonical_body.splitlines() if l.strip()]
    buggy_lines = [l for l in buggy_body.splitlines() if l.strip()]
    n_canon, n_buggy = len(canonical_lines), len(buggy_lines)
    size_ratio = n_buggy / max(1, n_canon)
    # both a relative AND absolute check: a tiny canonical (e.g. 2 lines)
    # growing to 4 lines is a 2x ratio but not remotely the same problem
    # as growing to 20 -- require genuine bulk added, not just a small
    # function being naturally sensitive to ratio math.
    if size_ratio > 3.0 and (n_buggy - n_canon) >= 5:
        return []

    all_sites = extract_meaningful_sites(canonical_full_source)
    # a site is "within" the changed statement if its own text is a
    # substring of the changed statement's text -- holds naturally since
    # sites are extracted from actual sub-nodes of real statements, and
    # normalizes whitespace on both sides first since changed_construct's
    # returned text may have different original formatting/indentation
    # than what _seg pulls fresh from the full source
    changed_norm = re.sub(r"\s+", " ", changed_text).strip()
    matches = []
    for s in all_sites:
        site_norm = re.sub(r"\s+", " ", s.source_text).strip()
        if site_norm and site_norm in changed_norm:
            matches.append(s)
    return matches


def _select_site_index(prompt: str, n_sites: int, policy,
                       require_causal_chain: bool = False,
                       min_chain_len: int = 60) -> Optional[Tuple[int, str, str]]:
    """Shared JSON-call/parse/validate plumbing for both localization tiers
    below -- identical failure-closed contract (None on any parsing/API
    failure, an out-of-range index, or an explicit -1 "no fit" answer).

    require_causal_chain: CONFIRMED needed on real data (distillation_probe
    v6 AND v7): merely instructing the model to "be conservative" and
    permitting site_index=-1 was NOT enough -- v7 still confidently picked
    a textually-similar-but-causally-unrelated site in both known-bad cases
    (BigCodeBench/6: picked the pattern-match call, which is provably
    behaviorally identical to canonical and cannot be the defect;
    BigCodeBench/15: picked a loop header entirely unrelated to the actual
    string-mismatch failure). A bare permission to decline doesn't force
    the model to actually check its own answer -- it still defaults to
    "textually closest to what looks different" over "causally explains
    the exact failure". Requiring a substantive causal_chain field (not
    just a one-sentence "reason") forces it to externalize that check: if
    it cannot articulate a concrete mechanism connecting THIS site's
    mutation to THIS exact failure text, a short/empty/generic chain is
    the tell, and this fails closed on it rather than trusting a bare
    site_index."""
    try:
        out = policy.generate(prompt).strip()
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", out, re.DOTALL)
        data = json.loads(m.group(1) if m else out)
        idx = data.get("site_index")
        if not isinstance(idx, int) or idx < 0 or idx >= n_sites:
            return None
        buggy_counterpart = str(data.get("buggy_counterpart", "")).strip()
        if not buggy_counterpart:
            return None
        causal_chain = str(data.get("causal_chain", "")).strip()
        if require_causal_chain and len(causal_chain) < min_chain_len:
            # too short to be a real mechanistic trace -- almost certainly a
            # filler sentence ("this site looks related") rather than actual
            # verification. Treat exactly like an unconfident answer: fail
            # closed, same as an explicit -1.
            return None
        reason = str(data.get("reason", "")).strip()
        return idx, buggy_counterpart, (causal_chain or reason)
    except Exception:
        return None


def find_causal_site_via_failure(canonical_full_source: str, buggy_full_source: str,
                                 test_failure: str, policy, entry_point: str = "",
                                 max_candidates: int = 40) -> Optional[Tuple[Site, str, str]]:
    """Fallback localization for cases where changed_construct's static
    alignment FAILS (a genuine rewrite -- different variable names,
    restructured control flow, often a whole new docstring) but the bug is
    still, empirically, a genuine FunctionVerifier-confirmed failure (see
    FunctionVerifier.is_valid_bug) -- meaning a real, causal behavioral
    divergence exists SOMEWHERE in the rewrite, even though static
    statement-level diffing cannot locate it (there's no reliable
    statement-to-statement correspondence once control flow itself has
    been restructured).

    CONFIRMED on real distillation-probe data that this is worth doing:
    rewrites are not random noise everywhere -- e.g.
    gpt_oss_20b/BigCodeBench/2 fully reimplements the function but the
    actual defect is a single literal changed (random.randint(0,100) ->
    random.randint(1,100)); gpt_oss_20b/BigCodeBench/181 fully
    reimplements but simply DROPS a time.sleep() call entirely. The
    previous behavior (find_changed_sites returning [] whenever alignment
    fails, with the caller then discarding the case) threw away real,
    usable bug signal in cases exactly like these.

    TWO TIERS -- CONFIRMED on real data (distillation_probe_v6, 33
    failure-grounded cases) that these are NOT equally reliable, and the
    difference tracks one concrete, checkable thing: whether the test
    failure includes a real traceback FRAME inside the buggy function
    itself (a raised exception -- IndexError/KeyError/TypeError/etc, see
    codeutils.CRASH_KINDS) versus a bare AssertionError (the buggy function
    ran to completion and just returned the wrong value, so there is no
    frame pointing at a specific line -- structurally nothing to localize
    to without genuine inference). Every correctly-localized v6 case had
    such a frame; every incorrectly-localized one (BigCodeBench/6
    mislocalized identically 3x, /15 mislocalized 2x, a cosmetic-string
    diff mislabeled a control-flow bug, one case with the defect's
    direction backwards) was a pure assertion failure -- i.e. the SAME
    prompt was being asked to do two different-difficulty jobs and never
    admitted the harder one was a guess (localization_failed=0/33).

    TIER 1 (error_kind(test_failure) in CRASH_KINDS, entry_point given):
    codeutils.extract_traceback_line deterministically extracts the EXACT
    buggy-function source line that raised -- ground truth, not an LLM
    guess. The LLM's job narrows to the much easier, corroborated
    question "which canonical candidate site is THIS VERBATIM LINE the
    counterpart of" -- and the returned buggy_counterpart is the
    deterministically-extracted line itself, never an LLM paraphrase.

    TIER 2 (everything else -- assertion/other/timeout, or no entry_point):
    genuinely no traceback frame to ground on. CONFIRMED on real data
    (distillation_probe_v7) that merely permitting the model to decline
    (site_index: -1) is NOT sufficient -- it still confidently picked a
    textually-similar-but-causally-wrong site in both known-bad cases
    (see _select_site_index's docstring for the specifics). Now requires
    an explicit, substantive CAUSAL CHAIN as part of the answer -- walk
    through what the ORIGINAL vs BUGGY code at the chosen site actually
    does, and confirm that difference mechanistically produces the EXACT
    observed failure text, not just "this looks like where it changed".
    A short/generic chain fails closed exactly like an explicit -1 (see
    _select_site_index) -- forcing the verification to happen, rather
    than trusting a bare pick, is what actually changed here, not the
    permission to decline (that alone was already present in v7 and
    didn't help).

    Fails closed (returns None) on any parsing/API failure, an
    out-of-range index, an explicit "no candidate fits" (-1) answer, or a
    causal chain too thin to trust -- consistent with the rest of this
    module; a failed localization must never silently produce a wrong
    site."""
    # max_candidates default CONFIRMED too tight at 15 on real data
    # (qwen7b/BigCodeBench/15, distillation_probe_v8): extract_meaningful_sites
    # walks statements in source order and the old cap of 15 was hit at
    # index 14 (the outer `for` loop's own header) -- BEFORE ever recursing
    # into the loop body, where the actual defect (an f.write(...) call with
    # a literal string) lived. The LLM never mislocalized this out of poor
    # reasoning; the true site was never offered as a candidate at all, at
    # any prompt quality. Raised to 40 so this class of case (the real
    # defect sitting inside a loop/with/if body, not near the top of the
    # function) has a real chance of being in the candidate list.
    sites = extract_meaningful_sites(canonical_full_source, max_sites=max_candidates)
    if not sites:
        return None

    listing = "\n".join(
        f"[{i}] ({s.site_type}/{s.operator or 'n/a'}): {s.source_text!r}"
        for i, s in enumerate(sites)
    )

    buggy_line = ""
    if entry_point and error_kind(test_failure) in CRASH_KINDS:
        buggy_line = extract_traceback_line(test_failure, entry_point)

    if buggy_line:
        # TIER 1: ground truth buggy line in hand -- only ask for site
        # ATTRIBUTION, a much more constrained and corroborated task than
        # diagnosing the failure from scratch.
        prompt = (
            "A function was changed from a CORRECT (canonical) version into "
            "a BUGGY version. The buggy version RAISED A REAL EXCEPTION when "
            "tested, and the exact source line that raised is known with "
            "certainty (extracted directly from the traceback, verbatim -- "
            "this is ground truth, not a guess). Your ONLY job is to "
            "identify which location in the CANONICAL function this exact "
            "buggy line is the counterpart of.\n\n"
            f"CANONICAL (correct) function:\n{canonical_full_source}\n\n"
            f"BUGGY (failing) function:\n{buggy_full_source}\n\n"
            f"THE EXACT BUGGY LINE THAT RAISED (verbatim from the traceback, "
            f"ground truth -- not paraphrased, not inferred):\n{buggy_line}\n\n"
            f"THE ACTUAL EXCEPTION (for context):\n{test_failure[-800:]}\n\n"
            f"CANDIDATE LOCATIONS in the canonical function (choose ONE by "
            f"index):\n{listing}\n\n"
            "Which single candidate is the canonical counterpart of the "
            "exact buggy line above -- i.e. the same conceptual location, "
            "before the change? If none of the candidates correspond to "
            "that location at all, answer site_index: -1.\n\n"
            "Reply with ONLY a JSON object, no other text:\n"
            '{"site_index": <int>, "buggy_counterpart": "...", "reason": "one short sentence"}'
        )
        picked = _select_site_index(prompt, len(sites), policy)
        if picked is not None:
            idx, _llm_counterpart, rationale = picked
            # the deterministic line is strictly more trustworthy than
            # anything the LLM could paraphrase -- always prefer it as the
            # returned buggy_counterpart, ignore the LLM's own version
            return sites[idx], buggy_line, rationale
        # deterministic extraction succeeded but site attribution failed --
        # fails closed rather than falling through to the weaker Tier 2
        # prompt on a case we already know has a locatable ground truth;
        # a wrong Tier-2 guess here would be strictly worse than admitting
        # failure.
        return None

    # TIER 2: no traceback frame into the buggy function exists (pure
    # AssertionError/other/timeout, or entry_point unavailable) -- genuinely
    # ambiguous, real inference required. CONFIRMED on real data
    # (distillation_probe_v7) that just permitting the model to decline
    # (site_index: -1) does NOT stop it defaulting to a textually-similar
    # but causally-wrong site -- e.g. it kept picking a regex-match call
    # that is provably behaviorally identical to canonical (so cannot be
    # the defect), and separately picked a loop header entirely unrelated
    # to an actual string-literal mismatch in an except-block message. In
    # both cases the model was matching on "this text changed nearby" not
    # "this change explains the observed failure". Now REQUIRES a
    # substantive causal_chain that mechanistically connects the site's
    # mutation to the exact failure text -- see _select_site_index.
    prompt = (
        "A function was changed from a CORRECT (canonical) version into a "
        "BUGGY version that genuinely fails a real test (shown below). The "
        "buggy version may be a substantial rewrite (renamed variables, "
        "restructured control flow, an added docstring) rather than a "
        "small edit. Unlike a crash, this failure is a plain wrong-value "
        "mismatch -- the buggy function ran to completion, so there is NO "
        "traceback pointing at a specific line; the failure alone proves "
        "SOMETHING diverges but gives no direct structural evidence of "
        "WHERE. Your job is to locate WHICH part of the CANONICAL function "
        "that divergence most plausibly corresponds to, and then PROVE it "
        "to yourself before answering -- not to describe or generalize it "
        "yet.\n\n"
        f"CANONICAL (correct) function:\n{canonical_full_source}\n\n"
        f"BUGGY (failing) function:\n{buggy_full_source}\n\n"
        f"ACTUAL TEST FAILURE (ground truth for what actually went wrong, "
        f"but note: this is a value mismatch, NOT a crash -- it does not "
        f"point at a line):\n{test_failure[-2000:]}\n\n"
        f"CANDIDATE LOCATIONS in the canonical function (choose ONE by "
        f"index):\n{listing}\n\n"
        "Which single candidate, if the buggy version's corresponding code "
        "were substituted in for it, would most plausibly PRODUCE EXACTLY "
        "this failure? Also give the buggy version's own text at that same "
        "conceptual location (your best identification of the "
        "corresponding code in the BUGGY function, verbatim from it -- not "
        "paraphrased).\n\n"
        "MANDATORY VERIFICATION -- do this before answering, not after: a "
        "candidate merely being near other textual changes in the diff is "
        "NOT evidence it is the defect (two code snippets can differ in "
        "SURFACE form -- variable names, which library call is used, "
        "restructured control flow -- while being fully BEHAVIORALLY "
        "EQUIVALENT; a behaviorally-equivalent change cannot be the cause "
        "of a real test failure, no matter how much text around it "
        "changed). For your chosen candidate, you must be able to state, "
        "concretely: (a) what the CANONICAL code at this site actually "
        "DOES at runtime, (b) what the BUGGY counterpart does DIFFERENTLY "
        "at runtime (not just differently in text), and (c) how that "
        "runtime difference produces the SPECIFIC values/text seen in the "
        "failure above -- not just 'something related to this changed'. "
        "If you cannot fill in all three concretely, this candidate is "
        "not proven -- either find a different candidate that you CAN "
        "prove, or answer site_index: -1. A wrong confident answer here "
        "is worse than admitting you don't know, and this task is "
        "genuinely ambiguous often enough that -1 should be a common, "
        "expected answer, not a rare escape hatch.\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"site_index": <int>, "buggy_counterpart": "...", '
        '"causal_chain": "(a) canonical does: ... (b) buggy does '
        'differently: ... (c) this produces the observed failure because: '
        '...", "reason": "one short sentence summary"}'
    )
    picked = _select_site_index(prompt, len(sites), policy, require_causal_chain=True)
    if picked is None:
        return None
    idx, buggy_counterpart, rationale = picked
    return sites[idx], buggy_counterpart, rationale


@dataclass
class DistilledSkill:
    name: str
    mechanism: str
    semantic_preconditions: str
    negative_conditions: str
    structural_operators: Optional[List[str]] = None
    structural_keywords_any: Optional[List[str]] = None
    structural_parent_types: Optional[List[str]] = None
    source: str = ""        # bugbench source, e.g. "human", "qwen7b" --
                             # together with source_task_id/source_diff/
                             # localization, this is the full provenance
                             # record to_skill() turns into a SkillExample.
    source_task_id: str = ""
    source_diff: str = ""
    # "aligned" (static changed_construct path) or "failure_grounded"
    # (find_causal_site_via_failure fallback) -- kept for post-hoc analysis
    # of which localization path produced which skills, not used by
    # matching/generation itself.
    localization: str = "aligned"
    # Set True when the LLM's proposed structural generalization
    # (operators/keywords_any/parent_types) failed the self-match invariant
    # in distill_skill and was discarded in favor of the minimal
    # site_type-only precondition -- i.e. the abstraction could not even
    # match the site it was distilled FROM. Diagnostic only (lets a run
    # report how often generalization had to be thrown away); never read by
    # matching.
    precondition_fallback: bool = False


def pick_most_specific_site(sites: List[Site]) -> Site:
    """When find_changed_sites returns more than one candidate Site for the
    same aligned edit, pick the innermost/most-specific one to distill
    from: shortest source_text, used as a proxy for "innermost" that
    doesn't rely on ast.walk's traversal order being stable.

    This is a real algorithmic choice about WHICH site a mechanism gets
    generalized from (not glue code) -- previously lived inline in
    scripts/probe_distillation.py's loop, which meant the exact same
    decision would have to be silently re-implemented (and could drift)
    anywhere else find_changed_sites is used, including the live self-play
    training loop (generator-redesign-note.md section 7 runs this same
    localize-then-distill machinery on every productive bug, every round).
    Kept here, next to find_changed_sites, as the one place this logic
    lives."""
    return min(sites, key=lambda s: len(s.source_text))


def distill_skill(canonical_full_source: str, site: Site, buggy_snippet: str,
                  policy, test_failure: str = "", causal_chain: str = "",
                  source_diff: str = "", source_task_id: str = "",
                  source: str = "") -> Optional[DistilledSkill]:
    """ONE LLM call: given a real (canonical, site, buggy version) triple,
    distill it into a candidate REUSABLE skill -- the mechanism, its
    semantic conditions, and an ABSTRACTED structural precondition (never
    the literal function/method name -- same principle proven in sites.py
    and skills.py's own hand-authored bank: keywords_any=["axis"] transfers
    across fillna/skew/etc, operators=["df.fillna"] would not).

    The site's OWN site_type is used AS-IS (deterministic, not LLM-guessed)
    -- only operators/keywords_any/parent_types (the parts that need
    generalizing beyond this one concrete instance) are asked of the LLM.

    test_failure (optional): the real FunctionVerifier failure output for
    this case. When the site came from find_causal_site_via_failure (a
    rewrite case, not a clean local edit), the mechanism the LLM would
    otherwise describe purely from a large/noisy diff is often vague or
    misses the real defect -- CONFIRMED on real probe data. Showing the
    actual failure grounds the mechanism description in what genuinely
    went wrong, not just what text differs. This is explicitly NOT the
    same as teaching the skill this instance's literal values -- see the
    explicit anti-leak instruction in the prompt below, which applies
    whether or not test_failure is passed.

    causal_chain (optional): the failure-grounded localizer's OWN verified
    reasoning (find_causal_site_via_failure's rationale, when it came from
    the mandatory causal_chain check -- see that function). CONFIRMED
    NEEDED on real data (distillation_probe_v8,
    human_edited_lm/BigCodeBench/6): without this, distill_skill only sees
    a bare buggy_snippet (sometimes a thin, low-context fragment, e.g. a
    bare function reference like "os.path.getmtime" with no surrounding
    call) and reconstructs a mechanism from surface text alone -- produced
    a nonsensical "Missing Function Call Parentheses" skill in that exact
    case, even though the localizer's OWN causal_chain had already
    correctly identified the real mechanism (a bare filename used where a
    full path was required). Passing that verified reasoning through
    means distill_skill generalizes from the CORRECT causal story instead
    of re-deriving (and sometimes getting wrong) its own from a
    context-poor snippet.

    source_diff/source_task_id (optional): passed straight through onto the
    returned DistilledSkill (see its fields) -- set HERE, as part of
    distillation itself, rather than left for the caller to bolt on
    afterwards. Confirmed worth doing this way: to_skill() reads
    source_diff into a Skill's manifestations, and a caller that forgets
    the after-the-fact assignment silently produces a skill with an empty
    manifestations list -- a real bug that happened on the very first
    wiring of this pipeline. Making these constructor-time parameters of
    the one function that builds a DistilledSkill removes that failure
    mode structurally instead of relying on every future caller to
    remember an extra step.

    Returns None on any parsing/API failure (fails closed, consistent with
    judge_skill_applicability -- a broken distillation should silently
    produce nothing, never a malformed skill added to the bank)."""
    failure_block = (
        f"The ACTUAL TEST FAILURE this caused (context on what genuinely "
        f"went wrong -- see instruction 6 below: never copy a literal "
        f"value from this into your answer, use it only to understand the "
        f"GENERAL nature of the mistake):\n{test_failure[-1200:]}\n\n"
        if test_failure else ""
    )
    causal_chain_block = (
        f"A VERIFIED causal explanation of the defect, already checked "
        f"against the actual runtime behavior (this is the authoritative "
        f"account of what's actually wrong -- trust this over your own "
        f"reading of the snippet alone if they seem to differ, but still "
        f"generalize it per instruction 6, never copy its literal values):"
        f"\n{causal_chain}\n\n"
        if causal_chain else ""
    )
    prompt = (
        "A REAL bug was found: a canonical (correct) function was changed "
        "into a buggy version. Distill the underlying REUSABLE mechanism "
        "behind this specific change -- not a description of this one "
        "edit, but the general kind of mistake it represents, in a form "
        "that could apply to OTHER, different code with a similar "
        "underlying situation.\n\n"
        f"CANONICAL (correct) function:\n{canonical_full_source}\n\n"
        f"The exact site that was changed:\n{site.source_text}\n"
        f"(site type: {site.site_type}, concrete operator/call: "
        f"{site.operator or 'n/a'})\n\n"
        f"The BUGGY version of that same site:\n{buggy_snippet}\n\n"
        f"{failure_block}"
        f"{causal_chain_block}"
        "Describe:\n"
        "1. A short name for this bug mechanism/family.\n"
        "2. The mechanism: what kind of mistake this represents, in "
        "general terms (not tied to these specific variable/function "
        "names).\n"
        "3. Semantic preconditions: what must be true about the SURROUNDING "
        "CODE for this mechanism to genuinely apply somewhere else (not "
        "just 'looks similar' -- what role must the site actually play).\n"
        "4. Negative conditions: when this mechanism should NOT be applied "
        "even if the site type matches.\n"
        "5. An ABSTRACTED structural signal for finding OTHER sites this "
        "same mechanism could apply to. This has THREE separate fields, "
        "each with its OWN closed vocabulary -- using ANY value outside "
        "these lists makes the field silently useless (it will never "
        "match real code), so when nothing in a list genuinely fits, "
        "leave that field null rather than inventing a plausible-sounding "
        "value:\n"
        "   structural_operators: ONLY from this exact list, or null -- "
        "comparisons: Lt, LtE, Gt, GtE, Eq, NotEq, Is, IsNot, In, NotIn; "
        "arithmetic/bitwise: Add, Sub, Mult, Div, FloorDiv, Mod, Pow, "
        "BitAnd, BitOr, BitXor, LShift, RShift; boolean: And, Or. NEVER a "
        "function or method name (e.g. never 'Call', 'reshape', "
        "'json.dumps') -- those are not valid values here at all.\n"
        "   structural_keywords_any: the NAMES of KEYWORD ARGUMENTS that "
        "actually appear at THIS changed site (i.e. names written as "
        "`name=...` inside the call shown above), when one of them is what "
        "makes this mechanism apply (e.g. 'axis', 'default', 'key'). NEVER a "
        "function/method name (the thing being CALLED, like 'replace' or "
        "'hosts'), NEVER a variable, constant, or dictionary key, and NEVER a "
        "positional-argument value. If the changed site is not a call, or the "
        "call has no keyword argument that matters here, use null.\n"
        "   structural_parent_types: the kind of STATEMENT the changed site "
        "itself sits in, ONLY from this exact list or null -- Assign, Return, "
        "For, While, If, Expr. Use the site's OWN enclosing statement (an "
        "assignment target is 'Assign', a return value is 'Return'), NOT the "
        "surrounding block it happens to be nested inside. NEVER a made-up "
        "category like 'FunctionDefinition', 'groupby', or a library/class "
        "name.\n\n"
        "6. CRITICAL: never reference this example's specific literal "
        "values (numbers, strings, this task's particular variable or "
        "function names, its specific inputs or outputs) anywhere in "
        "fields 1-4 above -- describe the ABSTRACT CLASS of thing instead "
        "(e.g. 'a boundary literal on a numeric range', 'a deleted "
        "side-effecting call', 'a threshold comparison operator'), never "
        "the concrete value from this one instance. A skill written in "
        "terms of this instance's literal values is useless for any other "
        "task -- if you catch yourself about to write a specific number, "
        "string, or this task's own identifier, generalize it first.\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"name": "...", "mechanism": "...", "semantic_preconditions": "...", '
        '"negative_conditions": "...", "structural_operators": [...] or null, '
        '"structural_keywords_any": [...] or null, '
        '"structural_parent_types": [...] or null}'
    )
    try:
        out = policy.generate(prompt).strip()
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", out, re.DOTALL)
        data = json.loads(m.group(1) if m else out)

        # DEFENSIVE validation: even with the constrained prompt above, the
        # LLM can still violate it -- confirmed on real data (an earlier,
        # unconstrained prompt produced operators=['Call','Assign'],
        # parent_types=['FunctionDefinition','groupby'], none of which are
        # real values Stage A would ever match against). Silently drop
        # anything outside the closed vocabulary rather than trust
        # compliance blindly -- a dropped invalid value degrades to "rely
        # on site_type alone", never a silently-dead structural field.
        _VALID_OPERATORS = set(_COMPARE_OPS.values()) | set(_BINOP_OPS.values()) | set(_BOOLOP_OPS.values())
        _VALID_PARENT_TYPES = {"Assign", "Return", "For", "While", "If", "Expr"}

        def _clean(values, valid_set):
            if not values or not isinstance(values, list):
                return None
            cleaned = [v for v in values if isinstance(v, str) and v in valid_set]
            return cleaned or None

        def _clean_keywords(values):
            # keyword ARGUMENT names are open-vocabulary (task-specific), so
            # no closed set to check against -- but a real keyword arg name
            # never contains a dot; a function/method name always does
            # (confirmed misuse on real data: keywords_any=['json.dumps',
            # 'base64.b64encode'] -- function names, not kwarg names)
            if not values or not isinstance(values, list):
                return None
            cleaned = [v for v in values if isinstance(v, str) and "." not in v and "(" not in v]
            return cleaned or None

        # SITE-TYPE-AWARE operator filtering: confirmed needed on real data
        # -- 'Eq' is a genuinely valid vocabulary value, but only means
        # anything when paired with site_type in {comparison, binary_expr,
        # boolean_expr} (those are the only site types whose OWN .operator
        # is drawn from this vocabulary at all -- see sites.py). For any
        # other site_type (call, return, assignment, subscript,
        # loop_header, collection_op), site.operator is a function name, a
        # bare string, or a literal like "Assign"/"For" -- it can NEVER
        # equal a comparison/binop/boolop class, so an operators list there
        # is silently unmatchable no matter how valid the individual
        # values look. The earlier vocabulary-only check missed this --
        # 'Eq' kept passing that filter while being structurally dead on
        # every non-expression site type it was attached to.
        operators = _clean(data.get("structural_operators"), _VALID_OPERATORS)
        if site.site_type not in ("comparison", "binary_expr", "boolean_expr"):
            operators = None

        # SITE-GROUNDED VALIDATION (fix for the 56 dead skills in bank v2).
        # The vocabulary checks above only confirm a value is a legal TOKEN
        # -- not that the site extractor (sites.py) would ever actually
        # produce it for THIS site. The LLM may only NARROW or DROP the
        # founding site's real, extracted attributes; it may never introduce
        # one the extractor can't emit here. Every dead skill violated
        # exactly this, so validate each field against the concrete site.

        # keywords_any: sites.py populates site.keywords ONLY for call sites,
        # and only with real kwarg NAMES (k.arg). So on any non-call site it
        # is always () -> keywords_any can never match and must be null. On a
        # call site, keep only kwargs actually present at the founding call:
        # a proposed name that isn't there (a method name like 'replace', a
        # variable/constant like 'smtp'/'INPUT_JSON', a dict key like
        # 'content-type', or the keyword 'as') can't even match the origin.
        keywords_any = _clean_keywords(data.get("structural_keywords_any"))
        if site.site_type != "call" or not site.keywords:
            keywords_any = None
        elif keywords_any is not None:
            keywords_any = [k for k in keywords_any if k in site.keywords] or None

        # parent_types: for assignment/return/loop_header/branch sites,
        # sites.py hardcodes parent_type as a constant function of site_type
        # (assignment->'Assign', return->'Return', for->'For', if->'If'); for
        # call sites it is the enclosing statement's type. Either way, a
        # proposed parent_type that isn't the site's OWN can never match the
        # origin, so keep only that one (or null). This is what killed the
        # assignment+'For'/'Return' skills: the LLM described the surrounding
        # loop instead of the site's own statement.
        parent_types = _clean(data.get("structural_parent_types"), _VALID_PARENT_TYPES)
        if parent_types is not None:
            parent_types = [site.parent_type] if site.parent_type in parent_types else None

        # operators: already null for non-expression sites (above). If kept,
        # it must INCLUDE the founding operator -- widening (Lt -> [Lt, LtE])
        # is legitimate generalization; a list that excludes the site's own
        # operator fails to match its own origin.
        if operators is not None and site.operator not in operators:
            operators = None

        # SELF-MATCH INVARIANT: a freshly distilled skill MUST match the very
        # site it was distilled from. After the grounding above this should
        # always hold, but assert it explicitly and fall back to the minimal
        # (site_type-only) precondition otherwise -- so it is STRUCTURALLY
        # impossible to mint a skill that is dead against its own origin (the
        # defect behind ~50 of the 56 dead skills in bank v2).
        candidate = StructuralPrecondition(
            site_types=[site.site_type], operators=operators,
            keywords_any=keywords_any, parent_types=parent_types)
        precondition_fallback = not candidate.matches(site)
        if precondition_fallback:
            operators = keywords_any = parent_types = None

        return DistilledSkill(
            name=str(data.get("name", "")).strip(),
            mechanism=str(data.get("mechanism", "")).strip(),
            semantic_preconditions=str(data.get("semantic_preconditions", "")).strip(),
            negative_conditions=str(data.get("negative_conditions", "")).strip(),
            structural_operators=operators,
            structural_keywords_any=keywords_any,
            structural_parent_types=parent_types,
            source=source,
            source_diff=source_diff,
            source_task_id=source_task_id,
            precondition_fallback=precondition_fallback,
        )
    except Exception:
        return None


def to_skill(distilled: DistilledSkill, site_type: str, skill_id: str) -> Skill:
    """Converts a validated DistilledSkill into a real Skill object, ready
    to be added to a bank -- kept as a separate explicit step (not done
    inside distill_skill itself) so a human/validation pass can inspect
    the DistilledSkill BEFORE it becomes a live, matchable Skill.

    `family` defaults to the skill's own skill_id -- i.e. a freshly created
    skill is the head of its own new family until/unless dedup.resolve_skill
    later creates a "specialization" that explicitly shares this family.

    distinct_tasks/valid_applications are seeded to 1, not 0 -- CONFIRMED
    BUG this fixes: the founding case that creates a skill IS a real,
    distinct, valid application of it, but was previously never counted
    (only later merges via SkillBank.merge_manifestation incremented these),
    so every skill's stats undercounted by exactly one. `examples` carries
    the same founding case with full provenance (source/task_id/
    localization), the audit trail parallel to the plain-string
    `manifestations` list -- see SkillExample's docstring for why both
    exist."""
    example = SkillExample(source=distilled.source, task_id=distilled.source_task_id,
                           diff=distilled.source_diff, localization=distilled.localization)
    return Skill(
        skill_id=skill_id, name=distilled.name, family=skill_id,
        mechanism=distilled.mechanism,
        structural=StructuralPrecondition(
            site_types=[site_type],
            operators=distilled.structural_operators,
            keywords_any=distilled.structural_keywords_any,
            parent_types=distilled.structural_parent_types,
        ),
        semantic_preconditions=distilled.semantic_preconditions,
        negative_conditions=distilled.negative_conditions,
        manifestations=[distilled.source_diff] if distilled.source_diff else [],
        examples=[example],
        distinct_tasks=1,
        valid_applications=1,
    )
