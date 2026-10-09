"""Generator redesign, Phase 2 (see generator_redesign_note.docx): the Skill
data structure and a small, HAND-AUTHORED seed bank.

Deliberately hand-authored, not auto-discovered yet (see project notes --
auto-distillation, Section 7 of the design note, is a genuinely hard,
unvalidated subproblem; building it first would confound every downstream
test of matching quality with "are the auto-discovered skills any good").
Every seed skill below is grounded in a CONFIRMED real pattern found by hand
in tonight's actual fixer-eval regression report (scripts/
diagnose_fixer_eval.py output) -- cited in each skill's docstring-style
comment -- not invented abstractly.

CRITICAL design rule, proven in sites.py: structural preconditions must be
ABSTRACT (site type / parent type / keyword NAMES), never a literal
function/method name. A precondition of operators=["df.fillna"] would only
ever match df.fillna calls -- exactly as narrow as the cosine-similarity
matching this redesign exists to replace. A precondition of
keywords_any=["axis"] matches df.fillna(axis=1), skew(data, axis=0), and
any other call sharing that parameter regardless of function name --
confirmed directly: see sites.py's own test, fillna and skew both match,
sorted (genuinely unrelated) does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from memetic.sites import Site


@dataclass
class StructuralPrecondition:
    """Cheap, deterministic, NO LLM. A site must satisfy ALL given
    conditions (None = "no constraint on this dimension") to be a candidate
    for this skill -- Stage A's whole job, and ONLY this stage's job:
    narrow the search space cheaply, never decide final applicability."""
    site_types: List[str]
    operators: Optional[List[str]] = None      # coarse operator CLASSES when
                                               # genuinely meaningful (e.g.
                                               # ["Lt","LtE"] for a boundary
                                               # skill) -- NEVER a literal
                                               # function/method name
    keywords_any: Optional[List[str]] = None   # site must have >=1 of these
                                               # keyword arg NAMES (call sites)
    parent_types: Optional[List[str]] = None

    def matches(self, site: Site) -> bool:
        if site.site_type not in self.site_types:
            return False
        if self.operators is not None and site.operator not in self.operators:
            return False
        if self.keywords_any is not None and not any(k in site.keywords for k in self.keywords_any):
            return False
        if self.parent_types is not None and site.parent_type not in self.parent_types:
            return False
        return True


@dataclass
class SkillExample:
    """ONE concrete case that contributed to a Skill -- either the founding
    case that created it, or a later case dedup.resolve_skill() judged
    "same" and merged in. This is the AUDIT TRAIL: unlike manifestations
    (plain diff strings, generation-guidance only, see below), this
    records exactly where each example came from, so distinct_tasks/
    valid_applications are checkable against real data rather than bare
    counters nobody can verify later, and so a run can be reviewed case by
    case ("which examples created this skill, and were they really the
    same mechanism") the same way judge_skill_applicability's verdicts
    were hand-validated."""
    source: str          # bugbench source, e.g. "human", "qwen7b"
    task_id: str          # e.g. "BigCodeBench/32"
    diff: str              # the real diff (same text as in manifestations)
    localization: str      # "aligned" or "failure_grounded" -- see distill.py


@dataclass
class Skill:
    skill_id: str
    name: str                    # short label, e.g. "axis direction flip"
    family: str                  # broader grouping (several skills can share one)
    mechanism: str                # NL description of the underlying mistake
    structural: StructuralPrecondition
    semantic_preconditions: str   # NL, for Stage B's LLM applicability check
    negative_conditions: str      # NL, when this skill should NOT apply
    manifestations: List[str] = field(default_factory=list)  # real example
                                   # diffs -- GUIDANCE FOR GENERATION ONLY.
                                   # INVARIANT: never read by matches() or
                                   # any future matching/ranking code --
                                   # only structural/semantic preconditions
                                   # may determine applicability. Using
                                   # manifestations as a similarity key would
                                   # silently recreate construct-matching's
                                   # exact problem (one specific edit
                                   # standing in for the whole mechanism).
                                   # diffs -- GUIDANCE, not mandatory recipes
    examples: List[SkillExample] = field(default_factory=list)  # the SAME
                                   # underlying cases as manifestations, but
                                   # with full provenance (source, task_id,
                                   # localization) attached -- the audit
                                   # trail. Also subject to the invariant
                                   # above: never read by matching/ranking,
                                   # inspection/audit only.
    # statistics (Phase 8, not populated yet -- see project notes on EMA
    # difficulty vs. a permanent static value)
    distinct_tasks: int = 0
    valid_applications: int = 0
    recent_use_count: int = 0
    solve_rate_ema: Optional[float] = None
    last_evaluated_round: int = 0


# NOTE: bank persistence (save/load) and the live, mutable, keyed
# collection of Skill objects that the generator retrieves from and that
# distillation writes into -- both at init time and, later, continuously
# during self-play (generator-redesign-note.md section 7) -- live in
# skill_bank.SkillBank, not here. This module is deliberately just the data
# model (Skill/StructuralPrecondition) plus the hand-authored debug/
# baseline seed bank; see skill_bank.py.

# ---- hand-authored seed skills, each grounded in a CONFIRMED real bug -----

SEED_SKILLS: List[Skill] = [
    Skill(
        skill_id="axis_direction_flip",
        name="axis direction flip",
        family="aggregation_direction",
        mechanism="A row-wise vs. column-wise (or similar directional) "
                  "parameter on an aggregation/reduction call is flipped to "
                  "the wrong direction, silently mixing data across the "
                  "wrong dimension instead of raising an error.",
        structural=StructuralPrecondition(
            site_types=["call"], keywords_any=["axis", "dim"]),
        semantic_preconditions="The call's result is used for a per-row or "
                  "per-column computation (e.g. filling missing values, "
                  "computing a statistic, reducing a DataFrame/array) where "
                  "the axis genuinely changes the meaning of the result.",
        negative_conditions="The axis value is fixed/irrelevant to "
                  "correctness for this specific call (e.g. a 1-D array "
                  "where only one axis is even valid).",
        manifestations=[
            # CONFIRMED real, from tonight's fixer-eval regressions:
            "-    df = df.fillna(df.mean(axis=1))\n+    df = df.fillna(df.mean(axis=0))",
            "-    skewness = skew(data_matrix, axis=0)\n+    skewness = skew(data_matrix, axis=1)",
        ],
    ),
    Skill(
        skill_id="boolean_negation_flip",
        name="boolean negation flip",
        family="control_flow_inversion",
        mechanism="A boolean condition guarding a branch has its logical "
                  "sense inverted (a `not` added or removed, or the "
                  "comparison flipped), causing the branch to fire under "
                  "exactly the wrong circumstances.",
        structural=StructuralPrecondition(
            site_types=["branch", "boolean_expr"]),
        semantic_preconditions="The condition gates a meaningfully different "
                  "code path depending on its truth value (e.g. an early "
                  "return, a validation check, a default-vs-override choice) "
                  "-- not a condition whose two branches happen to behave "
                  "identically.",
        negative_conditions="The condition's two outcomes are symmetric or "
                  "equivalent in effect (flipping it would not actually "
                  "change behavior).",
        manifestations=[
            # CONFIRMED real:
            "-    if not output_path:\n+    if output_path:",
        ],
    ),
    Skill(
        skill_id="boundary_comparison",
        name="boundary comparison mistake",
        family="boundary_condition",
        mechanism="An inclusive/exclusive boundary in a comparison is "
                  "shifted by one, so an edge element is wrongly included "
                  "or excluded.",
        structural=StructuralPrecondition(
            site_types=["comparison"], operators=["Lt", "LtE", "Gt", "GtE"]),
        semantic_preconditions="One side of the comparison behaves as a "
                  "counter/index and the other as a meaningful bound on a "
                  "sequence/range this code actually indexes or iterates.",
        negative_conditions="Equality at the boundary is intentionally "
                  "valid either way (the off-by-one would not change which "
                  "elements are processed).",
        manifestations=[],   # not directly observed in tonight's regression
                             # sample -- included from the design note's own
                             # worked example + general bug-taxonomy
                             # knowledge, NOT yet confirmed from our own data
    ),
    Skill(
        skill_id="boundary_loop_range",
        name="loop-range boundary mistake",
        family="boundary_condition",   # SAME family as boundary_comparison --
                                       # structurally distinct site type, same
                                       # underlying mechanism (see review: a
                                       # single AND-constrained precondition
                                       # can't span operator-bearing and
                                       # operator-less site types; splitting
                                       # into variants keeps matches() simple
                                       # rather than adding OR-clause support
                                       # to the precondition language)
        mechanism="A loop's range/bound is shifted by one (e.g. range(n) vs. "
                  "range(n+1), or an off-by-one in a manual index bound), "
                  "causing the loop to run one iteration too few or too many.",
        structural=StructuralPrecondition(site_types=["loop_header"]),
        semantic_preconditions="The loop bound is derived from (or should "
                  "correspond to) the length/size of a sequence this loop "
                  "actually indexes or processes.",
        negative_conditions="The exact loop bound doesn't affect correctness "
                  "here (e.g. an idempotent operation, or extra/missing "
                  "iterations have no observable effect).",
        manifestations=[],
    ),
    Skill(
        skill_id="boundary_slice_index",
        name="slice/index boundary mistake",
        family="boundary_condition",   # same family, see boundary_loop_range
        mechanism="A slice or index boundary is shifted by one, wrongly "
                  "including or excluding an edge element of a sequence.",
        structural=StructuralPrecondition(site_types=["subscript"]),
        semantic_preconditions="The subscript/slice bound is derived from "
                  "(or should correspond to) a meaningful length/position in "
                  "the sequence being accessed, not an arbitrary fixed index.",
        negative_conditions="The exact boundary doesn't affect correctness "
                  "here (e.g. accessing a fixed, always-present element).",
        manifestations=[],
    ),
    Skill(
        skill_id="wrong_statistic",
        name="wrong statistical measure",
        family="wrong_aggregate",
        mechanism="The wrong summary statistic or aggregate function is "
                  "used in place of the intended one (e.g. mean instead of "
                  "median, sum instead of count).",
        structural=StructuralPrecondition(site_types=["call"]),
        semantic_preconditions="The call computes a summary/aggregate value "
                  "from a collection or column where a DIFFERENT specific "
                  "statistic was clearly the intended one from context "
                  "(variable naming, surrounding code, docstring).",
        negative_conditions="No other statistic would plausibly fit here -- "
                  "swapping would not read as a realistic mistake.",
        manifestations=[
            # CONFIRMED real:
            "-    df['median'] = df['list'].apply(np.median)\n+    df['median'] = df['list'].apply(np.mean)",
        ],
    ),
    Skill(
        skill_id="wrong_return_value",
        name="wrong value returned",
        family="return_value_mismatch",
        mechanism="A function returns a different (but superficially "
                  "related) object than the one the caller actually needs "
                  "-- e.g. a view/method-result instead of the object "
                  "itself, or an unrelated variable with a similar name.",
        structural=StructuralPrecondition(site_types=["return"]),
        semantic_preconditions="A DIFFERENT variable or expression is "
                  "available at the return point that a reader could "
                  "plausibly confuse with the correct one (similar name, "
                  "similar type, or derived from it via one extra method "
                  "call).",
        negative_conditions="No plausible alternative value exists in scope "
                  "-- there is nothing a real mistake could substitute.",
        manifestations=[
            # CONFIRMED real:
            "-    return mean_dict\n+    return mean_dict.values()",
            "-    return mean_dict\n+    return random_dict",
        ],
    ),
    Skill(
        skill_id="wrong_variable_reference",
        name="wrong variable/path reference",
        family="reference_mismatch",
        mechanism="An operation (file access, removal, lookup) uses the "
                  "wrong -- but similarly-named or similarly-typed -- "
                  "variable than the one that was actually intended, "
                  "typically a leftover or a confusingly-named sibling.",
        structural=StructuralPrecondition(
            site_types=["call", "subscript"]),
        semantic_preconditions="Another variable is in scope at this point "
                  "that plays a similar role (e.g. both are paths, both are "
                  "the same collection at different stages) and could "
                  "plausibly be substituted by mistake.",
        negative_conditions="No similarly-typed alternative variable exists "
                  "in scope to substitute.",
        manifestations=[
            # CONFIRMED real:
            "-    os.remove(file)\n+    os.remove(backup_file)",
            "-    if not os.path.exists(directory):\n+    if not os.path.exists(backup_dir):",
        ],
    ),
]
