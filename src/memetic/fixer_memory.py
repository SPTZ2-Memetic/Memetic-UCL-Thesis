"""Fixer repair-memory -- the proper system (not the throwaway probes).

Two layers:
  RepairSkill : an abstract, reusable repair pattern -- recognition (query-side
                signals) -> root_cause (mechanism) -> repair_strategy (a
                PARAMETRIZED EDIT OPERATOR) -> negatives. Carries a FAULT
                FINGERPRINT (operation token + symptom class + repair shape) for
                discriminative retrieval, and utility stats for the NULL gate.
  RepairCase  : one concrete verified experience.

RETRIEVAL (v2 -- fault-fingerprint). The old hybrid key (semantic + a 3-bucket
failure_class + a whole-function structural existence check) SATURATED: failure_class
was constant within a stratum and the structural channel fired on ~every function
(it checked "does the function contain a site of this type anywhere"), so ranking
collapsed to a weak prose cosine. The v2 key keys on the diagnosed FAULT:
  - operation token : the callee/operator at the fault site (re.findall, astype,
                      Lt ...) -- what encodes the library + semantics. PRIMARY.
  - symptom class   : wrong_type / wrong_shape_len / wrong_value / missing_element /
                      crash:<kind> -- read off expected-vs-actual.
  - semantic        : embedding of recognition + root_cause -- TIE-BREAK only.
See memetic.fault. The query-side fault fingerprint is built by the advisor
from the v2 diagnosis's quoted construct; the skill-side one from its fix diff.

Leakage boundary: recognition/root_cause and the fingerprint are all derivable
from read-time signals (buggy code + failure); the distiller sees the fix
(write-time supervision) but is told to keep recognition symptom-side.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from memetic.embeddings import Embedder, cosine_similarity
from memetic.codeutils import error_signature, error_kind, CRASH_KINDS
from memetic.sites import extract_meaningful_sites
from memetic import fault as F


# ---------------------------------------------------------------- schema ----
@dataclass
class RepairCase:
    task_id: str
    source: str
    buggy_local: str
    failure: str
    failure_class: str
    fixed_diff: str
    repair_summary: str
    outcome: str = "canonical"   # canonical | fixer_pass | fixer_fail(negative)
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_round: int = 0


@dataclass
class RepairSkill:
    skill_id: str
    name: str
    family: str
    root_cause: str
    recognition: str
    repair_strategy: str
    negative_conditions: str = ""
    # ---- retrieval channels (all read-time-derivable) ----
    failure_class: str = ""
    site_type: str = ""
    operator: str = ""
    keywords: Tuple[str, ...] = ()
    # ---- v2 fault fingerprint (new; default-safe so old banks still load) ----
    symptom_class: str = ""
    shape_tag: str = ""
    op_multiset: Tuple[str, ...] = ()
    n_fix_sites: int = 0
    # ---- evidence ----
    case_ids: List[str] = field(default_factory=list)
    manifestations: List[str] = field(default_factory=list)
    negatives: List[str] = field(default_factory=list)
    # ---- utility stats (drive the NULL gate) ----
    distinct_tasks: int = 0
    retrieval_count: int = 0
    selection_count: int = 0
    naked_baseline_success: int = 0
    success_with_skill: int = 0
    rescue_count: int = 0
    harm_count: int = 0
    estimated_uplift: float = 0.0
    paired_n: int = 0
    status: str = "cold"     # cold | active | quarantined | merged
    last_round: int = 0
    created_round: int = 0

    def retrieval_text(self) -> str:
        return f"{self.name}. {self.root_cause} {self.recognition}".strip()


# ------------------------------------------------------ failure normalize ----
def normalize_failure(error: str) -> Tuple[str, str]:
    """(failure_class, one-line symptom) -- query-side, no fix needed. Kept for the
    coarse back-compat channel and the negative-symptom log; the discriminative
    signal is now fault.symptom_class."""
    sig = error_signature(error or "")
    k = error_kind(error or "")
    if k in CRASH_KINDS:
        fclass = f"exception:{k}"
    elif k == "assertion":
        fclass = "value_mismatch"
    elif k == "timeout":
        fclass = "timeout"
    else:
        fclass = "unknown"
    return fclass, sig[:160]


# ----------------------------------------------------- structural channel ----
# (legacy -- kept for the back-compat retrieve path; the v2 path uses fault.py)
def _site_tuple(s) -> Tuple[str, str, Tuple[str, ...]]:
    return (s.site_type, s.operator, tuple(s.keywords))


def query_sites(buggy_full: str) -> List[Tuple[str, str, Tuple[str, ...]]]:
    return [_site_tuple(s) for s in extract_meaningful_sites(buggy_full)]


def _struct_match(skill: RepairSkill, q_sites) -> float:
    if not skill.site_type:
        return 0.0
    for (st, op, kw) in q_sites:
        if st != skill.site_type:
            continue
        if skill.operator and op != skill.operator:
            continue
        if skill.keywords and not (set(skill.keywords) & set(kw)):
            continue
        return 1.0
    return 0.0


# ------------------------------------------------------------- distiller ----
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _parse_json(out: str) -> Optional[dict]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", out, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = _JSON_RE.search(out or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


_DISTILL_PROMPT = """\
You are distilling ONE reusable code-repair skill from a VERIFIED bug->fix example.

FAILING TEST OUTPUT (this is what a future fixer will also see):
{failure}

BUGGY CODE:
{buggy}

VERIFIED FIX (unified diff, '-'=buggy '+'=correct):
{diff}

Produce reusable repair knowledge, not a description of this one case.

Rules:
- recognition: what a future fixer could observe FROM THE BUGGY CODE AND THE
  FAILURE ALONE (never from the fix) that signals this kind of bug. Symptom-side only.
- root_cause: the underlying mechanism, one concrete sentence (NOT a vague
  category like "input validation error").
- repair_strategy: an ACTIONABLE operator -- specific about the KIND of change to
  make and how to decide it, general about exact tokens. Good: "identify the
  reduction call whose axis controls output orientation; infer the intended axis
  from the task's expected output shape and set it; do not copy a prior value."
  Bad (vague): "fix the axis". Bad (too specific): "change axis=0 to axis=1".
- negative_conditions: when this should NOT be applied / a plausible-but-wrong move.
- name: 3-6 words. family: a broad group.

Return ONLY JSON:
{{"name":..., "family":..., "root_cause":..., "recognition":..., "repair_strategy":..., "negative_conditions":..., "repair_summary":...}}
"""


def distill_repair_skill(buggy_full: str, fixed_diff: str, failure: str, policy,
                         *, task_id: str = "", source: str = "",
                         round_idx: int = 0) -> Optional[Tuple[RepairSkill, RepairCase]]:
    """Write-time distillation from a verified buggy->fix pair. `buggy_full` is
    the full buggy function source (code_prompt + body) so sites can be
    extracted. Returns a provisional (skill, case) or None."""
    if not (fixed_diff or "").strip():
        return None
    prompt = _DISTILL_PROMPT.format(failure=(failure or "")[:600],
                                    buggy=buggy_full[:1600], diff=fixed_diff[:1200])
    d = None
    for _ in range(3):   # reasoning models intermittently return empty/non-JSON
        try:
            out = policy.generate(prompt)
        except Exception:
            out = ""
        d = _parse_json(out)
        if d and d.get("repair_strategy") and d.get("root_cause"):
            break
        d = None
    if d is None:
        return None
    fclass, _ = normalize_failure(failure)
    # v2 fault fingerprint (operation token at the fix site + symptom + repair shape)
    st, op, kw = F.fix_fingerprint(buggy_full, fixed_diff)
    sym = F.symptom_class(failure)
    shape = F.fix_shape(buggy_full, fixed_diff)
    case = RepairCase(task_id=task_id, source=source, buggy_local=buggy_full[:1600],
                      failure=(failure or "")[:600], failure_class=fclass,
                      fixed_diff=fixed_diff, repair_summary=str(d.get("repair_summary", ""))[:200],
                      created_round=round_idx)
    skill = RepairSkill(
        skill_id=uuid.uuid4().hex, name=str(d.get("name", ""))[:80],
        family=str(d.get("family", ""))[:60], root_cause=str(d.get("root_cause", "")),
        recognition=str(d.get("recognition", "")), repair_strategy=str(d.get("repair_strategy", "")),
        negative_conditions=str(d.get("negative_conditions", "")),
        failure_class=fclass, site_type=st, operator=op, keywords=tuple(kw),
        symptom_class=sym, shape_tag=shape.get("shape", ""),
        op_multiset=tuple(shape.get("op_multiset", ())), n_fix_sites=int(shape.get("n_sites", 0)),
        case_ids=[case.case_id], manifestations=[fixed_diff],
        distinct_tasks=1, created_round=round_idx, last_round=round_idx)
    return skill, case


# -------------------------------------------------------- consolidation ----
_RELATION_PROMPT = """\
Two code-repair skills. Decide if the NEW one is the SAME underlying repair as
the EXISTING one, a VARIANT (same family, meaningfully different recognition or
strategy), or DIFFERENT.

EXISTING
root_cause: {e_rc}
recognition: {e_rec}
repair_strategy: {e_rs}

NEW
root_cause: {n_rc}
recognition: {n_rec}
repair_strategy: {n_rs}

SAME = same mechanism AND substantially the same repair reasoning. VARIANT = same
broad cause but applicability/strategy differs enough that merging loses info.
DIFFERENT = different cause and/or repair.
Return ONLY JSON: {{"relation":"same|variant|different","reason":"one sentence"}}
"""


def judge_repair_relation(existing: RepairSkill, cand: RepairSkill, policy) -> str:
    prompt = _RELATION_PROMPT.format(
        e_rc=existing.root_cause, e_rec=existing.recognition, e_rs=existing.repair_strategy,
        n_rc=cand.root_cause, n_rec=cand.recognition, n_rs=cand.repair_strategy)
    try:
        d = _parse_json(policy.generate(prompt)) or {}
        rel = str(d.get("relation", "different")).strip().lower()
        return rel if rel in ("same", "variant", "different") else "different"
    except Exception:
        return "different"     # fail-closed: never silently merge


# ---------------------------------------------------------------- bank ----
class RepairBank:
    # legacy fusion weights (back-compat retrieve path only)
    W_FAILCLASS = 0.20
    W_STRUCT = 0.12
    # v2 fault-fingerprint weights: operation token PRIMARY, symptom next, embedding
    # as a tie-break. An exact operation match (+0.60) dominates a weak prose gap.
    W_OP = 0.60
    W_SITE = 0.15
    W_SYM = 0.25
    W_KW = 0.10

    MIN_OUTCOMES_FOR_STATUS = 4
    QUARANTINE_FLOOR = 0.15
    ACTIVATE_FLOOR = 0.34

    def __init__(self, embedder: Optional[Embedder] = None):
        self.skills: Dict[str, RepairSkill] = {}
        self.cases: Dict[str, RepairCase] = {}
        self._emb = embedder or Embedder()
        self._vecs: Dict[str, np.ndarray] = {}

    def _vec(self, skill: RepairSkill) -> np.ndarray:
        if skill.skill_id not in self._vecs:
            self._vecs[skill.skill_id] = self._emb.embed_one(skill.retrieval_text())
        return self._vecs[skill.skill_id]

    # -- write / consolidate --------------------------------------------
    def admit(self, skill: RepairSkill, case: RepairCase, policy,
              *, text_threshold: float = 0.55, max_llm: int = 5) -> Tuple[str, RepairSkill]:
        self.cases[case.case_id] = case
        if not self.skills:
            return self._add_new(skill, case)
        qv = self._emb.embed_one(skill.retrieval_text())
        scored = sorted(((cosine_similarity(qv, self._vec(s)), s) for s in self.skills.values()),
                        key=lambda x: -x[0])
        checked, best_variant = 0, None
        for sim, existing in scored:
            if sim < text_threshold or checked >= max_llm:
                break
            checked += 1
            rel = judge_repair_relation(existing, skill, policy)
            if rel == "same":
                return self._merge(existing, case)
            if rel == "variant" and best_variant is None:
                best_variant = existing
        if best_variant is not None:
            skill.family = best_variant.family or skill.family
            return self._add_new(skill, case, "variant")
        return self._add_new(skill, case)

    def _add_new(self, skill: RepairSkill, case: RepairCase, outcome: str = "new"):
        self.skills[skill.skill_id] = skill
        self._vecs[skill.skill_id] = self._emb.embed_one(skill.retrieval_text())
        return outcome, skill

    def _merge(self, existing: RepairSkill, case: RepairCase):
        existing.case_ids.append(case.case_id)
        if case.fixed_diff and case.fixed_diff not in existing.manifestations:
            existing.manifestations.append(case.fixed_diff)
        existing.distinct_tasks += 1
        return "same", existing

    # -- write-back from fixer episodes (self-play self-evolution) -------
    def note_skill_outcome(self, skill_id: str, passed: bool, *, round_idx: int = 0) -> None:
        s = self.skills.get(skill_id)
        if s is None:
            return
        if passed:
            s.success_with_skill += 1
        else:
            s.harm_count += 1
        s.last_round = round_idx
        denom = s.success_with_skill + s.harm_count
        if denom:
            s.estimated_uplift = s.success_with_skill / denom
            s.paired_n = denom
        if denom >= self.MIN_OUTCOMES_FOR_STATUS:
            if s.estimated_uplift < self.QUARANTINE_FLOOR:
                s.status = "quarantined"
            elif s.status == "cold" and s.estimated_uplift >= self.ACTIVATE_FLOOR:
                s.status = "active"

    def record_negative(self, skill_id: str, symptom: str) -> None:
        s = self.skills.get(skill_id)
        if s and symptom and symptom not in s.negatives:
            s.negatives.append(symptom[:200])

    def record_fix_episode(self, buggy_full: str, failure: str, policy, *,
                           passed: bool, skill_passed: Optional[bool] = None,
                           fixer_fix_diff: str = "",
                           injected_skill_ids: Tuple[str, ...] = (),
                           task_id: str = "", source: str = "fixer",
                           round_idx: int = 0, learn_on_win: bool = True
                           ) -> Optional[str]:
        """The self-play write-back, called once per finished fixer episode.
        1) utility bookkeeping on every injected skill (drives the quarantine gate);
           on a loss, record the failing symptom as a negative.
        2) on a WIN, learn a NEW skill from the fixer's OWN verified fix and admit it.

        `passed` is the episode's OVERALL solve (used for learn-on-win). `skill_passed`
        is whether MEMORY specifically rescued the bug when it was invoked -- under
        escalation the injected skills should be judged on their rescue, not on the
        overall solve (which may have come from the naked attempt). Defaults to
        `passed` when not given (back-compat)."""
        sp = passed if skill_passed is None else skill_passed
        for sid in injected_skill_ids:
            self.note_skill_outcome(sid, sp, round_idx=round_idx)
            if not sp:
                _, symptom = normalize_failure(failure)
                self.record_negative(sid, symptom)

        if passed and learn_on_win and (fixer_fix_diff or "").strip():
            distilled = distill_repair_skill(buggy_full, fixer_fix_diff, failure, policy,
                                             task_id=task_id, source=source, round_idx=round_idx)
            if distilled:
                skill, case = distilled
                case.outcome = "fixer_pass"
                outcome, _ = self.admit(skill, case, policy)
                return outcome
        return None

    def counts_by_status(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.skills.values():
            out[s.status] = out.get(s.status, 0) + 1
        return out

    # -- read ------------------------------------------------------------
    def retrieve(self, query_text: str, k: int = 3, *,
                 q_failure_class: str = "", q_buggy_full: str = "",
                 q_fault: Optional[Tuple[str, str, tuple]] = None, q_symptom: str = "",
                 active_only: bool = False, with_scores: bool = False):
        """Retrieve top-k skills.

        v2 (preferred): pass q_fault=(site_type, op, keywords) and q_symptom -- the
        query's fault fingerprint, built by the advisor from the diagnosis. Scoring
        is operation-token-primary, symptom next, embedding as tie-break.

        legacy (back-compat for the old probes): if q_fault is None, fall back to the
        saturated semantic + failure_class + whole-function struct fusion.
        """
        if not self.skills:
            return []
        qv = self._emb.embed_one(query_text)
        v2 = q_fault is not None
        q_st, q_op, q_kw = (q_fault if v2 else ("", "", ()))
        q_sites = query_sites(q_buggy_full) if (not v2 and q_buggy_full) else []
        rows = []
        for s in self.skills.values():
            if s.status == "quarantined" or (active_only and s.status != "active"):
                continue
            sem = cosine_similarity(qv, self._vec(s))
            if v2:
                op_m = F.op_match(q_op, s.operator)
                site_m = 1.0 if (q_st and s.site_type and q_st == s.site_type) else 0.0
                sym_m = 1.0 if (q_symptom and s.symptom_class and q_symptom == s.symptom_class) else 0.0
                kw_m = 1.0 if (set(q_kw) & set(s.keywords)) else 0.0
                score = sem + self.W_OP * op_m + self.W_SITE * site_m + self.W_SYM * sym_m + self.W_KW * kw_m
                ch = {"sem": round(float(sem), 3), "op": round(op_m, 2),
                      "site": site_m, "sym": sym_m, "kw": kw_m}
            else:
                fc = 1.0 if (q_failure_class and s.failure_class == q_failure_class) else 0.0
                st = _struct_match(s, q_sites) if q_sites else 0.0
                score = sem + self.W_FAILCLASS * fc + self.W_STRUCT * st
                ch = {"sem": round(float(sem), 3), "fclass": fc, "struct": st}
            rows.append((score, ch, s))
        rows.sort(key=lambda x: -x[0])
        top = rows[:k]
        for _sc, _ch, s in top:
            s.retrieval_count += 1
        if with_scores:
            return [(sc, ch, s) for (sc, ch, s) in top]
        return [s for _sc, _ch, s in top]

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        def _clean(d):    # tuples -> lists for JSON
            d["keywords"] = list(d.get("keywords", []))
            d["op_multiset"] = list(d.get("op_multiset", []))
            return d
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"skills": [_clean(asdict(s)) for s in self.skills.values()],
                       "cases": [asdict(c) for c in self.cases.values()]}, f, indent=2)

    @classmethod
    def load(cls, path: str, embedder: Optional[Embedder] = None) -> "RepairBank":
        b = cls(embedder)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        valid = set(RepairSkill.__dataclass_fields__)
        for s in raw.get("skills", []):
            s = {k: v for k, v in s.items() if k in valid}   # tolerate schema drift
            s["keywords"] = tuple(s.get("keywords", []))
            s["op_multiset"] = tuple(s.get("op_multiset", []))
            b.skills[s["skill_id"]] = RepairSkill(**s)
        for c in raw.get("cases", []):
            b.cases[c["case_id"]] = RepairCase(**c)
        return b

    def __len__(self) -> int:
        return len(self.skills)
