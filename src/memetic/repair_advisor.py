"""Production read path for the repair-memory system -- the fixer's "advisor".

    diagnose the mechanism (v2: assertion-grounded; quotes the faulty construct)
      -> build the query FAULT FINGERPRINT from the quoted construct (fault.py)
      -> fault-fingerprint retrieve from the RepairBank (operation-token primary +
         symptom class, embedding as tie-break)
      -> NULL gate (synthesizer may decline if nothing genuinely applies)
      -> synthesize a specific, actionable fix instruction from the retrieved
         repair_strategy OPERATOR + repair shape (or return None on abstain)

Design invariants (do not regress):
  * The synthesizer is a SINGLE policy; the caller decides its model. Holding it
    fixed across "no-memory" and "memory" isolates memory's contribution.
  * Recognition/diagnosis is symptom-side only (buggy code + failure). The bank's
    write side saw the fix; the read side never does.
  * NULL gate: if no retrieved skill genuinely matches, `advise` returns
    instruction=None (abstained=True); the loop injects nothing -> naked fallback.
  * Leakage guard: pass `exclude_task_id` so a skill distilled from the very task
    now being fixed is never retrieved back into it.

Every model call is wrapped in a hard timeout; a timed-out/empty call degrades
gracefully (empty diagnosis -> no retrieval -> abstain).
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from memetic.fixer_memory import RepairBank, RepairSkill, normalize_failure
from memetic import fault as F


# --------------------------------------------------------------- model I/O ----
# Some gateways prepend a fixed boilerplate line to every completion; strip it so
# it never leaks into advice/distilled text. Set GATEWAY_BOILERPLATE to that line
# for your provider; defaults to a harmless no-op phrase.
_EXEC = ThreadPoolExecutor(max_workers=8)
_GW_PHRASE = os.environ.get("GATEWAY_BOILERPLATE", "llm gateway api layer").strip().lower()
_GW_PREFIX = re.compile(r"(?im)^\s*" + re.escape(_GW_PHRASE) + r"\s*[:.\-]*\s*")


def _clean(s: str) -> str:
    s = _GW_PREFIX.sub("", s or "")
    out = []
    for line in s.splitlines():
        core = line.strip().strip("#").strip().strip("\"'").strip()
        if core.lower() == _GW_PHRASE:
            continue
        out.append(line)
    return "\n".join(out).strip()


def _timed(fn, *args, timeout: float = 90.0, default: str = "") -> str:
    try:
        return _EXEC.submit(fn, *args).result(timeout=timeout) or default
    except Exception:
        return default


def _gen(policy, prompt: str, *, timeout: float = 90.0, tries: int = 3) -> str:
    for _ in range(max(1, tries)):
        out = _clean(_timed(policy.generate, prompt, timeout=timeout))
        if out:
            return out
    return ""


# ------------------------------------------------------------- data types ----
@dataclass
class Advice:
    instruction: Optional[str]
    abstained: bool
    diagnosis: str
    failure_class: str
    retrieved: List[dict] = field(default_factory=list)
    selected_skill_ids: List[str] = field(default_factory=list)
    q_fault: Tuple[str, str, tuple] = ("", "", ())
    q_symptom: str = ""

    @property
    def fired(self) -> bool:
        return not self.abstained and bool(self.instruction)


# ---------------------------------------------------------------- prompts ----
_ASSERT_PAT = re.compile(
    r"(AssertionError|[A-Za-z]*Error|Exception|!=|Lists differ|Tuples differ|"
    r"assert\w*\(|expected|actual|Traceback)", re.I)


def _failure_signal(error: str, limit: int = 1800) -> str:
    """Pull the diagnostic core out of raw test output: the assertion/exception
    lines + the expected-vs-actual the old [:500] truncation often dropped."""
    s = error or ""
    sig = [ln for ln in s.splitlines() if _ASSERT_PAT.search(ln)]
    if not sig:
        return s[:limit]
    return ("\n".join(sig[:20]) + "\n...\n" + s[-900:])[:limit]


_DIAGNOSE_V2 = (
    "A Python function is failing its tests. Diagnose the ACTUAL cause, grounded in "
    "the specific failing assertion -- not a generic category.\n\n"
    "CODE:\n```python\n{code}\n```\n\n"
    "FAILING TEST OUTPUT (assertion + traceback):\n{sig}\n\n"
    "Work in this order, 2-3 short sentences total:\n"
    "1. State exactly what the test EXPECTED and what the code PRODUCED -- quote the "
    "values/types from the assertion.\n"
    "2. Point to the SINGLE construct in the code that produces that specific "
    "difference -- quote the exact expression in backticks.\n"
    "3. Name the mechanism in a few words.\n"
    "Do NOT guess. Do NOT say the bug is about the return value/shape UNLESS the "
    "assertion is literally about what is returned. Reply with only the sentences."
)

_SYNTH_NOMEM = (
    "You are writing a precise fix instruction for a code fixer.\n\n"
    "CURRENT BUGGY CODE:\n```python\n{code}\n```\n\n"
    "FAILING TEST OUTPUT:\n{err}\n\n"
    "Diagnosed mechanism: {dx}\n\n"
    "Write a SPECIFIC, ACTIONABLE instruction for THIS code: name the exact "
    "construct that is wrong and the exact change to make (e.g. 'In the mean() "
    "call, change axis=0 to axis=1.'). 1-3 sentences. Do NOT output the full "
    "corrected function."
)

_SKILL_BLOCK = (
    "[{i}] {name}\n"
    "    root_cause: {rc}\n"
    "    repair_strategy (the operator to apply): {rs}{shape}{neg}{ex}"
)

_NULL = "NULL"


def _is_abstain(raw: str) -> bool:
    t = (raw or "").strip()
    if not t:
        return False
    if t.upper().rstrip(".:") == _NULL:
        return True
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    return bool(lines) and lines[-1].upper().rstrip(".:") == _NULL

_SYNTH_MEM = (
    "You are writing a precise fix instruction for a code fixer, and you may draw "
    "on past repair skills retrieved from a memory of verified fixes.\n\n"
    "CURRENT BUGGY CODE:\n```python\n{code}\n```\n\n"
    "FAILING TEST OUTPUT:\n{err}\n\n"
    "Diagnosed mechanism: {dx}\n\n"
    "RETRIEVED REPAIR SKILLS (reference for the repair pattern, NOT to copy verbatim):\n"
    "{skills}\n\n"
    "Select the SINGLE retrieved skill whose repair_strategy genuinely matches THIS "
    "bug's mechanism and applies to THIS code. Judge each skill on its own merits; do "
    "NOT combine, average, or blend across skills, and do not use a skill that only "
    "loosely resembles the bug. If the skill notes a repair shape touching several "
    "sites, make sure your instruction covers ALL of them, not just the obvious one. "
    f"If NONE genuinely applies, reply with exactly {_NULL} and nothing else.\n"
    "Otherwise, using ONLY that one matching skill's repair_strategy operator, write a "
    "SPECIFIC, ACTIONABLE instruction for THIS code: name the exact construct that is "
    "wrong and the exact change to make (e.g. 'In the mean() call, change axis=0 to "
    "axis=1.'). 1-3 sentences. Do NOT output the full corrected function."
)


def _render_skills(rows: List[Tuple[float, dict, RepairSkill]]) -> str:
    out = []
    for i, (_sc, _ch, s) in enumerate(rows):
        neg = f"\n    do_NOT: {s.negative_conditions}" if s.negative_conditions else ""
        shape = ""
        if s.shape_tag and s.shape_tag not in ("", "single-edit", "unknown"):
            ops = f"; operations: {', '.join(s.op_multiset)}" if s.op_multiset else ""
            shape = f"\n    repair shape: {s.shape_tag} across ~{s.n_fix_sites} site(s){ops}"
        ex = ""
        if s.manifestations:
            ex = f"\n    example fix of this kind:\n{(s.manifestations[0] or '')[:280]}"
        out.append(_SKILL_BLOCK.format(i=i + 1, name=s.name, rc=s.root_cause,
                                       rs=s.repair_strategy, shape=shape, neg=neg, ex=ex))
    return "\n".join(out)


# ------------------------------------------------------------- the advisor ----
class RepairAdvisor:
    def __init__(self, bank: RepairBank, synth_policy, *,
                 retrieve_k: int = 3, gate: bool = True, timeout: float = 90.0):
        self.bank = bank
        self.pol = synth_policy
        self.k = retrieve_k
        self.gate = gate
        self.timeout = timeout

    def _skill_task_ids(self, skill) -> set:
        return {self.bank.cases[c].task_id for c in skill.case_ids
                if c in self.bank.cases}

    # -- pieces (public so the loop can log / reuse) --------------------
    def diagnose(self, buggy_full: str, error: str) -> str:
        """v2 assertion-grounded diagnosis -- quotes the faulty construct in
        backticks so the query fault fingerprint can be localized from it."""
        sig = _failure_signal(error)
        out = _gen(self.pol, _DIAGNOSE_V2.format(code=buggy_full, sig=sig),
                   timeout=self.timeout)
        return " ".join(out.split())[:400] if out else ""

    def query_fingerprint(self, diagnosis: str, buggy_full: str, error: str):
        """(q_fault, q_symptom) for retrieval, from the v2 diagnosis's quoted
        construct and the raw failure."""
        spans = F.spans_from_diagnosis(diagnosis)
        q_fault = F.fault_fingerprint(buggy_full, spans)
        q_sym = F.symptom_class(error)
        return q_fault, q_sym

    def retrieve(self, diagnosis: str, buggy_full: str, q_fault, q_symptom: str, *,
                 exclude_task_id: Optional[str] = None
                 ) -> List[Tuple[float, dict, RepairSkill]]:
        if not diagnosis:
            return []
        got = self.bank.retrieve(diagnosis, k=self.k + 4, q_fault=q_fault,
                                 q_symptom=q_symptom, with_scores=True)
        rows = []
        for (score, ch, s) in got:
            if exclude_task_id and exclude_task_id in self._skill_task_ids(s):
                continue
            rows.append((score, ch, s))
            if len(rows) >= self.k:
                break
        return rows

    # -- the full read path ---------------------------------------------
    def advise(self, buggy_full: str, error: str, *,
               exclude_task_id: Optional[str] = None) -> Advice:
        fclass, _ = normalize_failure(error)
        dx = self.diagnose(buggy_full, error)
        q_fault, q_sym = self.query_fingerprint(dx, buggy_full, error)
        rows = self.retrieve(dx, buggy_full, q_fault, q_sym, exclude_task_id=exclude_task_id)

        retrieved = [{"skill_id": s.skill_id, "name": s.name, "score": round(sc, 3),
                      "sem": ch.get("sem"), "op": ch.get("op"), "site": ch.get("site"),
                      "sym": ch.get("sym"),
                      "fault": f"{s.site_type}/{s.operator}" if s.site_type else "",
                      "skill_symptom": s.symptom_class, "shape": s.shape_tag,
                      "repair_strategy": s.repair_strategy}
                     for (sc, ch, s) in rows]

        base = dict(diagnosis=dx, failure_class=fclass, retrieved=retrieved,
                    q_fault=q_fault, q_symptom=q_sym)

        if not rows:
            return Advice(instruction=None, abstained=True, **base)

        prompt = _SYNTH_MEM.format(code=buggy_full, err=(error or "")[:500], dx=dx,
                                   skills=_render_skills(rows))
        raw = _gen(self.pol, prompt, timeout=self.timeout).strip()

        if self.gate and _is_abstain(raw):
            return Advice(instruction=None, abstained=True, **base)

        instr = raw[:500] if raw and not _is_abstain(raw) else ""
        if not instr:
            instr = _gen(self.pol, _SYNTH_NOMEM.format(code=buggy_full,
                         err=(error or "")[:500], dx=dx), timeout=self.timeout).strip()[:500]
            if not instr:
                return Advice(instruction=None, abstained=True, **base)

        sel = [s.skill_id for (_sc, _ch, s) in rows]
        for (_sc, _ch, s) in rows:
            s.selection_count += 1
        return Advice(instruction=instr, abstained=False, selected_skill_ids=sel, **base)

    # -- injection helper -----------------------------------------------
    @staticmethod
    def as_cap(advice: Advice) -> str:
        if not advice or not advice.instruction:
            return ""
        return (f"A diagnosis of this bug and how to fix it:\n{advice.instruction}\n\n"
                "Apply this correction to the code.")
