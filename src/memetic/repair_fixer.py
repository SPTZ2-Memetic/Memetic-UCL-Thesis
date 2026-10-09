"""The fixer episode driver -- one bug, K ESCALATION trials, memory read + write-back.

ESCALATION (the fixer's "overall capability, with or without skills"):
  Each of K trials first tries the fixer NAKED; only if that attempt fails does it
  escalate -- diagnose + retrieve + inject a repair instruction (computed once per
  bug, cached) and retry. The trial is SOLVED if either the naked or the memory
  attempt passes. solve_rate = solved_trials / K is therefore the fixer's overall
  solve capability -- which is what the generator's curriculum tracks, so the
  generator stays on the frontier of the WHOLE fixer system as memory improves.

Why this shape:
  * Memory never DRAGS DOWN a bug the fixer already solves naked (it is only
    consulted on a naked miss), so it can only ever add -- the harm the eval saw
    from injecting on easy bugs is gone by construction.
  * The repair-bank utility signal is clean: a skill is credited/blamed only on
    trials where it was actually invoked to rescue a naked failure (skill_passed),
    separate from the episode's overall solve (which drives learn-on-win).
  * Gateway cost is bounded: easy bugs run ~K naked calls; memory calls happen only
    on naked misses.

Reuses build_fix_prompt/_FIX_SENTINEL from fixer.py (the baseline prompt +
injection point) and the verifier. Every model call is hard-timeout-wrapped.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from memetic.fixer import build_fix_prompt, _FIX_SENTINEL
from memetic.codeutils import parse_and_extract_function
from memetic.seed import short_diff
from memetic.repair_advisor import RepairAdvisor, Advice


_EXEC = ThreadPoolExecutor(max_workers=8)


def _timed_generate(policy, prompt: str, *, timeout: float = 90.0) -> str:
    try:
        return _EXEC.submit(policy.generate, prompt).result(timeout=timeout) or ""
    except Exception:
        return ""


def _gen_nonempty(policy, prompt: str, *, timeout: float = 90.0, tries: int = 3) -> str:
    """gpt-5-nano intermittently returns an empty visible completion (reasoning eats
    the budget). Retry until non-empty; starvation empties return fast."""
    for _ in range(tries):
        out = _timed_generate(policy, prompt, timeout=timeout).strip()
        if out:
            return out
    return ""


@dataclass
class FixEpisode:
    solved: bool
    solve_rate: float                 # solved_trials / trials -> generator difficulty oracle
    passes: int                       # solved trials
    attempts_made: int                # trials run
    advice: Optional[Advice] = None    # read-path output (None if memory never consulted)
    injected: bool = False             # was a memory instruction ever injected
    learned_outcome: Optional[str] = None
    fix_diff: str = ""                 # first winning diff (naked or memory)
    mem_invoked: int = 0               # trials where memory was tried (naked had failed)
    mem_solved: int = 0                # trials where the memory attempt passed
    attempts: List[dict] = field(default_factory=list)


def run_fix_episode(
    record,
    bug_body: str,
    bug_output: str,
    *,
    fixer_policy,
    verifier,
    bank,
    advisor: RepairAdvisor,
    distill_policy,
    k: int = 4,
    round_idx: int = 0,
    exclude_task_id: Optional[str] = None,
    write_back: bool = True,
    stop_on_first_pass: bool = False,
    timeout: float = 90.0,
) -> FixEpisode:
    """One fixer episode on one verified bug, run as K escalation trials."""
    ep = record.get("entry_point") or "task_func"
    cp = record.get("code_prompt", "") or ""
    buggy_full = cp + bug_body
    task_id = str(record.get("task_id"))
    if exclude_task_id is None:
        exclude_task_id = task_id

    advice: Optional[Advice] = None    # diagnosed+retrieved once, lazily, on first naked miss
    cap = ""
    feedback = bug_output

    solved_trials = 0
    mem_invoked = 0
    mem_solved = 0
    first_win_body: Optional[str] = None
    attempts: List[dict] = []
    made = 0

    def _attempt(inject_cap: str):
        """One fixer generation + verify, threading `feedback`. Returns (passed, body)."""
        nonlocal feedback
        prompt = build_fix_prompt(record, bug_body, feedback)
        if inject_cap:
            prompt = prompt.replace(_FIX_SENTINEL, inject_cap + "\n" + _FIX_SENTINEL, 1)
        raw = _gen_nonempty(fixer_policy, prompt, timeout=timeout)
        body, _ = parse_and_extract_function(raw, ep)
        if body is None:
            feedback = "fix did not parse into a function body"
            return False, None
        res = verifier.verify(record, cp + body)
        feedback = res.output
        return bool(res.passed), body

    for i in range(k):
        made += 1
        # 1) naked attempt
        npass, nbody = _attempt("")
        trial_solved = npass
        if npass and first_win_body is None:
            first_win_body = nbody

        # 2) escalate only on a naked miss
        escalated = mpass = False
        if not npass:
            if advice is None:                       # diagnose + retrieve once per bug
                advice = advisor.advise(buggy_full, bug_output, exclude_task_id=exclude_task_id)
                cap = RepairAdvisor.as_cap(advice)
            if cap:
                escalated = True
                mem_invoked += 1
                mpass, mbody = _attempt(cap)
                if mpass:
                    trial_solved = True
                    mem_solved += 1
                    if first_win_body is None:
                        first_win_body = mbody

        if trial_solved:
            solved_trials += 1
        attempts.append({"trial": i, "naked_passed": npass,
                         "escalated": escalated, "mem_passed": mpass})
        if trial_solved and stop_on_first_pass:
            break

    solved = solved_trials > 0
    solve_rate = (solved_trials / made) if made else 0.0
    fix_diff = short_diff(bug_body, first_win_body) if first_win_body else ""
    injected = mem_invoked > 0

    # 3) WRITE-BACK -- skills judged on their RESCUE (skill_passed), learning on the
    #    overall solve (whichever attempt won).
    learned = None
    if write_back:
        sel = tuple(advice.selected_skill_ids) if (advice and injected) else ()
        learned = bank.record_fix_episode(
            buggy_full, bug_output, distill_policy,
            passed=solved, skill_passed=(mem_solved > 0),
            fixer_fix_diff=fix_diff, injected_skill_ids=sel,
            task_id=task_id, source="fixer", round_idx=round_idx)

    return FixEpisode(
        solved=solved, solve_rate=solve_rate, passes=solved_trials, attempts_made=made,
        advice=advice, injected=injected, learned_outcome=learned, fix_diff=fix_diff,
        mem_invoked=mem_invoked, mem_solved=mem_solved, attempts=attempts)
