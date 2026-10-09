"""Self-play loop v2 -- NEW generator (SkillBank + curriculum) + NEW fixer
(RepairBank + advisor + escalation write-back), both banks evolving, no weights.

One "round" = one seed. Each round:
  generate_with_curriculum (reads SkillBank) -> for each valid bug, run an
  ESCALATION fixer episode (naked first, memory on miss; see repair_fixer) ->
  update the guiding skill's solve_rate_ema (curriculum moving target) ->
  re-distill productive bugs into the SkillBank -> fixer write-back into the
  RepairBank.

PARALLELISM (for feasible 900-round runs) -- batch_size B:
  A round's work splits into a READ phase (generation + the fixer episodes, which
  only READ both banks) and a COMMIT phase (the three state mutations:
  record_fix_episode, update_skill_ema, redistill_bug). B rounds run their READ
  phase concurrently against a FROZEN bank snapshot, then their COMMITs are applied
  SERIALLY in seed order. So the only dynamics change is update granularity
  (per-seed -> per-batch); B=1 is identical to the fully-sequential loop.
  Reads bump only telemetry counters (retrieval/selection_count), tolerated under
  the GIL; every dynamics-bearing write happens in the serial commit.

No weights move. The only couplings between roles are solve_rate (generator
difficulty) and the two banks. Frontier = 0 < solve_rate < 1.
"""
from __future__ import annotations

import collections
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from memetic.generation import (
    SkilledPair, BugCandidate, retrieve_skilled_pairs,
    generate_skilled, generate_exploration,
)
from memetic.redistill import redistill_bug
from memetic.repair_advisor import RepairAdvisor
from memetic.repair_fixer import run_fix_episode, FixEpisode


# ---------------------------------------------- thread-safe embedder proxy ----
class _LockingEmbedder:
    """Serializes all embedder method calls behind one lock -- sentence-transformer
    encode isn't thread-safe, and the parallel read phase hits the embedder from
    the generator ranking AND the advisor at once. Embeddings are fast and local,
    so serializing them is free relative to the gateway calls we DO parallelize.
    __getattr__ wraps any callable attribute; non-callables pass through."""
    def __init__(self, emb):
        object.__setattr__(self, "_emb", emb)
        object.__setattr__(self, "_lock", threading.Lock())

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_emb"), name)
        if callable(attr):
            lock = object.__getattribute__(self, "_lock")
            def wrapped(*a, **k):
                with lock:
                    return attr(*a, **k)
            return wrapped
        return attr


# ------------------------------------------------------------------ config ----
@dataclass
class SelfPlayConfig:
    n_total: int = 4
    n_skilled_target: int = 3
    fix_attempts: int = 4
    judge_budget: int = 40
    retrieve_slack: int = 3
    frontier_target: float = 0.5
    ema_alpha: float = 0.3
    trivial_ema: float = 0.9
    trivial_min_uses: int = 3
    unseen_priority: float = 0.9
    redistill: bool = True
    dedup_text_threshold: float = 0.40
    save_every: int = 10
    stop_on_first_pass: bool = False
    trace_path: Optional[str] = None
    # longitudinal capture (for the thesis tables -- see supervisor-experiment-asks)
    metrics_path: Optional[str] = None      # one structured line per round
    checkpoint_every: int = 0               # round-tagged bank snapshots every N (0=off)
    checkpoint_dir: Optional[str] = None


# ------------------------------------------------------------- curriculum ----
def _norm_task_id(tid) -> str:
    """Align task ids across datasets that format them differently -- the clean pool
    uses 'bigcodebench_<N>' while the bug sources use 'BigCodeBench/<N>'. The trailing
    integer is the shared BigCodeBench task index, so key reference lookup on that."""
    m = re.search(r"(\d+)\s*$", str(tid))
    return m.group(1) if m else str(tid)


def record_is_runnable(record) -> bool:
    ep = record.get("entry_point") or ""
    cp = record.get("code_prompt") or ""
    return bool(record.get("test")) and (f"def {ep}" in cp)


def update_skill_ema(skill, solve_rate: float, cfg: SelfPlayConfig, round_idx: int) -> None:
    if skill.solve_rate_ema is None:
        skill.solve_rate_ema = solve_rate
    else:
        skill.solve_rate_ema = (cfg.ema_alpha * solve_rate
                                + (1 - cfg.ema_alpha) * skill.solve_rate_ema)
    skill.recent_use_count += 1
    skill.last_evaluated_round = round_idx


def _curriculum_score(skill, cfg: SelfPlayConfig) -> float:
    ema = skill.solve_rate_ema
    if ema is None:
        return cfg.unseen_priority
    return max(0.0, 1.0 - abs(ema - cfg.frontier_target) / max(cfg.frontier_target, 1e-6))


def curriculum_rank(pairs: List[SkilledPair], cfg: SelfPlayConfig) -> List[SkilledPair]:
    tier = {"strong": 0, "plausible": 1}

    def _demoted(sp) -> int:
        s = sp.skill
        return int(s.solve_rate_ema is not None
                   and s.recent_use_count >= cfg.trivial_min_uses
                   and s.solve_rate_ema >= cfg.trivial_ema)

    return sorted(pairs, key=lambda sp: (_demoted(sp),
                                         -_curriculum_score(sp.skill, cfg),
                                         tier.get(sp.level, 2)))


def generate_with_curriculum(seed, skill_bank, gen_policy, embedder,
                             cfg: SelfPlayConfig, *, parallel: bool = True) -> List[BugCandidate]:
    pairs = retrieve_skilled_pairs(
        seed, skill_bank, gen_policy, embedder,
        judge_budget=cfg.judge_budget,
        n_skills=cfg.n_skilled_target + cfg.retrieve_slack)
    pairs = curriculum_rank(pairs, cfg)

    n_skilled = min(len(pairs), cfg.n_skilled_target, cfg.n_total)
    n_explore = cfg.n_total - n_skilled
    jobs = [("skilled", pairs[i]) for i in range(n_skilled)] + [("explore", None)] * n_explore

    def _run(job):
        kind, pair = job
        return (generate_skilled(seed, gen_policy, pair) if kind == "skilled"
                else generate_exploration(seed, gen_policy))

    if not parallel or len(jobs) <= 1:
        return [_run(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        return list(ex.map(_run, jobs))


# ---------------------------------------------------------------- results ----
@dataclass
class BugRecord:
    is_exploration: bool
    skill_id: str
    valid: bool
    solve_rate: Optional[float]
    productive: bool
    injected: bool
    learned_repair: Optional[str]
    redistill_outcome: Optional[str]
    status: str


@dataclass
class RoundResult:
    task_id: str
    n_generated: int
    n_malformed: int
    n_valid: int
    bugs: List[BugRecord] = field(default_factory=list)
    skipped: Optional[str] = None

    @property
    def mean_solve_rate(self) -> Optional[float]:
        srs = [b.solve_rate for b in self.bugs if b.solve_rate is not None]
        return sum(srs) / len(srs) if srs else None


# -------------------------------------------------------- read / commit ----
@dataclass
class _ScoredItem:
    cand: BugCandidate
    vres: object
    epi: FixEpisode


@dataclass
class _SeedRead:
    """Everything the READ phase produced for one seed -- no bank mutations yet.
    `slots` preserves generation order; each is a malformed/invalid record (with
    its trace dict) or a scored item carrying the episode to commit."""
    task_id: str
    round_idx: int
    skipped: Optional[str] = None
    n_generated: int = 0
    n_malformed: int = 0
    n_valid: int = 0
    # ("bad", BugRecord, trace_dict) | ("scored", _ScoredItem, cand)
    slots: List[tuple] = field(default_factory=list)


def _seed_read(seed, *, skill_bank, repair_bank, advisor, gen_policy, fixer_policy,
               distill_policy, verifier, embedder, cfg, round_idx) -> _SeedRead:
    """READ phase for one seed: generate + run fixer episodes (write_back=False).
    Touches no dynamics-bearing bank state (only telemetry counters)."""
    task_id = seed.get("task_id", "")
    if not record_is_runnable(seed):
        return _SeedRead(task_id, round_idx, skipped="not_runnable")

    cands = generate_with_curriculum(seed, skill_bank, gen_policy, embedder, cfg)
    sr = _SeedRead(task_id, round_idx, n_generated=len(cands))

    for cand in cands:
        if not cand.success:
            sr.n_malformed += 1
            rec = BugRecord(cand.is_exploration, cand.skill_id or "", False,
                            None, False, False, None, None, cand.reason or "malformed")
            tr = {"round": round_idx, "task_id": task_id, "kind": "malformed",
                  "is_exploration": cand.is_exploration,
                  "guiding_skill_id": cand.skill_id or "", "reason": cand.reason or "malformed"}
            sr.slots.append(("bad", rec, tr))
            continue

        vres = verifier.verify(seed, cand.reconstructed)
        if not vres.is_valid_bug:
            rec = BugRecord(cand.is_exploration, cand.skill_id or "", False,
                            None, False, False, None, None, vres.outcome.value)
            tr = {"round": round_idx, "task_id": task_id, "kind": "invalid_bug",
                  "is_exploration": cand.is_exploration, "guiding_skill_id": cand.skill_id or "",
                  "verify_outcome": vres.outcome.value, "bug_mutation": cand.mutation}
            sr.slots.append(("bad", rec, tr))
            continue

        sr.n_valid += 1
        epi = run_fix_episode(
            seed, cand.body, vres.output,
            fixer_policy=fixer_policy, verifier=verifier, bank=repair_bank,
            advisor=advisor, distill_policy=distill_policy,
            k=cfg.fix_attempts, round_idx=round_idx,
            write_back=False,                      # defer ALL bank writes to commit
            stop_on_first_pass=cfg.stop_on_first_pass)
        sr.slots.append(("scored", _ScoredItem(cand, vres, epi), cand))
    return sr


def _real_bug_read(rec, *, repair_bank, advisor, fixer_policy, distill_policy,
                   verifier, cfg, round_idx) -> _SeedRead:
    """READ phase for a REAL published bug (reference-bug grounding): no generation --
    just run the escalation fixer episode (write-back deferred). Fixer-only; the
    generator SkillBank is untouched. Grows the RepairBank from the real bug
    distribution (the same 4 sources the held-out eval draws from)."""
    task_id = str(rec.get("task_id", ""))
    sr = _SeedRead(task_id, round_idx, n_generated=0)
    cp = rec.get("code_prompt", "") or ""
    buggy = rec.get("buggy", "") or ""
    if not buggy.strip() or not record_is_runnable(rec):
        sr.skipped = "real_unusable"
        return sr
    vb = verifier.verify(rec, cp + buggy)
    if vb.passed:                       # the published "bug" doesn't fail here -> skip
        sr.skipped = "real_not_failing"
        return sr
    sr.n_valid = 1
    epi = run_fix_episode(
        rec, buggy, vb.output,
        fixer_policy=fixer_policy, verifier=verifier, bank=repair_bank,
        advisor=advisor, distill_policy=distill_policy,
        k=cfg.fix_attempts, round_idx=round_idx,
        write_back=False, stop_on_first_pass=cfg.stop_on_first_pass)
    sr.slots.append(("real", epi, rec, vb.output))
    return sr


def _commit_seed(seed, read: _SeedRead, *, skill_bank, repair_bank, distill_policy,
                 cfg, trace=None) -> RoundResult:
    """COMMIT phase for one seed -- applies the three state mutations serially, in
    generation order. Must run single-threaded (one seed at a time)."""
    if read.skipped:
        return RoundResult(read.task_id, 0, 0, 0, skipped=read.skipped)

    cp = seed.get("code_prompt", "") or ""
    round_idx = read.round_idx
    bugs: List[BugRecord] = []

    for slot in read.slots:
        if slot[0] == "bad":
            _, rec, tr = slot
            bugs.append(rec)
            if trace:
                trace(tr)
            continue

        if slot[0] == "real":
            _, epi, rbug, bug_out = slot
            buggy_full = (rbug.get("code_prompt", "") or "") + (rbug.get("buggy", "") or "")
            sel = tuple(epi.advice.selected_skill_ids) if (epi.advice and epi.injected) else ()
            learned = repair_bank.record_fix_episode(
                buggy_full, bug_out, distill_policy,
                passed=epi.solved, skill_passed=(epi.mem_solved > 0),
                fixer_fix_diff=epi.fix_diff, injected_skill_ids=sel,
                task_id=str(rbug.get("task_id", "")), source="real", round_idx=round_idx)
            sr_val = epi.solve_rate
            bugs.append(BugRecord(False, "", True, sr_val, 0.0 < sr_val < 1.0,
                                  epi.injected, learned, None, "real"))
            if trace:
                adv = epi.advice
                trace({"round": round_idx, "task_id": read.task_id, "kind": "real",
                       "bug_error": (bug_out or "")[:1000],
                       "diagnosis": adv.diagnosis if adv else "",
                       "injected": epi.injected, "retrieved": adv.retrieved if adv else [],
                       "q_fault": list(getattr(adv, "q_fault", ())) if adv else [],
                       "solve_rate": sr_val, "mem_invoked": epi.mem_invoked,
                       "mem_solved": epi.mem_solved, "learned_repair": learned})
            continue

        _, item, cand = slot
        epi = item.epi
        sr_val = epi.solve_rate
        productive = 0.0 < sr_val < 1.0

        # (1) RepairBank write-back: utility on injected skills (rescue-scored) +
        #     learn a new skill from the fixer's own win.
        buggy_full = cp + cand.body
        sel = tuple(epi.advice.selected_skill_ids) if (epi.advice and epi.injected) else ()
        learned = repair_bank.record_fix_episode(
            buggy_full, item.vres.output, distill_policy,
            passed=epi.solved, skill_passed=(epi.mem_solved > 0),
            fixer_fix_diff=epi.fix_diff, injected_skill_ids=sel,
            task_id=str(seed.get("task_id", "")), source="fixer", round_idx=round_idx)

        # (2) curriculum: update the guiding skill's difficulty EMA (skilled only)
        if not cand.is_exploration and cand.skill_id:
            gs = skill_bank.get(cand.skill_id)
            if gs is not None:
                update_skill_ema(gs, sr_val, cfg, round_idx)

        # (3) generator self-evolution: re-distill productive bugs into the SkillBank
        redistill_outcome = None
        if cfg.redistill and productive:
            source = "generator_skilled" if not cand.is_exploration else "generator_exploration"
            try:
                dec = redistill_bug(
                    skill_bank, seed, cand.body, distill_policy,
                    source=source, test_failure=item.vres.output,
                    text_threshold=cfg.dedup_text_threshold,
                    guiding_skill_id=(cand.skill_id or None) if not cand.is_exploration else None)
                redistill_outcome = dec.outcome if dec is not None else None
            except Exception:
                redistill_outcome = None

        bugs.append(BugRecord(
            cand.is_exploration, cand.skill_id or "", True, sr_val, productive,
            epi.injected, learned, redistill_outcome, "scored"))

        if trace:
            adv = epi.advice
            trace({
                "round": round_idx, "task_id": read.task_id, "kind": "scored",
                "is_exploration": cand.is_exploration,
                "guiding_skill_id": cand.skill_id or "",
                "bug_mutation": cand.mutation,
                "bug_error": (item.vres.output or "")[:1000],
                "diagnosis": adv.diagnosis if adv else "",
                "abstained": adv.abstained if adv else True,
                "injected": epi.injected,
                "injected_instruction": (adv.instruction if adv and adv.instruction else ""),
                "retrieved": adv.retrieved if adv else [],
                "q_fault": list(getattr(adv, "q_fault", ())) if adv else [],
                "q_symptom": getattr(adv, "q_symptom", "") if adv else "",
                "solve_rate": sr_val, "productive": productive,
                "mem_invoked": epi.mem_invoked, "mem_solved": epi.mem_solved,
                "fix_attempts": epi.attempts,
                "fixer_fix_diff": epi.fix_diff,
                "learned_repair": learned,
                "redistill_outcome": redistill_outcome,
            })

    return RoundResult(read.task_id, read.n_generated, read.n_malformed, read.n_valid, bugs)


# ------------------------------------------------------------------ round ----
def run_round(seed, *, skill_bank, repair_bank, advisor: RepairAdvisor,
              gen_policy, fixer_policy, distill_policy, verifier, embedder,
              cfg: SelfPlayConfig, round_idx: int, trace=None) -> RoundResult:
    """One seed, fully sequential (read then commit). Kept for B=1 / external use;
    identical dynamics to the pre-parallel loop."""
    read = _seed_read(seed, skill_bank=skill_bank, repair_bank=repair_bank, advisor=advisor,
                      gen_policy=gen_policy, fixer_policy=fixer_policy,
                      distill_policy=distill_policy, verifier=verifier, embedder=embedder,
                      cfg=cfg, round_idx=round_idx)
    return _commit_seed(seed, read, skill_bank=skill_bank, repair_bank=repair_bank,
                        distill_policy=distill_policy, cfg=cfg, trace=trace)


# ------------------------------------------------------------------- loop ----
def run_selfplay(seeds, *, skill_bank, repair_bank, advisor: RepairAdvisor,
                 gen_policy, fixer_policy, distill_policy, verifier, embedder,
                 cfg: SelfPlayConfig, rounds: int, seed_rng: int = 0,
                 skill_bank_path: Optional[str] = None,
                 repair_bank_path: Optional[str] = None,
                 batch_size: int = 1,
                 verifier_factory: Optional[Callable[[], object]] = None,
                 real_bugs: Optional[list] = None, real_bug_frac: float = 0.0,
                 log=print) -> List[RoundResult]:
    """Run `rounds` self-play rounds over a pool of runnable seeds.

    batch_size B: B rounds run their READ phase concurrently against a frozen bank
    snapshot; their COMMITs apply serially in seed order. B=1 == fully sequential.
    verifier_factory: called once per worker thread to get a private verifier
    (FunctionVerifier spawns subprocesses -- don't share across threads). Falls
    back to the shared `verifier` when not given (fine for B=1)."""
    import json
    import os
    import random
    import time

    rng = random.Random(seed_rng)
    pool = [s for s in seeds if record_is_runnable(s)]
    if not pool:
        raise ValueError("no runnable seeds (need code_prompt with 'def <entry_point>' and tests)")

    # wrap the embedder so concurrent encode is safe; rebind the bank's embedder too
    # (the advisor shares this same RepairBank object).
    emb = _LockingEmbedder(embedder) if batch_size > 1 else embedder
    if batch_size > 1:
        repair_bank._emb = emb

    def _verifier():
        return verifier_factory() if verifier_factory else verifier

    def _open_append(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return open(path, "a", encoding="utf-8")

    trace_f = _open_append(cfg.trace_path) if cfg.trace_path else None
    trace = (lambda d: (trace_f.write(json.dumps(d, default=str) + "\n"), trace_f.flush())) \
        if trace_f else None
    metrics_f = _open_append(cfg.metrics_path) if cfg.metrics_path else None

    results: List[RoundResult] = []
    t0 = time.time()
    # each round is ("gen", clean seed) or ("real", published bug). With prob
    # real_bug_frac, run a fixer-only episode on a real bug drawn from the 4 sources
    # (reference-bug grounding) so the RepairBank sees the real bug distribution too.
    # REFERENCE MIXING (paper 4.5.1): a reference bug is tied to the SAME training
    # task the round plays -- with prob p_mix we REPLACE that task's generated bug with
    # its reference bug and run the fixer on it (fixer-only write-back; the generator is
    # not updated, since the reference bug wasn't sampled from the generator). The
    # reference set is the 4 sources' TRAIN split, disjoint from the eval TEST split.
    real_pool = [s for s in (real_bugs or [])
                 if record_is_runnable(s) and (s.get("buggy") or "").strip()]
    real_by_task = collections.defaultdict(list)
    for rb in real_pool:
        real_by_task[_norm_task_id(rb.get("task_id"))].append(rb)

    # generator rounds draw the clean pool WITHOUT replacement (shuffle + cycle) so the
    # full training set is covered; each round consumes one clean task, generated-on or
    # (with prob p_mix, if it has a reference bug) reference-mixed-on.
    gen_order = list(pool)
    rng.shuffle(gen_order)
    gi = 0
    chosen = []
    n_ref = 0
    for _ in range(rounds):
        if gi >= len(gen_order):
            rng.shuffle(gen_order)
            gi = 0
        x = gen_order[gi]; gi += 1
        refs = real_by_task.get(_norm_task_id(x.get("task_id")))
        if refs and rng.random() < real_bug_frac:
            chosen.append(("real", rng.choice(refs)))      # reference bug FOR THIS task
            n_ref += 1
        else:
            chosen.append(("gen", x))
    print(f"  seed pool: {len(pool)} clean tasks | reference bugs: {len(real_pool)} over "
          f"{len(real_by_task)} tasks | reference-mixing rounds: {n_ref}/{rounds} "
          f"(p_mix={real_bug_frac})", flush=True)

    def _log_round(ridx, r):
        msr = r.mean_solve_rate
        status = repair_bank.counts_by_status()
        prod = sum(1 for b in r.bugs if b.productive)
        log(f"[round {ridx+1}/{rounds}] {r.task_id} valid={r.n_valid} malformed={r.n_malformed} "
            f"productive={prod} mean_sr={'-' if msr is None else f'{msr:.2f}'} "
            f"| skillbank={len(skill_bank)} repairbank={len(repair_bank)} {dict(status)} "
            f"| {time.time()-t0:.0f}s")

    def _metrics_line(ridx, r):
        # generator Skill.family is per-skill (≈ id), not a coarse grouping -- group by
        # structural site-type instead (small, meaningful); detailed family-over-time
        # is recovered from the round-tagged checkpoints. Repair family IS coarse (the
        # distiller's "broad group"), so keep it as-is.
        site_s = collections.Counter(
            getattr(getattr(s, "structural", None), "site_type", "") or "?" for s in skill_bank)
        fam_r = collections.Counter(getattr(s, "family", "") or "?" for s in repair_bank.skills.values())
        return {
            "round": ridx,
            "skillbank_size": len(skill_bank),
            "repairbank_size": len(repair_bank),
            "repair_status": dict(repair_bank.counts_by_status()),
            "skill_site_counts": dict(site_s),
            "repair_family_counts": dict(fam_r),
            "n_valid": r.n_valid, "n_malformed": r.n_malformed,
            "frontier_count": sum(1 for b in r.bugs if b.productive),
            "mean_sr": r.mean_solve_rate,
            "n_skill_new": sum(1 for b in r.bugs if b.redistill_outcome in ("new", "specialization")),
            "n_skill_merge": sum(1 for b in r.bugs if b.redistill_outcome == "same"),
            "n_repair_new": sum(1 for b in r.bugs if b.learned_repair in ("new", "variant")),
            "n_repair_merge": sum(1 for b in r.bugs if b.learned_repair == "same"),
            "n_injected": sum(1 for b in r.bugs if b.injected),
        }

    def _checkpoint(ridx):
        if cfg.checkpoint_every and cfg.checkpoint_dir and (ridx + 1) % cfg.checkpoint_every == 0:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            skill_bank.save(os.path.join(cfg.checkpoint_dir, f"skillbank_r{ridx+1}.json"))
            repair_bank.save(os.path.join(cfg.checkpoint_dir, f"repairbank_r{ridx+1}.json"))

    def _after_commit(ridx, r):
        results.append(r)
        _log_round(ridx, r)
        if metrics_f:
            metrics_f.write(json.dumps(_metrics_line(ridx, r), default=str) + "\n")
            metrics_f.flush()
        if cfg.save_every and (ridx + 1) % cfg.save_every == 0:
            if skill_bank_path:
                skill_bank.save(skill_bank_path)
            if repair_bank_path:
                repair_bank.save(repair_bank_path)
        _checkpoint(ridx)

    def _read_round(kind, rec, ridx):
        if kind == "real":
            return _real_bug_read(rec, repair_bank=repair_bank, advisor=advisor,
                                  fixer_policy=fixer_policy, distill_policy=distill_policy,
                                  verifier=_verifier(), cfg=cfg, round_idx=ridx)
        return _seed_read(rec, skill_bank=skill_bank, repair_bank=repair_bank, advisor=advisor,
                          gen_policy=gen_policy, fixer_policy=fixer_policy,
                          distill_policy=distill_policy, verifier=_verifier(),
                          embedder=emb, cfg=cfg, round_idx=ridx)

    if batch_size <= 1:
        for i, (kind, rec) in enumerate(chosen):
            read = _read_round(kind, rec, i)
            r = _commit_seed(rec, read, skill_bank=skill_bank, repair_bank=repair_bank,
                             distill_policy=distill_policy, cfg=cfg, trace=trace)
            _after_commit(i, r)
    else:
        # warm every skill vector once (serial) so the parallel read phase only
        # embeds QUERIES, not the whole bank, under the lock.
        for s in list(repair_bank.skills.values()):
            repair_bank._vec(s)

        for b0 in range(0, len(chosen), batch_size):
            batch = chosen[b0:b0 + batch_size]
            idxs = list(range(b0, b0 + len(batch)))
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                reads = list(ex.map(lambda t: _read_round(t[0][0], t[0][1], t[1]),
                                    zip(batch, idxs)))
            # serial, deterministic commit in round order
            for (kind, rec), ridx, read in zip(batch, idxs, reads):
                r = _commit_seed(rec, read, skill_bank=skill_bank, repair_bank=repair_bank,
                                 distill_policy=distill_policy, cfg=cfg, trace=trace)
                _after_commit(ridx, r)

    if skill_bank_path:
        skill_bank.save(skill_bank_path)
    if repair_bank_path:
        repair_bank.save(repair_bank_path)
    if trace_f:
        trace_f.close()
    if metrics_f:
        metrics_f.close()
    return results

