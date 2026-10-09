"""Does the fault-fingerprint retrieval beat the old hybrid retrieval?

ONE script, run-and-sleep. Two phases:

  PHASE 1 -- migrate (read-only on the original bank):
    Re-fingerprint every skill with fault.fix_fingerprint (operation-token, not
    the 'Assign' container label), compute its symptom_class and fix_shape, and
    write a SEPARATE bank file. The original bank JSON is never modified.

  PHASE 2 -- evaluate, 5 paired arms on the SAME held-out bugs:
    naked              fixer sees only code + error (floor)
    dx_synth           v2 diagnosis -> synth instruction, NO memory (the bar)
    dx_mem_old         OLD retrieval (sem + saturated failure_class + saturated
                       whole-function struct) -> current render (skill + diff)
    dx_mem_new         NEW retrieval (operation-token primary + symptom class,
                       embedding as tie-break) -> render with diff + repair shape
    dx_mem_new_nodiff  NEW retrieval -> ABSTRACT skill text only (no diff/shape)

  Bugs are evaluated CONCURRENTLY (--workers); each bug is independent and the
  bank is only read (retrieval telemetry counters aside), so parallelism changes
  nothing in the measurement, only the wall time. The embedder is lock-wrapped
  (concurrent encode isn't thread-safe) and each worker gets its own verifier.

Usage (de-noised fixer; uses the gateway + the local embedder):
    nohup python scripts/probe_retrieval.py \
        --old-bank runs/selfplay_900/repairbank.json \
        --new-bank runs/repairbank_900_fp.json \
        --test-sources human human_edited_lm qwen7b gpt_oss_20b --test-per-source 40 \
        --fixer-model gpt-5-nano --synth-model qwen-coder --workers 6 \
        --exec-python $HOME/verify-env/bin/python \
        --out runs/retrieval_900.txt > runs/retrieval_900.log 2>&1 &
"""
import argparse, json, os, re, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Set, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from memetic import FunctionVerifier
from memetic.loader import load_bugbench_records
from memetic.loop_v2 import record_is_runnable
from memetic.embeddings import Embedder, cosine_similarity
from memetic.fixer_memory import RepairBank, normalize_failure
from memetic import fault as F

from probe_repair_eval import (
    build_policy, gen_nonempty, synth_nomem, _render_skills,
    cap_instruction, solve, bucket_of, _NULL,
)


# ---- v2 diagnoser ----------------------------------------------------------
_ASSERT_PAT = re.compile(
    r"(AssertionError|[A-Za-z]*Error|Exception|!=|Lists differ|Tuples differ|"
    r"assert\w*\(|expected|actual|Traceback)", re.I)


def _failure_signal(error, limit=1800):
    s = error or ""
    sig = [ln for ln in s.splitlines() if _ASSERT_PAT.search(ln)]
    if not sig:
        return s[:limit]
    return ("\n".join(sig[:20]) + "\n...\n" + s[-900:])[:limit]


def diagnose_mechanism_v2(buggy_code, error, policy):
    sig = _failure_signal(error)
    prompt = (
        "A Python function is failing its tests. Diagnose the ACTUAL cause, grounded "
        "in the specific failing assertion -- not a generic category.\n\n"
        "CODE:\n```python\n" + buggy_code + "\n```\n\n"
        "FAILING TEST OUTPUT (assertion + traceback):\n" + sig + "\n\n"
        "Work in this order, 2-3 short sentences total:\n"
        "1. State exactly what the test EXPECTED and what the code PRODUCED -- quote "
        "the values/types from the assertion.\n"
        "2. Point to the SINGLE construct in the code that produces that specific "
        "difference -- quote the exact expression in backticks.\n"
        "3. Name the mechanism in a few words.\n"
        "Do NOT guess. Do NOT say the bug is about the return value/shape UNLESS the "
        "assertion is literally about what is returned. Reply with only the sentences.")
    out = gen_nonempty(policy, prompt, timeout=90, tries=3)
    return " ".join(out.split())[:400] if out else ""


# ------------------------------------------------------ migrated skill index ----
@dataclass
class NewSkill:
    skill_id: str
    name: str
    root_cause: str
    recognition: str
    repair_strategy: str
    negative_conditions: str
    manifestations: List[str]
    site_type: str
    op: str
    keywords: Tuple[str, ...]
    symptom_class: str
    shape: dict
    task_ids: Set[str] = field(default_factory=set)
    status: str = "cold"
    vec: object = None

    def retrieval_text(self):
        return f"{self.name}. {self.root_cause} {self.recognition}".strip()


def migrate(old_bank: RepairBank, embedder: Embedder, new_path: str) -> List[NewSkill]:
    """Re-fingerprint every skill from its originating case (buggy_local + diff)."""
    index: List[NewSkill] = []
    dump = []
    for sid, s in old_bank.skills.items():
        case = next((old_bank.cases[c] for c in s.case_ids if c in old_bank.cases), None)
        buggy = (case.buggy_local if case else "") or ""
        diff = (s.manifestations[0] if s.manifestations
                else (case.fixed_diff if case else "")) or ""
        if buggy and diff:
            st, op, kw = F.fix_fingerprint(buggy, diff)
            shape = F.fix_shape(buggy, diff)
        else:
            st, op, kw = "", "", ()
            shape = {"op_multiset": [], "n_sites": 0, "shape": "unknown"}
        sym = F.symptom_class(case.failure if case else "")
        task_ids = {old_bank.cases[c].task_id for c in s.case_ids if c in old_bank.cases}
        ns = NewSkill(
            skill_id=sid, name=s.name, root_cause=s.root_cause, recognition=s.recognition,
            repair_strategy=s.repair_strategy, negative_conditions=s.negative_conditions,
            manifestations=list(s.manifestations), site_type=st, op=op, keywords=tuple(kw),
            symptom_class=sym, shape=shape, task_ids=task_ids, status=s.status)
        ns.vec = embedder.embed_one(ns.retrieval_text())
        index.append(ns)
        dump.append({"skill_id": sid, "name": s.name, "root_cause": s.root_cause,
                     "recognition": s.recognition, "repair_strategy": s.repair_strategy,
                     "negative_conditions": s.negative_conditions,
                     "manifestations": list(s.manifestations),
                     "fault": {"site_type": st, "op": op, "keywords": list(kw),
                               "symptom_class": sym, "shape": shape},
                     "task_ids": sorted(task_ids), "status": s.status})
    os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump({"skills": dump}, f, indent=2)
    return index


# -------------------------------------------------------------- new retrieval ----
W_OP, W_SITE, W_SYM, W_KW = 0.60, 0.15, 0.25, 0.10


def retrieve_new(index, qv, q_fault, q_sym, k, drop_tasks):
    q_st, q_op, q_kw = q_fault
    rows = []
    for ns in index:
        if ns.status == "quarantined" or (ns.task_ids & drop_tasks):
            continue
        sem = float(cosine_similarity(qv, ns.vec))
        op_m = F.op_match(q_op, ns.op)
        site_m = 1.0 if (q_st and ns.site_type and q_st == ns.site_type) else 0.0
        sym_m = 1.0 if (q_sym and ns.symptom_class and q_sym == ns.symptom_class) else 0.0
        kw_m = 1.0 if (set(q_kw) & set(ns.keywords)) else 0.0
        score = sem + W_OP * op_m + W_SITE * site_m + W_SYM * sym_m + W_KW * kw_m
        rows.append((score, {"sem": round(sem, 3), "op": round(op_m, 2), "site": site_m,
                             "sym": sym_m, "kw": kw_m}, ns))
    rows.sort(key=lambda x: -x[0])
    return rows[:k]


def render_new(rows, with_diff: bool) -> str:
    out = []
    for i, (_sc, _ch, ns) in enumerate(rows):
        neg = f"\n    do_NOT: {ns.negative_conditions}" if ns.negative_conditions else ""
        extra = ""
        if with_diff:
            sh = ns.shape
            extra += f"\n    repair shape: {sh.get('shape','?')} across ~{sh.get('n_sites',0)} site(s)"
            if sh.get("op_multiset"):
                extra += f"; operations touched: {', '.join(sh['op_multiset'])}"
            if ns.manifestations:
                extra += f"\n    example fix of this kind:\n{(ns.manifestations[0] or '')[:280]}"
        out.append(f"[{i+1}] {ns.name}\n    root_cause: {ns.root_cause}\n    "
                   f"recognition: {ns.recognition}\n    "
                   f"repair_strategy (the operator to apply): {ns.repair_strategy}{neg}{extra}")
    return "\n".join(out)


def synth_mem_block(buggy_code, error, diagnosis, block, policy):
    prompt = (
        "You are writing a precise fix instruction for a code fixer, and you may draw on "
        "past repair skills retrieved from a memory of verified fixes.\n\n"
        "CURRENT BUGGY CODE:\n```python\n" + buggy_code + "\n```\n\n"
        "FAILING TEST OUTPUT:\n" + (error or "")[:500] + "\n\n"
        f"Diagnosed mechanism: {diagnosis}\n\n"
        "RETRIEVED REPAIR SKILLS (reference for the repair pattern, NOT to copy verbatim):\n"
        + block + "\n\n"
        "First decide: does at least ONE retrieved skill's repair_strategy genuinely match "
        "THIS bug's mechanism and apply to THIS code? "
        f"If NONE genuinely applies, reply with exactly {_NULL} and nothing else.\n"
        "Otherwise, using the matching repair_strategy operator(s), write a SPECIFIC, "
        "ACTIONABLE instruction for THIS code: name the exact construct that is wrong and "
        "the exact change to make (e.g. 'In the mean() call, change axis=0 to axis=1.'). "
        "1-3 sentences. Do NOT output the full corrected function.")
    out = gen_nonempty(policy, prompt, timeout=90, tries=3).strip()
    if out.upper().rstrip(".:") == _NULL:
        return _NULL
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if lines and lines[-1].upper().rstrip(".:") == _NULL:
        return _NULL
    return out[:500]


# ---- thread-safe embedder proxy (concurrent encode isn't safe) ----
class _LockEmb:
    def __init__(self, e):
        self._e = e
        self._lock = threading.Lock()

    def __getattr__(self, n):
        a = getattr(object.__getattribute__(self, "_e"), n)
        if callable(a):
            lock = object.__getattribute__(self, "_lock")
            def w(*x, **k):
                with lock:
                    return a(*x, **k)
            return w
        return a


ALL_CONDS = ("naked", "dx_synth", "dx_mem_old", "dx_mem_new", "dx_mem_new_nodiff")
ALL_MEM = ("dx_mem_old", "dx_mem_new", "dx_mem_new_nodiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-bank", default="runs/selfplay_900/repairbank.json",
                    help="read-only; never modified")
    ap.add_argument("--new-bank", default="runs/repairbank_900_fp.json",
                    help="output: the re-fingerprinted bank (separate file)")
    ap.add_argument("--test-sources", nargs="*",
                    default=["human", "human_edited_lm", "qwen7b", "gpt_oss_20b"])
    ap.add_argument("--test-per-source", type=int, default=40)
    ap.add_argument("--retrieve-k", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fixer-model", default="gpt-5-nano")
    ap.add_argument("--synth-model", default="qwen-coder")
    ap.add_argument("--exec-python", default=None)
    ap.add_argument("--fixer-max-tokens", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8,
                    help="bugs evaluated concurrently (each is independent; bank is read-only)")
    ap.add_argument("--arms", default="naked,dx_synth,dx_mem_new",
                    help="comma list of arms to run (naked+dx_synth always included). Default is "
                         "the 3 that answer the headline; add dx_mem_old / dx_mem_new_nodiff for "
                         "the full ablation. Fewer arms = fewer fixer calls = faster.")
    ap.add_argument("--out", default="runs/retrieval_cov.txt")
    args = ap.parse_args()

    # selected arms (naked + dx_synth are always kept as the baselines); the old-retrieval
    # bank.retrieve + de-saturation stats are still computed for the quality block, but the
    # old ARM's synth+solve only run if requested.
    sel = {x.strip() for x in args.arms.split(",") if x.strip()} | {"naked", "dx_synth"}
    CONDS = tuple(c for c in ALL_CONDS if c in sel)
    MEM_ARMS = tuple(a for a in ALL_MEM if a in sel)
    print(f"arms: {', '.join(CONDS)}", flush=True)

    fixer_pol = build_policy(args.fixer_model, temperature=0.2, max_tokens=args.fixer_max_tokens)
    synth_pol = build_policy(args.synth_model, temperature=0.4, max_tokens=2000)
    embedder = Embedder()
    lemb = _LockEmb(embedder)
    setup_verifier = FunctionVerifier(timeout_seconds=30, python_executable=args.exec_python)

    # ---- PHASE 1: load (read-only), migrate ----
    bank = RepairBank.load(args.old_bank, embedder)
    bank._emb = lemb                       # query-embedding during parallel retrieve -> locked
    skill_tasks = {sid: {bank.cases[c].task_id for c in s.case_ids if c in bank.cases}
                   for sid, s in bank.skills.items()}
    index = migrate(bank, embedder, args.new_bank)
    with_fp = sum(1 for ns in index if ns.op)
    print(f"loaded old bank: {len(bank)} skills <- {args.old_bank}", flush=True)
    print(f"migrated -> {args.new_bank}: {len(index)} skills, "
          f"{with_fp} with an operation token ({100*with_fp/max(1,len(index)):.0f}%)", flush=True)
    # warm all skill vectors (serial) so the parallel phase only embeds queries under the lock
    for s in list(bank.skills.values()):
        bank._vec(s)

    # ---- held-out test bugs ----
    # --test-per-source 0 (or negative) == use the ENTIRE test split of each source
    # (all available test data). Otherwise keep the first N runnable+failing per source.
    per = args.test_per_source
    all_test = per <= 0
    tests = []
    for src in args.test_sources:
        kept = 0
        lim = None if all_test else per * 3
        for rec in load_bugbench_records(src, "test", limit=lim):
            if not all_test and kept >= per:
                break
            if not record_is_runnable(rec) or not (rec.get("buggy") or "").strip():
                continue
            cp = rec.get("code_prompt", "") or ""
            if not setup_verifier.verify(rec, cp + rec["canonical_solution"]).passed:
                continue
            vb = setup_verifier.verify(rec, cp + rec["buggy"])
            if vb.passed:
                continue
            tests.append((rec, rec["buggy"], vb.output, bucket_of(vb.output)))
            kept += 1
        print(f"  test {src}: {kept}{' (ALL)' if all_test else ''}", flush=True)
    print(f"total {len(tests)} test bugs; k={args.k}, retrieve-k={args.retrieve_k}, "
          f"workers={args.workers}\n", flush=True)

    # ---- per-bug work (pure: reads bank, calls gateway, own verifier) ----
    def process_bug(item):
        rec, bug_body, bug_out, bucket = item
        verifier = FunctionVerifier(timeout_seconds=30, python_executable=args.exec_python)
        cp = rec.get("code_prompt", "") or ""
        buggy_full = cp + bug_body
        task_id = str(rec.get("task_id"))
        fclass, _ = normalize_failure(bug_out)

        dx = diagnose_mechanism_v2(buggy_full, bug_out, synth_pol)
        spans = F.spans_from_diagnosis(dx)
        q_fault = F.fault_fingerprint(buggy_full, spans)
        q_sym = F.symptom_class(bug_out)

        old_rows = []
        if dx:
            got = bank.retrieve(dx, k=args.retrieve_k + 4, q_failure_class=fclass,
                                q_buggy_full=buggy_full, with_scores=True)
            for (sc, ch, s) in got:
                if task_id in skill_tasks.get(s.skill_id, ()):
                    continue
                old_rows.append((sc, ch, s))
                if len(old_rows) >= args.retrieve_k:
                    break
        new_rows = []
        if dx:
            qv = lemb.embed_one(dx)
            new_rows = retrieve_new(index, qv, q_fault, q_sym, args.retrieve_k, {task_id})

        rql = {
            "n": 1, "q_has_op": int(bool(q_fault[1])),
            "old_top1_struct1": int(bool(old_rows) and old_rows[0][1]["struct"] == 1),
            "old_n": int(bool(old_rows)), "old_sem_sum": old_rows[0][1]["sem"] if old_rows else 0.0,
            "new_n": int(bool(new_rows)), "new_sem_sum": new_rows[0][1]["sem"] if new_rows else 0.0,
            "new_top1_opmatch": int(bool(new_rows) and bool(q_fault[1]) and new_rows[0][2].op == q_fault[1]),
            "top1_changed": int(bool(old_rows) and bool(new_rows)
                                and old_rows[0][2].skill_id != new_rows[0][2].skill_id),
        }

        instr_nomem = synth_nomem(buggy_full, bug_out, dx, synth_pol)

        def mem_instr(rows, block):
            if not rows:
                return instr_nomem, True
            raw = synth_mem_block(buggy_full, bug_out, dx, block, synth_pol)
            if raw == _NULL:
                return instr_nomem, True
            return raw, False

        # only pay for the synth call of a memory arm that is actually selected
        if "dx_mem_old" in sel:
            instr_old, ab_old = mem_instr(old_rows, _render_skills(old_rows) if old_rows else "")
        else:
            instr_old, ab_old = "", True
        if "dx_mem_new" in sel:
            instr_new, ab_new = mem_instr(new_rows, render_new(new_rows, True) if new_rows else "")
        else:
            instr_new, ab_new = "", True
        if "dx_mem_new_nodiff" in sel:
            instr_new_so, ab_new_so = mem_instr(new_rows, render_new(new_rows, False) if new_rows else "")
        else:
            instr_new_so, ab_new_so = "", True

        caps = {"naked": "", "dx_synth": cap_instruction(instr_nomem),
                "dx_mem_old": cap_instruction(instr_old),
                "dx_mem_new": cap_instruction(instr_new),
                "dx_mem_new_nodiff": cap_instruction(instr_new_so)}

        row = {"task_id": task_id, "bucket": bucket, "diagnosis": dx,
               "q_fault": list(q_fault), "q_sym": q_sym,
               "abstain": {"old": ab_old, "new": ab_new, "new_nodiff": ab_new_so},
               "old_retrieved": [{"name": s.name, "score": round(sc, 3), "sem": ch["sem"],
                                  "fclass": ch["fclass"], "struct": ch["struct"],
                                  "site": f"{s.site_type}/{s.operator}" if s.site_type else ""}
                                 for (sc, ch, s) in old_rows],
               "new_retrieved": [{"name": ns.name, "score": round(sc, 3), "sem": ch["sem"],
                                  "op": ch["op"], "sym": ch["sym"], "site_m": ch["site"],
                                  "fault": f"{ns.site_type}/{ns.op}", "shape": ns.shape.get("shape"),
                                  "skill_sym": ns.symptom_class}
                                 for (sc, ch, ns) in new_rows],
               "instr_nomem": instr_nomem, "instr_old": instr_old,
               "instr_new": instr_new, "instr_new_nodiff": instr_new_so}

        perarm = {}
        for c in CONDS:
            p, k = solve(rec, bug_body, bug_out, caps[c], fixer_pol, verifier, args.k)
            perarm[c] = (p, k)
            row[c] = {"passes": p, "solved": p > 0}

        return {"row": row, "bucket": bucket, "perarm": perarm, "q_op": q_fault[1],
                "abstain": {"dx_mem_old": ab_old, "dx_mem_new": ab_new,
                            "dx_mem_new_nodiff": ab_new_so}, "rq": rql}

    # ---- accumulators ----
    solved = defaultdict(lambda: defaultdict(int))
    sr = defaultdict(lambda: defaultdict(float))
    n_bucket = defaultdict(int)
    paired = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    esc = defaultdict(lambda: defaultdict(int))   # bucket -> arm -> escalation-solved (naked OR arm)
    n_abstain = defaultdict(lambda: defaultdict(int))
    fired_solved = defaultdict(lambda: defaultdict(int))
    fired_n = defaultdict(int)
    rq = defaultdict(float)
    trace_out = args.out + ".jsonl"

    # ---- run bugs concurrently; reduce in order ----
    done = 0
    with open(trace_out, "w", encoding="utf-8") as tf, \
         ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for res in ex.map(process_bug, tests):
            row = res["row"]; bucket = res["bucket"]
            n_bucket[bucket] += 1
            for kk, vv in res["rq"].items():
                rq[kk] += vv
            for c in CONDS:
                p, k = res["perarm"][c]
                solved[bucket][c] += int(p > 0)
                sr[bucket][c] += (p / k if k else 0.0)
            nk = row["naked"]["solved"]
            for c in ("dx_synth",) + MEM_ARMS:
                cd = row[c]["solved"]
                if cd and not nk:
                    paired[bucket][c][0] += 1
                elif nk and not cd:
                    paired[bucket][c][1] += 1
                # escalation: naked first, escalate only on a naked miss -> solved iff either wins
                esc[bucket][c] += int(bool(nk) or bool(cd))
            for a in MEM_ARMS:
                ab = res["abstain"][a]
                n_abstain[bucket][a] += int(ab)
                if not ab:
                    fired_n[a] += 1
                    fired_solved[a]["dx_synth"] += int(row["dx_synth"]["solved"])
                    fired_solved[a][a] += int(row[a]["solved"])
            tf.write(json.dumps(row) + "\n"); tf.flush()
            done += 1
            print(f"  [{done}/{len(tests)}] {row['task_id']} [{bucket}] q_op={res['q_op'] or '-'} "
                  + " ".join(f"{c.replace('dx_mem_','m_').replace('dx_','')[:9]}="
                             f"{'P' if row[c]['solved'] else '.'}" for c in CONDS), flush=True)

    # ------------------------------------------------------------- report ----
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    buckets = [b for b in ("assertion", "crash", "other") if n_bucket[b]]
    N = sum(n_bucket.values())
    out("\n=== Fault-fingerprint retrieval vs old hybrid retrieval ===")
    out(f"  old_bank={len(bank)} | new_bank={len(index)} | synth={args.synth_model} (FIXED) "
        f"| fixer={args.fixer_model}@0.2 | k={args.k} | retrieve-k={args.retrieve_k}\n")
    out("  solved@k (>=1 of k passes):")
    out(f"  {'stratum':10}{'n':>4}" + "".join(f"{c:>19}" for c in CONDS))
    for b in buckets:
        n = n_bucket[b]
        out(f"  {b:10}{n:>4}" + "".join(f"{100*solved[b][c]/n:>18.0f}%" for c in CONDS))
    if N:
        out(f"  {'POOLED':10}{N:>4}" + "".join(
            f"{100*sum(solved[b][c] for b in buckets)/N:>18.0f}%" for c in CONDS))

    out("\n  mean solve_rate (passes/k, pooled):")
    for c in CONDS:
        tot = sum(sr[b][c] for b in buckets)
        out(f"    {c:20}{(tot/N if N else 0):.3f}")

    out("\n  paired vs naked (rescue = solved only with it; harm = broke only with it):")
    for c in ("dx_synth",) + MEM_ARMS:
        r = sum(paired[b][c][0] for b in buckets)
        h = sum(paired[b][c][1] for b in buckets)
        out(f"    {c:20} rescue={r:>2} harm={h:>2} net={r-h:+d}")

    out("\n  ESCALATION solved@k (run naked first; escalate only on a naked miss -> cannot harm):")
    nkp = sum(solved[b]["naked"] for b in buckets)
    out(f"    {'naked (baseline)':24}{100*nkp/max(1,N):>5.0f}%")
    for c in ("dx_synth",) + MEM_ARMS:
        e = sum(esc[b][c] for b in buckets)
        out(f"    escalate->{c:14}{100*e/max(1,N):>5.0f}%")

    out("\n  NULL-gate abstention (per arm, pooled):")
    for a in MEM_ARMS:
        ab = sum(n_abstain[b][a] for b in buckets)
        out(f"    {a:20} abstained {ab:>3}/{N} ({100*ab/max(1,N):>3.0f}%)")

    out("\n  FIRED-ONLY (each arm vs dx_synth, only on bugs where that arm actually fired):")
    for a in MEM_ARMS:
        nf = fired_n[a]
        if not nf:
            out(f"    {a:20} fired on 0"); continue
        syn = 100 * fired_solved[a]["dx_synth"] / nf
        mem = 100 * fired_solved[a][a] / nf
        out(f"    {a:20} fired={nf:>3}  dx_synth={syn:>3.0f}%  {a.replace('dx_mem_','')}={mem:>3.0f}%  "
            f"delta={mem-syn:+.0f}pp")

    out("\n  RETRIEVAL QUALITY (de-saturation check):")
    nq = rq["n"] or 1
    out(f"    query had an operation token : {int(rq['q_has_op'])}/{int(nq)} ({100*rq['q_has_op']/nq:.0f}%)")
    if rq["old_n"]:
        out(f"    OLD top-1: struct==1 on {int(rq['old_top1_struct1'])}/{int(rq['old_n'])} "
            f"({100*rq['old_top1_struct1']/rq['old_n']:.0f}%, saturated)  mean sem={rq['old_sem_sum']/rq['old_n']:.3f}")
    if rq["new_n"]:
        om = rq["new_top1_opmatch"]
        out(f"    NEW top-1: op token matches query on {int(om)}/{int(rq['new_n'])} "
            f"({100*om/rq['new_n']:.0f}%)  mean sem={rq['new_sem_sum']/rq['new_n']:.3f}")
    out(f"    new top-1 differs from old top-1 on {int(rq['top1_changed'])}/{int(nq)} bugs "
        f"({100*rq['top1_changed']/nq:.0f}%)")

    out("\n  KEY: dx_mem_new>dx_mem_old = new retrieval wins. dx_mem_new>dx_synth = memory "
        "beats the diagnosis. dx_mem_new>dx_mem_new_nodiff = the concrete diff+shape adds.")
    out(f"\n  trace -> {trace_out}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
