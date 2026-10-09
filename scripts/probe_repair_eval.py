"""Phase 2 -- the READ PATH + trustworthy eval for the repair-memory system.

Unlike the throwaway probes, this loads the REAL distilled bank (runs/repair_bank.json,
built by scripts/build_repair_bank.py) and exercises the actual retrieval machinery:

  diagnose mechanism (query-side: buggy code + error, NO canonical)
    -> bank.retrieve(...)  HYBRID: semantic(recognition+root_cause) + failure_class + fix-site struct
    -> NULL gate: synthesizer may decline if no retrieved skill genuinely applies
    -> synthesize a SPECIFIC actionable instruction from the skills' repair_strategy OPERATORS
    -> frozen nano fixer applies it (temp 0.2)

Four paired conditions on the SAME held-out test bugs:
  naked        : fixer sees only code + error (baseline)
  dx_synth     : diagnosis + synthesized instruction, NO memory
  dx_mem       : diagnosis + synthesized instruction USING retrieved skills, WITH null gate
                 (on abstain, falls back to the dx_synth instruction -> cannot regress below it)
  oracle       : the true fix diff (ceiling)

CRITICAL for a fair test:
  - The synthesizer model is HELD FIXED (same policy) across dx_synth and dx_mem, so any
    difference is the MEMORY, not a stronger writer.
  - Only the FIXER is the weak frozen model (gpt-5-nano) at temp 0.2 (applying an instruction
    should be near-deterministic; temp 0.7 injected ~+/-15pp of pure noise).
  - Retrieved skills from the test bug's OWN task are filtered out (no leakage).
  - Stratified crash vs assertion (assertion = mechanism NOT already in the traceback =
    memory's fair shot); paired rescue/harm vs naked; mean solve_rate (passes/k) alongside
    solved@k (lower variance); abstention rate + fired-only sub-analysis.

Read:
  dx_mem   > dx_synth  => MEMORY helps (where it fires).
  dx_synth > naked     => diagnose-then-synthesize helps at all.
  abstention + no regression on POOLED => the NULL gate does its job.

Usage:
    python scripts/probe_repair_eval.py \
        --bank runs/repair_bank.json \
        --test-sources human gpt_oss_20b --test-per-source 30 \
        --retrieve-k 3 --k 4 \
        --fixer-model gpt-5-nano --synth-model qwen-coder \
        --exec-python $HOME/verify-env/bin/python \
        --out runs/repair_eval.txt
"""
import argparse, json, os, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from memetic import FunctionVerifier
from memetic.loader import load_bugbench_records
from memetic.loop_v2 import record_is_runnable
from memetic.seed import short_diff
from memetic.embeddings import Embedder
from memetic.fixer import build_fix_prompt, _FIX_SENTINEL
from memetic.codeutils import parse_and_extract_function, error_kind, CRASH_KINDS
from memetic.fixer_memory import RepairBank, normalize_failure


# ---- hard timeout around every model call (a hung gateway request is abandoned
# and treated as empty; the parse/gate downstream handles it) ----
_EXEC = ThreadPoolExecutor(max_workers=8)

def _timed(fn, *args, timeout=60, default=""):
    try:
        return _EXEC.submit(fn, *args).result(timeout=timeout)
    except Exception:
        return default

_GW_PHRASE = os.environ.get("GATEWAY_BOILERPLATE", "llm gateway api layer").strip().lower()
_GW_PREFIX = re.compile(r'(?im)^\s*' + re.escape(_GW_PHRASE) + r'\s*[:.\-]*\s*')

def _clean(s):
    # strip gateway boilerplate as a leading prefix AND as a whole (possibly
    # quoted / docstring / commented) line inside code blocks.
    s = _GW_PREFIX.sub('', s or '')
    out = []
    for line in s.splitlines():
        core = line.strip().strip('#').strip().strip('"\'').strip()
        if core.lower() == _GW_PHRASE:
            continue
        out.append(line)
    return "\n".join(out).strip()

def gen(policy, prompt, timeout=90):
    return _clean(_timed(policy.generate, prompt, timeout=timeout) or "")

def gen_nonempty(policy, prompt, timeout=90, tries=3):
    """Reasoning models intermittently return empty; retry until non-empty."""
    for _ in range(tries):
        out = gen(policy, prompt, timeout=timeout).strip()
        if out:
            return out
    return ""


def build_policy(model, temperature=0.7, max_tokens=1200):
    try:
        from memetic.prometheus_policy import PrometheusPolicy
    except Exception:
        from memetic.policy_prometheus import PrometheusPolicy
    api_base = os.environ.get("GATEWAY_API_BASE", "https://your-gateway.example.com")
    ca = os.environ.get("GATEWAY_CA_CERT", "secrets/gateway-ca.pem")
    return PrometheusPolicy(model=model, api_base=api_base, ca_cert=ca,
                           temperature=temperature, max_tokens=max_tokens)


def bucket_of(err):
    k = error_kind(err)
    return "crash" if k in CRASH_KINDS else ("assertion" if k == "assertion" else "other")


# --------------------------------------------------------------- read path ----
def diagnose_mechanism(buggy_code, error, policy):
    """QUERY-SIDE mechanism hypothesis: no canonical, just code + failure.
    Same style as the distiller's recognition so query and skill keys share a space."""
    prompt = (
        "A Python function is failing its tests.\n\n"
        "CODE:\n```python\n" + buggy_code + "\n```\n\n"
        "FAILING TEST OUTPUT:\n" + (error or "")[:500] + "\n\n"
        "In ONE short sentence, name the LIKELY KIND of mistake (the mechanism) -- "
        "e.g. 'aggregation over the wrong axis', 'off-by-one loop bound', 'missing "
        "DataFrame construction from the input', 'inverted boolean guard'. This is a "
        "hypothesis from the symptom; do not fix it. Reply with only the sentence.")
    out = gen_nonempty(policy, prompt, timeout=90, tries=3)
    return out.splitlines()[0][:160] if out else ""


def synth_nomem(buggy_code, error, diagnosis, policy):
    """dx_synth arm: instantiate the diagnosis into a specific fix for THIS code,
    with NO retrieved memory. The fixed-writer baseline the memory arm must beat."""
    prompt = (
        "You are writing a precise fix instruction for a code fixer.\n\n"
        "CURRENT BUGGY CODE:\n```python\n" + buggy_code + "\n```\n\n"
        "FAILING TEST OUTPUT:\n" + (error or "")[:500] + "\n\n"
        f"Diagnosed mechanism: {diagnosis}\n\n"
        "Write a SPECIFIC, ACTIONABLE instruction for THIS code: name the exact "
        "construct that is wrong and the exact change to make (e.g. 'In the mean() "
        "call, change axis=0 to axis=1.'). 1-3 sentences. Do NOT output the full "
        "corrected function.")
    return gen_nonempty(policy, prompt, timeout=90, tries=3).strip()[:500]


_SKILL_BLOCK = """[{i}] {name}
    root_cause: {rc}
    recognition: {rec}
    repair_strategy (the operator to apply): {rs}{neg}{ex}"""

def _render_skills(rows):
    """rows = list of (score, ch, skill). Render the repair_strategy OPERATORS +
    one example manifestation each, for the memory-arm synthesizer."""
    out = []
    for i, (_score, _ch, s) in enumerate(rows):
        neg = f"\n    do_NOT: {s.negative_conditions}" if s.negative_conditions else ""
        ex = ""
        if s.manifestations:
            d = (s.manifestations[0] or "")[:280]
            ex = f"\n    example fix of this kind:\n{d}"
        out.append(_SKILL_BLOCK.format(i=i + 1, name=s.name, rc=s.root_cause,
                                       rec=s.recognition, rs=s.repair_strategy,
                                       neg=neg, ex=ex))
    return "\n".join(out)


_NULL = "NULL"

def synth_mem(buggy_code, error, diagnosis, rows, policy):
    """dx_mem arm: synthesize a fix instruction for THIS code, DRAWING ON the retrieved
    repair skills' operators. NULL GATE: if none of the retrieved strategies genuinely
    matches this bug, reply exactly NULL (caller falls back to the no-memory instruction,
    so memory can never regress below dx_synth). Held to the SAME model as synth_nomem."""
    prompt = (
        "You are writing a precise fix instruction for a code fixer, and you may draw on "
        "past repair skills retrieved from a memory of verified fixes.\n\n"
        "CURRENT BUGGY CODE:\n```python\n" + buggy_code + "\n```\n\n"
        "FAILING TEST OUTPUT:\n" + (error or "")[:500] + "\n\n"
        f"Diagnosed mechanism: {diagnosis}\n\n"
        "RETRIEVED REPAIR SKILLS (reference for the repair pattern, NOT to copy verbatim):\n"
        + _render_skills(rows) + "\n\n"
        "First decide: does at least ONE retrieved skill's repair_strategy genuinely match "
        "THIS bug's mechanism and apply to THIS code? "
        f"If NONE genuinely applies, reply with exactly {_NULL} and nothing else.\n"
        "Otherwise, using the matching repair_strategy operator(s), write a SPECIFIC, "
        "ACTIONABLE instruction for THIS code: name the exact construct that is wrong and "
        "the exact change to make (e.g. 'In the mean() call, change axis=0 to axis=1.'). "
        "1-3 sentences. Do NOT output the full corrected function.")
    out = gen_nonempty(policy, prompt, timeout=90, tries=3).strip()
    # robust NULL detection: the model often hedges then abstains on the last line
    # ('...none apply ... NULL'), not a bare 'NULL' -- treat that as abstain too,
    # else the refusal text gets injected as if it were a fix instruction.
    if out.upper().rstrip(".:") == _NULL:
        return _NULL
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if lines and lines[-1].upper().rstrip(".:") == _NULL:
        return _NULL
    return out[:500]


def cap_instruction(instr):
    return (f"A diagnosis of this bug and how to fix it:\n{instr}\n\n"
            "Apply this correction to the code.") if instr else ""

def cap_oracle(fix_diff):
    return (f"The correct fix (unified diff, '-'=buggy '+'=fixed):\n{fix_diff}\n\n"
            "Apply this fix to the code.")


def solve(record, bug_body, bug_out, cap, policy, verifier, k):
    ep = record.get("entry_point") or "task_func"
    cp = record.get("code_prompt", "") or ""
    feedback = bug_out
    passes = 0
    for _ in range(k):
        prompt = build_fix_prompt(record, bug_body, feedback)
        if cap:
            prompt = prompt.replace(_FIX_SENTINEL, cap + "\n" + _FIX_SENTINEL, 1)
        raw = gen_nonempty(policy, prompt, timeout=90, tries=3)
        body, _ = parse_and_extract_function(raw, ep)
        if body is None:
            feedback = "fix did not parse into a function body"
            continue
        res = verifier.verify(record, cp + body)
        passes += int(res.passed)
        feedback = res.output
    return passes, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="runs/repair_bank.json")
    ap.add_argument("--test-sources", nargs="*", default=["human", "gpt_oss_20b"])
    ap.add_argument("--test-per-source", type=int, default=30)
    ap.add_argument("--retrieve-k", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fixer-model", default="gpt-5-nano")
    ap.add_argument("--synth-model", default="qwen-coder")
    ap.add_argument("--exec-python", default=None)
    ap.add_argument("--fixer-max-tokens", type=int, default=4000,
                    help="fixer completion budget. 1200 starved gpt-5-nano (a reasoning "
                         "model): reasoning ate the budget and the visible fix came back "
                         "empty ~half the time. 4000 gives the fix room after reasoning.")
    ap.add_argument("--out", default="runs/repair_eval.txt")
    args = ap.parse_args()

    # fixer: weak frozen model, LOW temp (applying an instruction ~deterministic).
    # synth: held FIXED across dx_synth and dx_mem so lift = memory, not a better writer.
    fixer_pol = build_policy(args.fixer_model, temperature=0.2, max_tokens=args.fixer_max_tokens)
    synth_pol = build_policy(args.synth_model, temperature=0.4, max_tokens=2000)
    verifier = FunctionVerifier(timeout_seconds=30, python_executable=args.exec_python)
    embedder = Embedder()

    bank = RepairBank.load(args.bank, embedder)
    # per-skill originating task_ids, to drop same-task retrievals (leakage guard)
    skill_tasks = {}
    for sid, s in bank.skills.items():
        skill_tasks[sid] = {bank.cases[cid].task_id for cid in s.case_ids
                            if cid in bank.cases}
    print(f"loaded bank: {len(bank)} skills, {len(bank.cases)} cases <- {args.bank}", flush=True)

    # ---- held-out test bugs (real bugs: canonical passes, buggy fails) ----
    tests = []
    for src in args.test_sources:
        recs = load_bugbench_records(src, "test", limit=args.test_per_source * 3)
        kept = 0
        for rec in recs:
            if kept >= args.test_per_source:
                break
            if not record_is_runnable(rec) or not (rec.get("buggy") or "").strip():
                continue
            cp = rec.get("code_prompt", "") or ""
            if not verifier.verify(rec, cp + rec["canonical_solution"]).passed:
                continue
            vb = verifier.verify(rec, cp + rec["buggy"])
            if vb.passed:
                continue
            tests.append((rec, rec["buggy"], vb.output, bucket_of(vb.output)))
            kept += 1
        print(f"  test {src}: {kept}", flush=True)
    print(f"total {len(tests)} test bugs; k={args.k}, retrieve-k={args.retrieve_k}\n", flush=True)

    conds = ("naked", "dx_synth", "dx_mem", "oracle")
    solved = defaultdict(lambda: defaultdict(int))
    sr = defaultdict(lambda: defaultdict(float))
    n_bucket = defaultdict(int)
    paired = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # bucket->cond->[rescue,harm]
    n_empty_dx = defaultdict(int)
    n_abstain = defaultdict(int)
    # fired-only (memory actually injected) sub-tally: bucket -> cond -> solved count / n
    fired_solved = defaultdict(lambda: defaultdict(int))
    fired_n = defaultdict(int)
    trace_out = args.out + ".jsonl"

    with open(trace_out, "w", encoding="utf-8") as tf:
        for i, (rec, bug_body, bug_out, bucket) in enumerate(tests):
            n_bucket[bucket] += 1
            cp = rec.get("code_prompt", "") or ""
            buggy_full = cp + bug_body
            task_id = str(rec.get("task_id"))
            fclass, _ = normalize_failure(bug_out)

            dx = diagnose_mechanism(buggy_full, bug_out, synth_pol)
            if not dx:
                n_empty_dx[bucket] += 1

            # ---- HYBRID retrieval from the real bank (drop same-task skills) ----
            rows = []
            if dx:
                got = bank.retrieve(dx, k=args.retrieve_k + 4, q_failure_class=fclass,
                                    q_buggy_full=buggy_full, with_scores=True)
                for (score, ch, s) in got:
                    if task_id in skill_tasks.get(s.skill_id, ()):
                        continue    # leakage: skill came from this very task
                    rows.append((score, ch, s))
                    if len(rows) >= args.retrieve_k:
                        break

            instr_nomem = synth_nomem(buggy_full, bug_out, dx, synth_pol)
            mem_raw = synth_mem(buggy_full, bug_out, dx, rows, synth_pol) if rows else _NULL
            abstained = (mem_raw == _NULL) or not rows
            # NULL gate: on abstain, memory arm == no-memory arm (never regresses below it)
            instr_mem = instr_nomem if abstained else mem_raw
            if abstained:
                n_abstain[bucket] += 1

            fix_diff = short_diff(bug_body, rec["canonical_solution"])
            caps = {"naked": "", "dx_synth": cap_instruction(instr_nomem),
                    "dx_mem": cap_instruction(instr_mem), "oracle": cap_oracle(fix_diff)}
            row = {"task_id": task_id, "bucket": bucket, "diagnosis": dx,
                   "abstained": abstained,
                   "retrieved": [{"name": s.name, "score": round(sc, 3),
                                  "sem": ch["sem"], "fclass": ch["fclass"], "struct": ch["struct"],
                                  "failure_class": s.failure_class,
                                  "site": f"{s.site_type}/{s.operator}" if s.site_type else "",
                                  "repair_strategy": s.repair_strategy}
                                 for (sc, ch, s) in rows],
                   "instr_nomem": instr_nomem, "instr_mem": instr_mem}
            for c in conds:
                p, k = solve(rec, bug_body, bug_out, caps[c], fixer_pol, verifier, args.k)
                solved[bucket][c] += int(p > 0)
                sr[bucket][c] += (p / k if k else 0.0)
                row[c] = {"passes": p, "solved": p > 0}
            for c in ("dx_synth", "dx_mem", "oracle"):
                nk, cd = row["naked"]["solved"], row[c]["solved"]
                if cd and not nk:
                    paired[bucket][c][0] += 1
                elif nk and not cd:
                    paired[bucket][c][1] += 1
            if not abstained:                 # memory genuinely fired here
                fired_n[bucket] += 1
                for c in ("dx_synth", "dx_mem"):
                    fired_solved[bucket][c] += int(row[c]["solved"])

            tf.write(json.dumps(row) + "\n")
            tf.flush()
            tag = "ABSTAIN" if abstained else f"{len(rows)} skills"
            print(f"  [{i+1}/{len(tests)}] {task_id} [{bucket}] {tag} dx='{dx[:36]}': "
                  + " ".join(f"{c[:5]}={'P' if row[c]['solved'] else '.'}" for c in conds),
                  flush=True)

    # ------------------------------------------------------------- report ----
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    buckets = [b for b in ("assertion", "crash", "other") if n_bucket[b]]
    N = sum(n_bucket.values())
    out("\n=== Repair-memory read path: diagnose -> HYBRID retrieve (real bank) -> "
        "null-gated synth -> weak fixer ===")
    out(f"  bank={len(bank)} skills | synth={args.synth_model} (FIXED across dx_synth/dx_mem) "
        f"| fixer={args.fixer_model}@0.2 | k={args.k}\n")
    out("  solved@k (>=1 of k passes):")
    out(f"  {'stratum':12}{'n':>4}" + "".join(f"{c:>13}" for c in conds))
    for b in buckets:
        n = n_bucket[b]
        out(f"  {b:12}{n:>4}" + "".join(f"{100*solved[b][c]/n:>12.0f}%" for c in conds))
    if N:
        out(f"  {'POOLED':12}{N:>4}" + "".join(
            f"{100*sum(solved[b][c] for b in buckets)/N:>12.0f}%" for c in conds))

    out("\n  mean solve_rate (passes/k, pooled -- lower variance than solved@k):")
    for c in conds:
        tot = sum(sr[b][c] for b in buckets)
        out(f"    {c:16}{(tot/N if N else 0):.3f}")

    out("\n  paired vs naked (rescue = solved only with it; harm = broke only with it):")
    for b in buckets:
        for c in ("dx_synth", "dx_mem", "oracle"):
            r, h = paired[b][c]
            out(f"    {b:10} {c:10} rescue={r:>2} harm={h:>2} net={r-h:+d}")

    out("\n  NULL gate:")
    for b in buckets:
        ab, n = n_abstain[b], n_bucket[b]
        out(f"    {b:10} abstained {ab:>2}/{n:<3} ({100*ab/n:>3.0f}%)  empty_dx={n_empty_dx[b]}")
    Nf = sum(fired_n.values())
    out(f"\n  FIRED-ONLY sub-analysis (memory actually injected; N={Nf}) -- the fair "
        "test of the operator:")
    if Nf:
        for c in ("dx_synth", "dx_mem"):
            tot = sum(fired_solved[b][c] for b in buckets)
            out(f"    {c:16}solved@k {100*tot/Nf:>3.0f}%  ({tot}/{Nf})")
        out("    (dx_mem > dx_synth here = the retrieved OPERATOR added signal beyond "
            "the plain diagnosis)")

    out("\n  KEY: dx_mem>dx_synth = memory helps where it fires. dx_synth>naked = diagnosis "
        "helps. POOLED dx_mem>=dx_synth (via the gate) = no net regression. assertion "
        "stratum is memory's fair shot (mechanism not already in the traceback).")
    out(f"\n  trace (diagnoses, retrieved skills + fusion channels, both instructions) -> {trace_out}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _EXEC.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
