"""Launch the v2 self-play loop: NEW generator (SkillBank + curriculum) + NEW
fixer (RepairBank + advisor + escalation write-back), both banks evolving, no
weights moved.

Roles / models (all frozen; three so each role can differ):
  --gen-model      generator + Stage B applicability judge   (frontier model)
  --fixer-model    writes the actual fix                       (weak, e.g. gpt-5-nano @0.2)
  --distill-model  advisor synth + write-back distill + redistill (reliable, e.g. qwen-coder)

Warm start:
  --skill-bank     generator SkillBank json (else the hand-authored SEED_SKILLS)
  --repair-bank    fixer RepairBank json (else an empty bank -- grows from fixer wins)
Evolving banks are saved to <out-dir>/skillbank.json / repairbank.json every
--rounds-save-every rounds.

Parallelism:
  --batch-size B   run B rounds' READ phase (generation + fixer episodes) concurrently
                   against a frozen bank snapshot, then COMMIT their bank writes
                   serially in seed order. B=1 is identical to the sequential loop;
                   validate equivalence at B=1, then scale (B=6-8 is a good start).

Smoke test first (cheap, catches integration bugs):
  python scripts/run_selfplay.py --rounds 3 --seed-sources human --seed-per-source 6 \
      --skill-bank runs/skill_bank.json --repair-bank runs/repair_bank.json \
      --gen-model gpt-5-mini --fixer-model gpt-5-nano --distill-model qwen-coder \
      --exec-python $HOME/verify-env/bin/python --out-dir runs/selfplay_smoke

Long run:
  nohup python scripts/run_selfplay.py --rounds 900 --batch-size 8 \
      --seed-sources human human_edited_lm qwen7b gpt_oss_20b --seed-per-source 60 \
      --skill-bank runs/skill_bank.json \
      --gen-model gpt-5-mini --fixer-model gpt-5-nano --distill-model qwen-coder \
      --exec-python $HOME/verify-env/bin/python --out-dir runs/selfplay_run \
      > runs/selfplay_run.log 2>&1 &
"""
import argparse, os, re, sys
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from memetic.embeddings import Embedder
from memetic.verifier import FunctionVerifier
from memetic.loader import load_bugbench_records, load_seed_pool, DEFAULT_SOURCES
from memetic.skills import SEED_SKILLS
from memetic.skill_bank import SkillBank
from memetic.fixer_memory import RepairBank
from memetic.repair_advisor import RepairAdvisor
from memetic.loop_v2 import SelfPlayConfig, run_selfplay


# ---- policy: hard-timeout wrapper + gateway-boilerplate strip ----
# Some gateways prepend a fixed boilerplate line to every completion; strip it
# so it never leaks into distilled text. Set GATEWAY_BOILERPLATE to that line
# (case-insensitive) for your provider; defaults to a harmless no-op phrase.
_GW_PHRASE = os.environ.get("GATEWAY_BOILERPLATE", "llm gateway api layer").strip().lower()
_GW_PREFIX = re.compile(r"(?im)^\s*" + re.escape(_GW_PHRASE) + r"\s*[:.\-]*\s*")


def _clean_gateway(s):
    s = _GW_PREFIX.sub("", s or "")
    out = []
    for line in s.splitlines():
        core = line.strip().strip("#").strip().strip("\"'").strip()
        if core.lower() == _GW_PHRASE:
            continue
        out.append(line)
    return "\n".join(out).strip()


class TimeoutPolicy:
    _EX = ThreadPoolExecutor(max_workers=16)

    def __init__(self, policy, timeout=90):
        self.p, self.timeout = policy, timeout

    def generate(self, prompt):
        try:
            out = self._EX.submit(self.p.generate, prompt).result(timeout=self.timeout) or ""
        except Exception:
            return ""
        return _clean_gateway(out)


def build_policy(model, temperature=0.7, max_tokens=2000):
    try:
        from memetic.prometheus_policy import PrometheusPolicy
    except Exception:
        from memetic.policy_prometheus import PrometheusPolicy
    api_base = os.environ.get("GATEWAY_API_BASE", "https://your-gateway.example.com")
    ca = os.environ.get("GATEWAY_CA_CERT", "secrets/gateway-ca.pem")
    return TimeoutPolicy(PrometheusPolicy(model=model, api_base=api_base, ca_cert=ca,
                                          temperature=temperature, max_tokens=max_tokens))


def load_seeds(sources, split, per_source):
    seeds = []
    for src in sources:
        n = 0
        for rec in load_bugbench_records(src, split, limit=per_source):
            seeds.append(rec)
            n += 1
        print(f"  seeds {src}: {n}", flush=True)
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed-sources", nargs="*",
                    default=["human", "human_edited_lm", "qwen7b", "gpt_oss_20b"])
    ap.add_argument("--seed-split", default="train")
    ap.add_argument("--seed-per-source", type=int, default=60,
                    help="(legacy) unused for generator seeding now; kept for compat")
    ap.add_argument("--seed-limit", type=int, default=0,
                    help="cap on clean generator tasks from bigcodebench_ours (0 = all ~900)")
    ap.add_argument("--real-bug-frac", type=float, default=0.0,
                    help="fraction of rounds that run a fixer-only episode on a REAL published "
                         "bug drawn from --seed-sources (reference-bug grounding). 0 = pure generation.")
    ap.add_argument("--real-per-source", type=int, default=0,
                    help="cap on real bugs loaded per source for the real-bug pool (0 = all)")
    ap.add_argument("--skill-bank", default=None, help="generator SkillBank json (else SEED_SKILLS)")
    ap.add_argument("--repair-bank", default=None, help="fixer RepairBank json (else empty)")
    ap.add_argument("--out-dir", default="runs/selfplay_run")
    ap.add_argument("--gen-model", default="gpt-5-mini")
    ap.add_argument("--fixer-model", default="gpt-5-nano")
    ap.add_argument("--distill-model", default="qwen-coder")
    ap.add_argument("--exec-python", default=None)
    ap.add_argument("--rounds-save-every", type=int, default=10)
    ap.add_argument("--n-total", type=int, default=4)
    ap.add_argument("--n-skilled", type=int, default=3)
    ap.add_argument("--fix-attempts", type=int, default=4)
    ap.add_argument("--retrieve-k", type=int, default=3, help="skills retrieved into the fixer per bug")
    ap.add_argument("--fixer-max-tokens", type=int, default=4000,
                    help="fixer completion budget; 1200 starved gpt-5-nano (reasoning ate the "
                         "budget -> empty fixes). 4000 leaves room for the fix after reasoning.")
    ap.add_argument("--judge-budget", type=int, default=12,
                    help="generator Stage-B applicability judge calls per round")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="rounds whose READ phase runs concurrently before a serial COMMIT. "
                         "B=1 == sequential (validate equivalence first), then scale (6-8).")
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="round-tagged bank snapshots (out-dir/checkpoints/*_r{n}.json) every N "
                         "rounds, for longitudinal per-skill tables; 0 to disable.")
    ap.add_argument("--no-redistill", action="store_true", help="don't grow the SkillBank in-loop")
    ap.add_argument("--no-gate", action="store_true", help="disable the fixer NULL gate (inject always)")
    ap.add_argument("--seed-rng", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_skill = os.path.join(args.out_dir, "skillbank.json")
    out_repair = os.path.join(args.out_dir, "repairbank.json")

    embedder = Embedder()

    def make_verifier():
        return FunctionVerifier(timeout_seconds=30, python_executable=args.exec_python)
    verifier = make_verifier()

    gen_policy = build_policy(args.gen_model, temperature=0.7)
    fixer_policy = build_policy(args.fixer_model, temperature=0.2, max_tokens=args.fixer_max_tokens)
    distill_policy = build_policy(args.distill_model, temperature=0.4, max_tokens=2000)

    # generator SkillBank
    if args.skill_bank and os.path.exists(args.skill_bank):
        skill_bank = SkillBank.load(args.skill_bank)
        print(f"loaded SkillBank: {len(skill_bank)} skills <- {args.skill_bank}", flush=True)
    else:
        skill_bank = SkillBank(list(SEED_SKILLS))
        print(f"SkillBank: {len(skill_bank)} hand-authored SEED_SKILLS (no --skill-bank given)", flush=True)

    # fixer RepairBank
    if args.repair_bank and os.path.exists(args.repair_bank):
        repair_bank = RepairBank.load(args.repair_bank, embedder)
        print(f"loaded RepairBank: {len(repair_bank)} skills <- {args.repair_bank}", flush=True)
    else:
        repair_bank = RepairBank(embedder)
        print("RepairBank: empty (no --repair-bank given; grows from fixer wins)", flush=True)

    advisor = RepairAdvisor(repair_bank, distill_policy,
                            retrieve_k=args.retrieve_k, gate=not args.no_gate)

    # generator seeds: the FULL clean training pool (bigcodebench_ours train split),
    # NOT the 4 bug-source datasets -- seeding from the bug sources (capped 60 each)
    # was the coverage bug in the last run.
    seeds = load_seed_pool(args.seed_split, limit=(args.seed_limit or None))
    # real-bug pool (reference-bug grounding): published bugs from the 4 sources, same split.
    real_bugs = []
    if args.real_bug_frac > 0:
        for src in args.seed_sources:
            real_bugs.extend(load_bugbench_records(src, args.seed_split,
                                                   limit=(args.real_per_source or None)))
    print(f"seeds (clean pool): {len(seeds)} | real bugs: {len(real_bugs)} | "
          f"rounds={args.rounds} | batch_size={args.batch_size} | "
          f"real_bug_frac={args.real_bug_frac}\n", flush=True)

    cfg = SelfPlayConfig(
        n_total=args.n_total, n_skilled_target=args.n_skilled,
        fix_attempts=args.fix_attempts, judge_budget=args.judge_budget,
        save_every=args.rounds_save_every, redistill=not args.no_redistill,
        trace_path=os.path.join(args.out_dir, "trace.jsonl"),
        metrics_path=os.path.join(args.out_dir, "metrics.jsonl"),
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=os.path.join(args.out_dir, "checkpoints"))

    log_path = os.path.join(args.out_dir, "selfplay.log")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    results = run_selfplay(
        seeds, skill_bank=skill_bank, repair_bank=repair_bank, advisor=advisor,
        gen_policy=gen_policy, fixer_policy=fixer_policy, distill_policy=distill_policy,
        verifier=verifier, embedder=embedder, cfg=cfg, rounds=args.rounds,
        seed_rng=args.seed_rng, skill_bank_path=out_skill, repair_bank_path=out_repair,
        batch_size=args.batch_size, verifier_factory=make_verifier,
        real_bugs=real_bugs, real_bug_frac=args.real_bug_frac, log=log)

    # ---- summary ----
    n_valid = sum(r.n_valid for r in results)
    n_prod = sum(1 for r in results for b in r.bugs if b.productive)
    srs = [b.solve_rate for r in results for b in r.bugs if b.solve_rate is not None]
    log("\n=== self-play finished ===")
    log(f"  rounds={len(results)} valid_bugs={n_valid} productive={n_prod} "
        f"({100*n_prod/max(1,n_valid):.0f}% of valid)")
    log(f"  mean solve_rate over valid bugs: {sum(srs)/len(srs):.3f}" if srs else "  no valid bugs")
    log(f"  final SkillBank={len(skill_bank)}  RepairBank={len(repair_bank)} "
        f"{dict(repair_bank.counts_by_status())}")
    log(f"  banks saved -> {out_skill} , {out_repair}")
    logf.close()


if __name__ == "__main__":
    main()
