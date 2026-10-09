#!/usr/bin/env python
"""
probe_stage_ablation.py -- E4 RETRIEVAL-STAGE ablation for the GENERATOR's
skill-applicability pipeline.

The generator decides which stored bug-mechanism skills apply to a clean
canonical through three stages:

    Stage A  structural routing   match_sites_to_skills      (deterministic)
    pre-rank embedding ranker     rank_candidates_by_embedding (MiniLM, cheap)
    Stage B  applicability judge   judge_skill_applicability   (LLM, expensive)

This probe asks whether each pre-filter stage earns its place, measured as
applicability RECALL / PRECISION against JUDGE COST (the only expensive stage).
It is a RETRIEVAL/judging experiment: no bug generation, no fixer solve loop.

CONFIGS (what pre-filters precede the Stage-B judge):
  struct_llm      Stage A -> judge EVERY structural survivor (no cap).
                  The thorough, tractable REFERENCE applicable set.
  full            Stage A -> embedding-rank -> top-B -> judge. Production.
  struct_randcap  Stage A -> RANDOM cap to B -> judge. Embedding OFF at the
                  same budget as `full` -> isolates the EMBEDDING ranker.
  embed_llm       NO structure: embed the full site x skill cross-product ->
                  top-B -> judge. Isolates the STRUCTURAL router.
  llm_only        NO structure, NO embedding: judge the whole cross-product.
                  The exhaustive gold; ~sites x |bank| judge calls/canonical,
                  so OFF by default and meant for a capped pool / small sample.

DESIGN DECISIONS (see the thesis write-up):
  * Reference defaults to `struct_llm`, not `llm_only`: a literal judge-
    everything arm over a 386-skill bank is tens of thousands of gpt-5-mini
    calls. Judging every structural survivor is the most complete labelling
    that is tractable. Enable `llm_only` (capped pool) only to test whether
    structural routing itself discards applicable pairs.
  * A verdict CACHE keyed on (task, site, skill) judges each unique pair once
    and shares the verdict across arms: this removes judge stochasticity as a
    confound (arms differ only by WHICH pairs they send) and cuts cost. The
    per-arm COST reported is "pairs sent" (production cost), not LLM calls made.
  * Judge model defaults to gpt-5-mini (the deployed Stage-B judge).
  * Budget B defaults to 40 (the methodology's judge_budget).
  * Leakage: canonicals from bigcodebench_ours train, minus --exclude-tasks.

EXAMPLE
  set -a; source .env; set +a
  python scripts/probe_stage_ablation.py \
    --bank runs/selfplay_thesis/skillbank.json \
    --exclude-tasks runs/selfplay_thesis/skillbank.json.task_ids.json \
    --n-canonicals 80 --judge-budget 40 --judge-model gpt-5-mini \
    --workers 6 --out runs/e4_stage_ablation.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from memetic.loader import load_seed_pool                       # noqa: E402
from memetic.seed import reconstruct_function                   # production canonical builder
try:                                                                # noqa: E402
    from memetic.loop import record_is_runnable                 # matches probe_skill_matching
except Exception:                                                   # pragma: no cover
    try:
        from memetic.loop_v2 import record_is_runnable
    except Exception:
        def record_is_runnable(_rec):  # fallback: do not filter
            return True
from memetic.matching import (                                  # noqa: E402
    match_sites_to_skills, rank_candidates_by_embedding, CandidatePair)
from memetic.sites import extract_meaningful_sites              # noqa: E402
from memetic.skills import SEED_SKILLS                          # noqa: E402
from memetic.skill_bank import SkillBank                        # noqa: E402
from memetic.judge_skill import judge_skill_applicability       # noqa: E402
from memetic.embeddings import Embedder                         # noqa: E402

ALL_CONFIGS = ("struct_llm", "full", "struct_randcap", "embed_llm", "llm_only")


# --------------------------------------------------------------------------- #
#  plumbing (mirrors scripts/probe_skill_matching.py)                         #
# --------------------------------------------------------------------------- #
def build_policy(model, temperature=0.3, max_tokens=300):
    """Route each model to its provider. gpt* -> OpenAI (OPENAI_API_KEY),
    qwen* -> a Qwen host (QWEN_API_KEY, default Together), else a generic
    OpenAI-compatible gateway. Public providers verify TLS against the system
    CA bundle (ca_cert=True); only the gateway fallback uses a custom cert."""
    try:
        from memetic.prometheus_policy import PrometheusPolicy
    except Exception:
        from memetic.policy_prometheus import PrometheusPolicy
    m = (model or "").lower()
    if m.startswith("gpt"):
        return PrometheusPolicy(
            model=model,
            api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com"),
            ca_cert=True, api_key_env="OPENAI_API_KEY",
            temperature=temperature, max_tokens=max_tokens)
    if m.startswith("qwen"):
        return PrometheusPolicy(
            model=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
            api_base=os.environ.get("QWEN_API_BASE", "https://api.together.xyz"),
            ca_cert=True, api_key_env="QWEN_API_KEY",
            temperature=temperature, max_tokens=max_tokens)
    return PrometheusPolicy(
        model=model,
        api_base=os.environ.get("GATEWAY_API_BASE", "https://your-gateway.example.com"),
        ca_cert=os.environ.get("GATEWAY_CA_CERT", "secrets/gateway-ca.pem"),
        temperature=temperature, max_tokens=max_tokens)


def normalize_task_id(tid: str) -> str:
    return tid.strip().lower().replace("/", "_")


def load_excluded(path: str) -> set:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        by_source = json.load(f)
    return {normalize_task_id(t) for ids in by_source.values() for t in ids}


def load_skills(args):
    if args.bank:
        bank = SkillBank.load(args.bank)
        skills = bank.all()
        print(f"loaded {len(skills)} distilled skills <- {args.bank}", flush=True)
    else:
        skills = list(SEED_SKILLS)
        print(f"--bank not given; falling back to {len(skills)} SEED_SKILLS", flush=True)
    # Optional pool cap: only relevant when `llm_only` is enabled (the
    # exhaustive arm is ~sites x |pool| judge calls). ALL arms share the
    # capped pool so the comparison stays fair.
    if args.skill_pool_cap and len(skills) > args.skill_pool_cap:
        rng = random.Random(args.seed)
        skills = rng.sample(skills, args.skill_pool_cap)
        print(f"capped skill pool to {len(skills)} (--skill-pool-cap, shared by all arms)",
              flush=True)
    return skills


# --------------------------------------------------------------------------- #
#  keys + candidate construction                                              #
# --------------------------------------------------------------------------- #
def site_key(site):
    return (getattr(site, "lineno", None), (getattr(site, "source_text", "") or "")[:160])


def pair_key(task_id, p):
    return (task_id, p.skill.skill_id, site_key(p.site))


def cross_pairs(canonical, skills, max_sites):
    sites = extract_meaningful_sites(canonical, max_sites=max_sites)
    return [CandidatePair(site=s, skill=k) for s in sites for k in skills]


def random_cap(pairs, b, seed):
    if len(pairs) <= b:
        return list(pairs)
    return random.Random(seed).sample(list(pairs), b)


def config_pairs(cfg, canonical, skills, embedder, b, max_sites, seed):
    """The (site, skill) pairs a given config would SEND to the Stage-B judge."""
    if cfg in ("struct_llm", "full", "struct_randcap"):
        survivors = match_sites_to_skills(canonical, skills, max_sites=max_sites)
        if cfg == "struct_llm":
            return survivors
        if cfg == "full":
            return rank_candidates_by_embedding(survivors, embedder, top_k=b)
        return random_cap(survivors, b, seed)           # struct_randcap
    cross = cross_pairs(canonical, skills, max_sites)
    if cfg == "embed_llm":
        return rank_candidates_by_embedding(cross, embedder, top_k=b)
    return cross                                         # llm_only


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default="runs/selfplay_thesis/skillbank.json",
                    help="distilled generator SkillBank json")
    ap.add_argument("--exclude-tasks", default="",
                    help="json manifest of task_ids the bank was distilled from "
                         "(namespace-normalised and removed from the canonical pool)")
    ap.add_argument("--pool-limit", type=int, default=0,
                    help="rows to read from the clean train pool (0 = all); "
                         "canonicals are sampled from these")
    ap.add_argument("--n-canonicals", type=int, default=80)
    ap.add_argument("--judge-model", default="gpt-5-mini")
    ap.add_argument("--judge-budget", type=int, default=40,
                    help="B: judge cap for full / struct_randcap / embed_llm")
    ap.add_argument("--max-sites", type=int, default=0,
                    help="cap meaningful sites per canonical (0 = no cap); "
                         "mainly bounds the llm_only cross-product")
    ap.add_argument("--arms", default="struct_llm,full,struct_randcap,embed_llm",
                    help="comma list from: " + ",".join(ALL_CONFIGS))
    ap.add_argument("--reference", default="auto",
                    help="config whose applicable set defines recall; "
                         "'auto' = llm_only if enabled else struct_llm")
    ap.add_argument("--skill-pool-cap", type=int, default=0,
                    help="cap the shared skill pool (recommended only with llm_only)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/e4_stage_ablation.txt")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ALL_CONFIGS:
            raise SystemExit(f"unknown arm '{a}'; choose from {ALL_CONFIGS}")
    ref = args.reference
    if ref == "auto":
        ref = "llm_only" if "llm_only" in arms else "struct_llm"
    if ref not in arms:
        raise SystemExit(f"--reference {ref} must be one of the enabled --arms {arms}")
    max_sites = None if args.max_sites <= 0 else args.max_sites

    print(f"arms={arms} | reference={ref} | B={args.judge_budget} | "
          f"judge={args.judge_model} | workers={args.workers}", flush=True)
    if "llm_only" in arms and not args.skill_pool_cap:
        print("WARNING: llm_only enabled without --skill-pool-cap -- the "
              "exhaustive arm judges sites x |bank| pairs per canonical.", flush=True)

    skills = load_skills(args)
    excluded = load_excluded(args.exclude_tasks)
    if args.exclude_tasks:
        print(f"excluding {len(excluded)} distillation task_ids <- {args.exclude_tasks}",
              flush=True)
    elif args.bank:
        print("WARNING: --bank without --exclude-tasks: pool may contain "
              "canonicals the bank was distilled from (recall, not generalisation).",
              flush=True)

    pool_limit = None if args.pool_limit <= 0 else args.pool_limit
    raw = [s for s in load_seed_pool("train", limit=pool_limit) if record_is_runnable(s)]
    canon = [s for s in raw if normalize_task_id(s["task_id"]) not in excluded]
    canon = canon[:args.n_canonicals]
    print(f"{len(raw)} runnable canonicals, {len(canon)} after exclusion/sampling\n",
          flush=True)

    embedder = Embedder()
    policy = build_policy(args.judge_model)

    # per-arm accumulators
    agg = {a: {"sent": 0, "applicable": 0, "ref_hit": 0, "appl_skills": 0} for a in arms}
    ref_total = 0
    trace_path = args.out + ".jsonl"
    t0 = time.time()

    with open(trace_path, "w", encoding="utf-8") as tf:
        for i, rec in enumerate(canon):
            task_id = rec["task_id"]
            canonical = reconstruct_function(rec, "canonical_solution")  # exact production input

            # 1) build each arm's sent-pairs (deterministic given --seed)
            per_arm = {a: config_pairs(a, canonical, skills, embedder,
                                       args.judge_budget, max_sites, args.seed)
                       for a in arms}

            # 2) judge the UNION once, cache by pair key (no judge noise across arms)
            cache = {}
            union = {}
            for a in arms:
                for p in per_arm[a]:
                    union[pair_key(task_id, p)] = (canonical, p)

            def work(item):
                k, (c, p) = item
                v = judge_skill_applicability(c, p.site, p.skill, policy)
                return k, bool(v.applicable)

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                for k, applic in ex.map(work, list(union.items())):
                    cache[k] = applic

            # 3) per-arm applicable sets + reference
            arm_appl = {}
            for a in arms:
                keys = [pair_key(task_id, p) for p in per_arm[a]]
                appl = {k for k in keys if cache.get(k)}
                arm_appl[a] = appl
                skills_appl = {k[1] for k in appl}  # distinct skill_ids judged applicable
                agg[a]["sent"] += len(per_arm[a])
                agg[a]["applicable"] += len(appl)
                agg[a]["appl_skills"] += len(skills_appl)
            ref_appl = arm_appl[ref]
            ref_total += len(ref_appl)
            for a in arms:
                agg[a]["ref_hit"] += len(arm_appl[a] & ref_appl)

            tf.write(json.dumps({
                "i": i, "task_id": task_id,
                "arms": {a: {"sent": len(per_arm[a]),
                             "applicable": len(arm_appl[a]),
                             "ref_recall_hit": len(arm_appl[a] & ref_appl)}
                         for a in arms},
                "ref": ref, "ref_applicable": len(ref_appl),
            }) + "\n")
            tf.flush()
            print(f"  [{i+1}/{len(canon)}] {task_id}  "
                  + "  ".join(f"{a}:{len(per_arm[a])}->{len(arm_appl[a])}" for a in arms),
                  flush=True)

    # ----------------------------------------------------------------- report
    n = len(canon)
    lines = []
    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out(f"\n=== Generator stage ablation (Stage A / embedding / Stage B judge) ===")
    out(f"  canonicals={n} | skills={len(skills)} | B={args.judge_budget} | "
        f"judge={args.judge_model} | reference={ref}")
    out(f"  elapsed {time.time()-t0:.0f}s\n")
    out(f"  {'config':<16}{'judge calls/can':>16}{'applicable/can':>16}"
        f"{'precision':>12}{'recall vs ref':>16}{'appl skills/can':>18}")
    for a in arms:
        sent = agg[a]["sent"]
        appl = agg[a]["applicable"]
        prec = (appl / sent) if sent else 0.0
        rec = (agg[a]["ref_hit"] / ref_total) if ref_total else 0.0
        out(f"  {a:<16}{sent/n:>16.1f}{appl/n:>16.2f}{prec:>12.2f}"
            f"{rec:>16.2f}{agg[a]['appl_skills']/n:>18.2f}")
    out("")
    out("  cost      = judge calls/canonical = pairs an arm SENDS to Stage B (production cost).")
    out("  precision = applicable / sent (how much judge budget lands on applicable pairs).")
    out(f"  recall    = applicable pairs also found by the reference ('{ref}'), / reference total.")
    out("  KEY: full vs struct_randcap = value of the embedding ranker at equal budget;")
    out("       full vs embed_llm      = value of structural routing;")
    out("       struct_llm vs full     = applicable pairs lost to capping, and the cost saved.")
    out(f"\n  trace -> {trace_path}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

