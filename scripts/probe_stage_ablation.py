#!/usr/bin/env python3
"""Bounded E4 generator retrieval-stage ablation for MEMETIC.

Production-comparable arms (each sends <= B site/skill pairs to the same judge):
  full           structural filter -> MiniLM top-B -> LLM applicability judge
  struct_randcap structural filter -> seeded random B -> same judge
  embed_llm      full site x skill cross-product -> MiniLM top-B -> same judge

Extra random audits estimate the applicability missed by (1) the capped
structurally-eligible population and (2) structural filtering itself.
The exact judge applies to ALL selected/audited pairs. The audit provides
statistical estimates, NOT ground-truth semantic applicability. No bugs are
generated or repaired here; this measures retrieval selection, not fixer gain.

Results: .txt report, .jsonl per-canonical rows, .pairs.jsonl paired verdict
trace, .sqlite persistent verdict cache, .tasks.json reproducible task manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from memetic.loader import load_seed_pool
from memetic.seed import reconstruct_function
try:
    from memetic.loop import record_is_runnable
except ImportError:
    from memetic.loop_v2 import record_is_runnable
from memetic.matching import match_sites_to_skills, rank_candidates_by_embedding, CandidatePair
from memetic.sites import extract_meaningful_sites
from memetic.skill_bank import SkillBank
from memetic.judge_skill import judge_skill_applicability
from memetic.embeddings import Embedder

ARMS = ("full", "struct_randcap", "embed_llm")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_digest(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def normalize_task_id(task_id):
    return str(task_id).strip().lower().replace("/", "_")


def read_exclusions(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        ids = [v for val in obj.values() for v in (val if isinstance(val, list) else [val])]
    elif isinstance(obj, list):
        ids = obj
    else:
        raise ValueError("Exclude manifest must be a list of task IDs or a dict of source -> ID lists")
    return {normalize_task_id(t) for t in ids}


def get_policy(model):
    try:
        from memetic.prometheus_policy import PrometheusPolicy
    except ImportError:
        from memetic.policy_prometheus import PrometheusPolicy

    api_base = os.environ.get("LLM_API_BASE", "").strip()
    if api_base:
        # Use the same max-token limit and temperature as the previous probe;
        # changing these would change the judge rather than only retrieval.
        return PrometheusPolicy(
            model=model, api_base=api_base, ca_cert=True,
            api_key_env=os.environ.get("LLM_API_KEY_ENV", "LLM_API_KEY"),
            temperature=0.3, max_tokens=8000,
            timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "300")))
    if model.startswith("gpt"):
        return PrometheusPolicy(model=model,
                                api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com"),
                                ca_cert=True, api_key_env="OPENAI_API_KEY",
                                temperature=0.3, max_tokens=8000)
    raise RuntimeError("Set LLM_API_BASE and LLM_API_KEY_ENV for this judge provider")


def site_key(site):
    # Same source and location => same logical site across constructors.
    return (getattr(site, "lineno", None),
            getattr(site, "col_offset", None),
            getattr(site, "end_lineno", None),
            (getattr(site, "source_text", "") or "")[:160])


def pair_id(pair):
    return (str(pair.skill.skill_id), site_key(pair.site))


def dedupe(pairs):
    result = {}
    for pair in pairs:
        result[pair_id(pair)] = pair
    return result


def skill_fingerprint(skill):
    try:
        contents = json.dumps(vars(skill), sort_keys=True, default=str, ensure_ascii=False)
    except TypeError:
        contents = repr(skill)
    return digest(contents)


def verdict_cache_key(task_id, canonical, pair, model, version, judge_code_sha):
    # Include model, endpoint, source, prompt revision and skill contents.
    fields = [version, model, os.environ.get("LLM_API_BASE", ""), judge_code_sha,
              task_id, digest(canonical), str(pair.skill.skill_id),
              skill_fingerprint(pair.skill), site_key(pair.site)]
    return digest(json.dumps(fields, sort_keys=True, default=str, ensure_ascii=False))


def sample_pairs(pairs_by_key, n, rng):
    keys = sorted(pairs_by_key.keys(), key=str)
    if n >= len(keys):
        return dict(pairs_by_key)
    return {k: pairs_by_key[k] for k in rng.sample(keys, n)}


def load_task_pool(args):
    """Prefer exclusion-filtered train; otherwise use disjoint 'test' split.

    We do NOT silently call an unfiltered train pool 'held-out'.
    """
    manifest = args.exclude_tasks or (args.bank + ".task_ids.json")
    if args.split == "auto":
        split = "train" if Path(manifest).exists() else "test"
    else:
        split = args.split
    if split == "train":
        if not Path(manifest).exists() and not args.allow_overlap:
            raise SystemExit(
                "Train split requested but no skill-provenance manifest found: " + manifest +
                "\nProvide --exclude-tasks PATH, use --split test, or use --allow-overlap "
                "ONLY for a non-held-out diagnostic.")
        excluded = read_exclusions(manifest) if Path(manifest).exists() else set()
        print(f"split=train; excluded {len(excluded)} tasks using {manifest if excluded else 'NONE (OVERLAP RISK)'}", flush=True)
    else:
        excluded = set()
        print("split=test (independent of train-bank construction, subject to dataset provenance)", flush=True)
    try:
        raw = load_seed_pool(split, limit=None if args.pool_limit <= 0 else args.pool_limit)
    except Exception as exc:
        raise SystemExit(f"Unable to load {split!r} split: {exc}. Provide a train task provenance manifest if needed.")
    # Unique task IDs: a benchmark may contain several bugs for the same canonical.
    unique = {}
    for rec in raw:
        tid = normalize_task_id(rec["task_id"])
        if tid in excluded or tid in unique or not record_is_runnable(rec):
            continue
        unique[tid] = rec
    pool = list(unique.values())
    if not pool:
        raise SystemExit(f"No eligible runnable canonicals in split={split} after exclusions.")
    rng = random.Random(args.seed)
    chosen = rng.sample(pool, min(args.n_canonicals, len(pool)))
    print(f"eligible unique canonicals={len(pool)}; sampled={len(chosen)} seed={args.seed}", flush=True)
    return chosen, split, manifest if split == "train" and excluded else None


def open_cache(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS verdicts (k TEXT PRIMARY KEY, applicable INTEGER NOT NULL, created REAL NOT NULL)")
    conn.commit()
    return conn


def bootstrap_paired(rows, a, b, samples=2000, seed=123):
    vals = [r["arms"][a]["applicable"] - r["arms"][b]["applicable"] for r in rows]
    n = len(vals)
    avg = sum(vals) / n
    if n == 1:
        return avg, None, None
    rng = random.Random(seed)
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    return avg, boots[int(.025 * samples)], boots[int(.975 * samples)]


def sample_estimate(N, n, hits):
    """Expansion estimator and SRS without replacement variance for binary labels."""
    if N == 0:
        return 0.0, 0.0
    if n == 0:
        raise ValueError("Audit size zero for a nonempty sampling stratum")
    p = hits / n
    total = N * p
    if n == N:
        return float(hits), 0.0
    if n <= 1:
        return total, float("nan")
    variance = (N ** 2) * (1 - n / N) * p * (1 - p) / (n - 1)
    return total, variance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="scripts/skillbank.json")
    parser.add_argument("--exclude-tasks", default="")
    parser.add_argument("--split", choices=("auto", "train", "test"), default="auto")
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--pool-limit", type=int, default=0)
    parser.add_argument("--n-canonicals", type=int, default=30)
    parser.add_argument("--judge-budget", type=int, default=40)
    parser.add_argument("--audit-struct", type=int, default=10,
                        help="sample structurally eligible pairs not selected by any arm")
    parser.add_argument("--audit-rejected", type=int, default=10,
                        help="sample structurally rejected pairs not selected by any arm")
    parser.add_argument("--judge-model", default="openai/gpt-5-mini")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sites", type=int, default=0, help="0=no cap, as production")
    parser.add_argument("--cache-version", default="v1", help="increment after changing any hidden judge prompts")
    parser.add_argument("--out", default="runs/e4_stage_ablation_final.txt")
    args = parser.parse_args()
    if min(args.judge_budget, args.audit_struct, args.audit_rejected, args.n_canonicals, args.workers) < 0 or args.judge_budget == 0 or args.workers == 0 or args.n_canonicals == 0:
        parser.error("n-canonicals, workers, and judge-budget must be positive; audit sizes must be nonnegative")
    if args.audit_struct == 0 or args.audit_rejected == 0:
        parser.error("Final design requires both audits > 0 to estimate missed applicability")
    if not Path(args.bank).exists():
        raise SystemExit(f"SkillBank not found: {args.bank}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    judge_file = ROOT / "src" / "memetic" / "judge_skill.py"
    judge_sha = file_digest(judge_file)
    skills = SkillBank.load(args.bank).all()
    print(f"SkillBank={args.bank} | skills={len(skills)} | B={args.judge_budget} | judge={args.judge_model}", flush=True)
    canonicals, split, manifest = load_task_pool(args)
    (Path(str(out_path) + ".tasks.json")).write_text(json.dumps({
        "task_ids": [r["task_id"] for r in canonicals],
        "split": split, "excluded_manifest": manifest,
        "bank_sha256": file_digest(args.bank), "judge_sha256": judge_sha,
        "model": args.judge_model, "B": args.judge_budget,
        "audit_struct": args.audit_struct, "audit_rejected": args.audit_rejected,
        "seed": args.seed, "max_sites": args.max_sites,
    }, indent=2), encoding="utf-8")
    policy = get_policy(args.judge_model)
    embedder = Embedder()
    cache = open_cache(str(out_path) + ".sqlite")
    rows = []
    newly_judged = 0
    cached_judged = 0
    start = time.time()
    try:
        with open(str(out_path) + ".jsonl", "w", encoding="utf-8") as trace, \
             open(str(out_path) + ".pairs.jsonl", "w", encoding="utf-8") as labels:
            for i, rec in enumerate(canonicals):
                task_id = str(rec["task_id"])
                canonical = reconstruct_function(rec, "canonical_solution")
                max_sites = None if args.max_sites <= 0 else args.max_sites
                # Build every candidate once, and identify the structural set.
                sites = extract_meaningful_sites(canonical, max_sites=max_sites)
                universe = dedupe(CandidatePair(site=s, skill=k) for s in sites for k in skills)
                structure = dedupe(match_sites_to_skills(canonical, skills, max_sites=max_sites))
                if not set(structure).issubset(universe):
                    raise RuntimeError(f"Structural matcher returned a site not in the cross-product: {task_id}. Check site keys.")
                rng = random.Random(digest(f"{args.seed}/{task_id}"))
                selected = {
                    "full": dedupe(rank_candidates_by_embedding(list(structure.values()), embedder, top_k=args.judge_budget)),
                    "struct_randcap": sample_pairs(structure, args.judge_budget, rng),
                    "embed_llm": dedupe(rank_candidates_by_embedding(list(universe.values()), embedder, top_k=args.judge_budget)),
                }
                selected_union = set().union(*(set(v) for v in selected.values()))
                unselected_struct = {k: v for k, v in structure.items() if k not in selected_union}
                rejected = {k: v for k, v in universe.items() if k not in structure and k not in selected_union}
                audit_eligible = sample_pairs(unselected_struct, args.audit_struct, rng)
                audit_reject = sample_pairs(rejected, args.audit_rejected, rng)
                judged = {}
                for part in (*selected.values(), audit_eligible, audit_reject):
                    judged.update(part)
                key_by_pair = {
                    pk: verdict_cache_key(task_id, canonical, pair, args.judge_model,
                                          args.cache_version, judge_sha)
                    for pk, pair in judged.items()
                }
                known = {}
                missing = {}
                for pk, pair in judged.items():
                    cv = cache.execute("SELECT applicable FROM verdicts WHERE k=?", (key_by_pair[pk],)).fetchone()
                    if cv is not None:
                        known[pk] = bool(cv[0])
                        cached_judged += 1
                    else:
                        missing[pk] = pair
                print(f"[{i+1}/{len(canonicals)}] {task_id}: sites={len(sites)} universe={len(universe)} "
                      f"struct={len(structure)} union={len(selected_union)} "
                      f"audit={len(audit_eligible)}+{len(audit_reject)} "
                      f"CACHED={len(known)} NEW_CALLS={len(missing)}", flush=True)
                def judge_one(pk, pair):
                    verdict = judge_skill_applicability(canonical, pair.site, pair.skill, policy)
                    if not hasattr(verdict, "applicable"):
                        raise RuntimeError(f"Judge did not return .applicable for task={task_id} pair={pk}")
                    return pk, bool(verdict.applicable)
                if missing:
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        futures = {executor.submit(judge_one, pk, pair): pk for pk, pair in missing.items()}
                        for future in as_completed(futures):
                            pk, is_applicable = future.result()  # errors terminate; completed labels remain cached
                            known[pk] = is_applicable
                            cache.execute("INSERT OR REPLACE INTO verdicts (k,applicable,created) VALUES (?,?,?)",
                                          (key_by_pair[pk], int(is_applicable), time.time()))
                            cache.commit()
                            newly_judged += 1
                arm_stats = {}
                for arm in ARMS:
                    pool = selected[arm]
                    applicable = sum(bool(known[pk]) for pk in pool)
                    distinct = len({pk[0] for pk in pool if known[pk]})
                    arm_stats[arm] = {"sent": len(pool), "applicable": applicable,
                                      "precision": applicable/len(pool) if pool else None,
                                      "applicable_skills": distinct}
                known_struct_positives = sum(known[pk] for pk in judged if pk in structure and pk not in audit_eligible)
                known_reject_positives = sum(known[pk] for pk in judged if pk not in structure and pk not in audit_reject)
                audit_struct_hits = sum(known[pk] for pk in audit_eligible)
                audit_reject_hits = sum(known[pk] for pk in audit_reject)
                struct_est, struct_var = sample_estimate(len(unselected_struct), len(audit_eligible), audit_struct_hits)
                reject_est, reject_var = sample_estimate(len(rejected), len(audit_reject), audit_reject_hits)
                row = {
                    "i": i, "task_id": task_id, "sites": len(sites),
                    "universe_pairs": len(universe), "struct_pairs": len(structure),
                    "unique_judged": len(judged), "arms": arm_stats,
                    "audit": {
                        "struct_omitted": {"population": len(unselected_struct), "sample": len(audit_eligible),
                                           "positive": audit_struct_hits, "estimated_positive": struct_est, "estimated_variance": struct_var},
                        "struct_rejected": {"population": len(rejected), "sample": len(audit_reject),
                                            "positive": audit_reject_hits, "estimated_positive": reject_est, "estimated_variance": reject_var},
                        "known_struct_positive": known_struct_positives,
                        "known_rejected_positive": known_reject_positives,
                        "estimated_struct_positive": known_struct_positives + struct_est,
                        "estimated_total_positive": known_struct_positives + known_reject_positives + struct_est + reject_est,
                    },
                }
                rows.append(row)
                trace.write(json.dumps(row, default=str) + "\n")
                trace.flush()
                for pk in judged:
                    labels.write(json.dumps({
                        "task_id": task_id, "skill_id": pk[0], "site": pk[1],
                        "selected_by": [arm for arm in ARMS if pk in selected[arm]],
                        "audit_group": "struct_omitted" if pk in audit_eligible else (
                            "struct_rejected" if pk in audit_reject else None),
                        "applicable": bool(known[pk]), "cache_key": key_by_pair[pk],
                    }, default=str) + "\n")
                labels.flush()
                print("    " + " | ".join(f"{a}: {v['applicable']}/{v['sent']}" for a, v in arm_stats.items()) +
                      f" | audit omitted {audit_struct_hits}/{len(audit_eligible)}; rejected {audit_reject_hits}/{len(audit_reject)}", flush=True)
    finally:
        cache.close()
    n = len(rows)
    elapsed = time.time() - start
    lines = []
    def out(s=""):
        lines.append(s)
        print(s, flush=True)
    out("=== MEMETIC E4: budget-matched generator retrieval ablation + sampled recall audit ===")
    out(f"canonicals={n} | skills={len(skills)} | B={args.judge_budget} | split={split} | seed={args.seed}")
    out(f"model={args.judge_model} | judge cached={cached_judged} | new API judgements={newly_judged} | elapsed={elapsed/60:.1f}min")
    out(f"{'arm':<18}{'sent/can':>12}{'appl/can':>12}{'precision':>13}{'skills/can':>13}")
    for arm in ARMS:
        sent = sum(r["arms"][arm]["sent"] for r in rows)
        app = sum(r["arms"][arm]["applicable"] for r in rows)
        distinct = sum(r["arms"][arm]["applicable_skills"] for r in rows)
        out(f"{arm:<18}{sent/n:>12.1f}{app/n:>12.2f}{app/sent if sent else float('nan'):>13.3f}{distinct/n:>13.2f}")
    out("\nPAIRED APPLICABLE-PAIR YIELD (per canonical; task-bootstrap 95% CIs):")
    for arm in ("struct_randcap", "embed_llm"):
        mean, lo, hi = bootstrap_paired(rows, "full", arm, seed=args.seed + 1)
        ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "n=1"
        out(f"  full minus {arm}: {mean:+.2f} applicable pairs/canonical; 95% bootstrap CI {ci}")
    known_s = sum(r["audit"]["known_struct_positive"] for r in rows)
    known_r = sum(r["audit"]["known_rejected_positive"] for r in rows)
    s_est = sum(r["audit"]["struct_omitted"]["estimated_positive"] for r in rows)
    r_est = sum(r["audit"]["struct_rejected"]["estimated_positive"] for r in rows)
    s_var = sum(r["audit"]["struct_omitted"]["estimated_variance"] for r in rows)
    r_var = sum(r["audit"]["struct_rejected"]["estimated_variance"] for r in rows)
    total = known_s + known_r + s_est + r_est
    tot_struct = known_s + s_est
    full_hits = sum(r["arms"]["full"]["applicable"] for r in rows)
    out("\nSAMPLED AUDIT (design-based expansion; judge-defined applicability):")
    out(f"  structurally eligible applicable estimate: {tot_struct:.1f} (observed selected + audit expansion)")
    out(f"  structurally rejected applicable estimate: {known_r+r_est:.1f} (observed embed-only + audit expansion)")
    out(f"  total applicable estimate: {total:.1f} from a site x skill universe")
    if total > 0:
        est_recall = full_hits / total
        if math.isfinite(s_var+r_var):
            se = full_hits * math.sqrt(s_var + r_var) / total**2
            out(f"  FULL estimated pair recall vs entire universe: {est_recall:.3f} "
                f"(approx. design SE={se:.3f}; normal 95% [{max(0,est_recall-1.96*se):.3f}, {min(1,est_recall+1.96*se):.3f}])")
        else:
            out(f"  FULL estimated pair recall vs entire universe: {est_recall:.3f} (audit too small for SE)")
        out(f"  structural-stage estimated pair recall: {tot_struct/total:.3f}")
        out(f"  FULL estimated pair recall within structural survivors: {full_hits/tot_struct:.3f}" if tot_struct else "  no structural positives")
    out("\nINTERPRETATION: Precision is agreement with the SAME LLM applicability judge, NOT human-labelled precision.")
    out("Recall is a weighted random-audit estimate, NOT exhaustive recall. Its normal-approximation interval")
    out("covers sampling error only, not judge errors, prompt bias, or between-task generalisation; small audits can under-cover.")
    out("Arm cost is the number of pairs that arm would send in isolation; unique API calls above include audits and cache reuse.")
    out("All models/skills/prompt versions must remain fixed; this experiment does not measure generated-bug quality.")
    out(f"Per-canonical trace: {out_path}.jsonl")
    out(f"Pair verdict trace: {out_path}.pairs.jsonl")
    out(f"Persistent cache: {out_path}.sqlite")
    out(f"Sampled tasks: {out_path}.tasks.json")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nREPORT WRITTEN: {out_path}", flush=True)


if __name__ == "__main__":
    main()

