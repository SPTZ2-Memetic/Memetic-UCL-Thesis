# Thesis figure & analysis scripts (MEMETIC)

Scripts that generate the figures/tables in the thesis and compute the reported
numbers. Drop this folder into the repo (e.g. as `analysis/` or `paper/`).

**Dependencies:** `python>=3.10`, `matplotlib`, `numpy`. Nothing else; no gateway/LLM
calls — these only read the run outputs and plot.

**Inputs** (produced by the self-play run / eval probes, under `runs/`):
- `runs/selfplay_thesis/skillbank.json`   — generator SkillBank (386 skills)
- `runs/selfplay_thesis/repairbank.json`  — fixer RepairBank (1298 skills, 2465 cases)
- `runs/eval_thesis/pass1.txt.jsonl`, `pass2.txt.jsonl` — held-out eval (per-bug)
- generator ablation run dirs (`runs/e3_*`), capability-sweep eval dirs (`runs/e5_*`)

Several of the `figures/` scripts embed the final aggregate numbers inline (so they
reproduce the exact published figure without re-reading the runs); the `analysis/`
scripts recompute those numbers from the run outputs above.

## figures/  (script -> thesis figure)

| script | figure (label) | what it draws |
|---|---|---|
| `make_figs.py` | fig2_bank_growth, fig3_sites, fig45_frontier_meansr, fig7_lifecycle, fig10_utility_harm, fig11_eval_by_source, fig12_eval_strata | the E1 self-play + eval figures (growth, site composition, frontier, lifecycle, utility, by-source, by-stratum) |
| `build_difficulty_fig.py` | `fig:difficulty` | per-bug solve-rate distribution |
| `build_gen_figs.py` | `fig:gen-scatter`, `fig:reuse` | generator generalisation scatter + reuse histogram |
| `build_reliability_fig.py` | `fig:reliability` | 4-attempt outcome redistribution |
| `build_transition_fig.py` | `fig:lifecycle-flow` | RepairBank lifecycle transitions per window |
| `build_ablation_fig.py` | `fig:gen-ablation` | E3 generator ablations (exploration / difficulty-target) |
| `build_e5_fig.py` | `fig:e5-capability` | E5 capability sweep (naked/memory/escalation per fixer) |
| `build_e5_persource_bars.py` | `fig:e5-persource` | E5 per-source grouped bars, faceted by fixer |
| `build_e1_tables.py`, `compute_tables.py` | `tab:*` | emits the LaTeX table bodies from the banks/eval |

## analysis/  (recompute reported numbers)

| script | computes |
|---|---|
| `inspect_banks.py` | bank sizes, status split, provenance |
| `reuse_dist.py` | skill reuse distribution, single-task tail |
| `site_dist.py` | site-type composition / entropy |
| `semdup.py`, `sem_fields.py` | near-duplicate clustering of mechanisms |
| `gen_check.py`, `rows.py` | generator per-skill tables |
| `eval_family.py` | held-out eval by failure class |

## probe_stage_ablation.py

The generator retrieval-stage ablation (Stage A / embedding / Stage B judge).
Standalone; see its module docstring for flags. Routes `gpt*`→OpenAI and
`qwen*`→a Qwen host via env vars, or a generic OpenAI-compatible gateway by default.

---
**Note on `distinct_tasks`:** the bank's stored `distinct_tasks` field is a merge
counter and equals the case/application count; the thesis tables use the *true*
unique task count, computed as
`len(set(cases[cid].task_id for cid in skill.case_ids))`. The `analysis/` scripts
compute it that way — do not read `skill.distinct_tasks` directly.
