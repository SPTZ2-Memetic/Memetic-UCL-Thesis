# memetic

Generator/fixer self-play with memory over frozen frontier models (no weight
training). Two roles co-evolve against 2 respective growing memory mechanisms: a **generator**
that proposes bugs guided by a `SkillBank` curriculum, and a **fixer** that
repairs them with retrieval-augmented advice from a `RepairBank`. Neither model's
weights are updated; the only learning signal flows through the two banks and the
curriculum's difficulty EMA.

## Layout

```
src/memetic/        # the package
scripts/            # entry points (self-play run + held-out eval)
```

Key modules:

- `loop_v2` -- the top-level self-play orchestration loop (read/commit phases,
  batched parallelism, reference-bug mixing).
- `generation`, `skills`, `skill_bank`, `matching`, `judge_skill` -- the
  generator side (skilled generation + applicability judging + curriculum).
- `repair_fixer`, `repair_advisor`, `fixer_memory`, `fault` -- the fixer side
  (escalation episodes + retrieval-augmented advice + the RepairBank).
- `distill`, `dedup`, `redistill` -- turning verified fixes into reusable skills.
- `verifier` -- sandboxed function verification.
- `embeddings`, `loader`, `policy_prometheus`, `codeutils`, `sites`,
  `constructs`, `seed`, `types`, `interfaces` -- shared primitives.

## Install

```bash
pip install -e .
```

## Running

The full thesis pipeline (self-play run, then the full-test eval) is driven by:

```bash
bash scripts/run_thesis_pipeline.sh
```

The self-play run and the held-out retrieval eval can also be launched
individually via `scripts/run_selfplay.py` and `scripts/probe_retrieval_v2.py`.

The pipeline expects a frozen-model gateway, a local code embedder, and the
bug/seed datasets; these are not distributed with the repository.
## Citation

This code accompanies the master's thesis *Memetic Code Repair: Evolving
Debugging Knowledge through Self-Play* (University College London, 2026). If you
use it, please cite:

```bibtex
@mastersthesis{anonymous2026memetic,
  author = {Anonymous},
  title  = {Memetic Code Repair: Evolving Debugging Knowledge through Self-Play},
  school = {University College London},
  year   = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
