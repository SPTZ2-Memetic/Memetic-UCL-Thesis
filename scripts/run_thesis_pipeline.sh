#!/usr/bin/env bash
# Definitive thesis pipeline: the full self-play run, then the full-test eval,
# run automatically one after the other. Launch once and leave it:
#
#   cd ~/memetic
#   nohup bash scripts/run_thesis_pipeline.sh > runs/thesis_pipeline.log 2>&1 &
#   tail -f runs/thesis_pipeline.log
#
# The eval runs ONLY if self-play exits cleanly (exit 0), so a crashed run never
# gets scored as if it were the definitive bank. Eval runs on ALL test data of all
# four sources (--test-per-source 0), twice, so you get a variance band to average.
set -uo pipefail
cd "$(dirname "$0")/.."                       # repo root (~/memetic)

PY="${EXEC_PYTHON:-$HOME/verify-env/bin/python}"
OUT="runs/selfplay_thesis"
EVAL="runs/eval_thesis"
mkdir -p "$EVAL"

echo "=== [$(date)] SELF-PLAY starting -> $OUT ==="
python scripts/run_selfplay.py \
  --rounds 1350 --batch-size 8 --real-bug-frac 0.3333 \
  --seed-sources human human_edited_lm qwen7b gpt_oss_20b \
  --skill-bank runs/skill_bank.json \
  --gen-model gpt-5-mini --fixer-model gpt-5-nano --distill-model qwen-coder \
  --exec-python "$PY" \
  --checkpoint-every 150 \
  --out-dir "$OUT"
RC=$?

if [[ $RC -ne 0 ]]; then
  echo "=== [$(date)] self-play exited $RC ; NOT auto-running eval (bank is not definitive) ==="
  exit $RC
fi

BANK="$OUT/repairbank.json"
if [[ ! -s "$BANK" ]]; then
  echo "=== [$(date)] self-play exited 0 but $BANK is missing/empty ; skipping eval ==="
  exit 1
fi

echo "=== [$(date)] SELF-PLAY done. bank=$BANK . Full-test eval starting (all 4 sources, ALL test data) ==="
for pass in 1 2; do
  echo "--- [$(date)] eval pass $pass ---"
  python scripts/probe_retrieval_v2.py \
    --old-bank "$BANK" --new-bank "$OUT/repairbank_fp.json" \
    --test-sources human human_edited_lm qwen7b gpt_oss_20b --test-per-source 0 \
    --arms naked,dx_synth,dx_mem_new \
    --fixer-model gpt-5-nano --synth-model qwen-coder \
    --exec-python "$PY" --workers 8 \
    --out "$EVAL/pass${pass}.txt" \
    || echo "--- eval pass $pass failed (continuing) ---"
done

echo "=== [$(date)] ALL DONE. eval -> $EVAL/pass1.txt , $EVAL/pass2.txt (average the two) ==="
