#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Reproduce Table 1 of the paper.
# Trains all four UniDA methods on both tasks and prints the summary table.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."   # ensure we are in the repo root

METHODS=(ppot mlnet eiakda lead)
CONFIGS=(configs/occupancy.yaml configs/activity.yaml)

mkdir -p results

for cfg in "${CONFIGS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "──────────────────────────────────────────"
    echo "  Config: $cfg  |  Method: $method"
    echo "──────────────────────────────────────────"
    python train.py --config "$cfg" --method "$method"
  done
done

echo ""
echo "══════════════  Final Results (Table 1)  ══════════════"
python evaluate.py --results_dir results/
