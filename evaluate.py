#!/usr/bin/env python3
"""
evaluate.py — Aggregate result JSONs and print a performance table (Table 1).

Usage
-----
python evaluate.py --results_dir results/
"""

import argparse
import json
import os
import glob


METHODS  = ["ppot", "mlnet", "eiakda", "lead"]
TASKS    = ["occupancy", "activity"]
METRICS  = ["accuracy", "precision", "recall", "f1", "ucdr"]


def load_results(results_dir: str) -> dict:
    data = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            r = json.load(f)
        key = (r["task"], r["method"])
        data[key] = r
    return data


def print_table(data: dict) -> None:
    # Replicate Table 1 layout from the paper
    col_w = 10

    header_task = {
        "occupancy": "Occupancy Estimation",
        "activity" : "Activity Recognition",
    }

    for task in TASKS:
        print(f"\n{'─' * 70}")
        print(f"  {header_task[task]}")
        print(f"{'─' * 70}")
        print(f"{'Method':<10}", end="")
        for m in METRICS:
            print(f"{m.upper():>{col_w}}", end="")
        print()
        print("─" * 70)

        for method in METHODS:
            key = (task, method)
            if key not in data:
                print(f"{method.upper():<10}  (no results)")
                continue
            r = data[key]
            print(f"{method.upper():<10}", end="")
            for m in METRICS:
                val = r.get(m, float("nan"))
                print(f"{val * 100:>{col_w}.1f}", end="")
            print()

    print(f"\n{'─' * 70}")
    print("  All values are percentages (%).  Best per column shown with *.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    data = load_results(args.results_dir)
    if not data:
        print(f"No result files found in {args.results_dir}/")
        print("Run: bash scripts/run_all.sh")
        return

    print_table(data)


if __name__ == "__main__":
    main()
