"""Aggregate clip_benchmark per-(dataset, model, epoch) JSON outputs into a single CSV.

Each result JSON is shaped like:
    {
        "dataset": str,
        "model": str,         # arch name
        "pretrained": str,    # full checkpoint path (.../checkpoints/epoch_N)
        "task": str,
        "metrics": {...},
        "language": str,
    }

Output CSV columns: model_name, epoch, arch, language, dataset, task,
                    plus one column per metric key, sorted lexicographically.
"""

import argparse
import csv
import json
import os
import sys
from glob import glob


def find_result_jsons(results_root):
    return sorted(glob(os.path.join(results_root, "**", "*.json"), recursive=True))


def parse_pretrained(pretrained):
    """Pull (model_name, epoch) out of '.../<model_name>/checkpoints/epoch_<N>'."""
    parts = os.path.normpath(pretrained).split(os.sep)
    epoch_seg = parts[-1]  # epoch_4
    epoch = int(epoch_seg.split("_")[-1])
    model_name = parts[-3]  # ... / model_name / checkpoints / epoch_X
    return model_name, epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=str, help="root dir containing per-checkpoint JSON outputs")
    parser.add_argument("--output", type=str, default=None, help="output CSV path (default: <results_root>/summary.csv)")
    args = parser.parse_args()

    out_path = args.output or os.path.join(args.results_root, "summary.csv")

    files = find_result_jsons(args.results_root)
    if not files:
        print(f"No JSON files under {args.results_root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    metric_keys = set()
    for path in files:
        with open(path) as f:
            data = json.load(f)
        model_name, epoch = parse_pretrained(data["pretrained"])
        row = {
            "model_name": model_name,
            "epoch": epoch,
            "arch": data["model"],
            "language": data.get("language", ""),
            "dataset": data["dataset"],
            "task": data["task"],
            "json_path": os.path.relpath(path, args.results_root),
        }
        for k, v in data.get("metrics", {}).items():
            row[k] = v
            metric_keys.add(k)
        rows.append(row)

    base_cols = ["model_name", "epoch", "arch", "language", "dataset", "task"]
    metric_cols = sorted(metric_keys)
    fieldnames = base_cols + metric_cols + ["json_path"]

    rows.sort(key=lambda r: (r["model_name"], r["epoch"], r["dataset"], r["task"]))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(f"Models: {sorted({r['model_name'] for r in rows})}")
    print(f"Datasets: {sorted({r['dataset'] for r in rows})}")
    print(f"Metric columns: {metric_cols}")


if __name__ == "__main__":
    main()
