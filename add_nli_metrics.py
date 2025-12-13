import argparse
import json
import os
from typing import List, Union

from eval_metrics import compute_metrics_from_json


"""
Utility to recompute and add NLI-based metrics to existing result JSON files.

Examples:

1) python add_nli_metrics.py --txt paths.txt
2) python add_nli_metrics.py results/run1.json results/run2.json
3) python add_nli_metrics.py results/*.json --nli_threshold 0.65
4) python add_nli_metrics.py results/*.json --overwrite
5) python add_nli_metrics.py --txt paths.txt --out_dir updated_results
"""


def load_paths(input_paths: List[str], txt_file: str | None) -> List[str]:
    paths = []

    if txt_file is not None:
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p.endswith(".json"):
                    paths.append(p)

    for p in input_paths:
        if p.endswith(".json"):
            paths.append(p)

    # deduplicate, preserve order
    seen = set()
    final = []
    for p in paths:
        if p not in seen:
            final.append(p)
            seen.add(p)
    return final


def main():
    ap = argparse.ArgumentParser(description="Recompute and add NLI metrics to result JSON files.")
    ap.add_argument("json_paths", nargs="*", help="Paths to result JSON files.")
    ap.add_argument("--txt", help="Text file containing JSON paths (one per line).")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="Embedding similarity threshold.")
    ap.add_argument("--nli_threshold", type=float, default=None,
                    help="NLI resolution threshold (defaults to embedding threshold).")
    ap.add_argument("--out_dir", default=None,
                    help="Directory to save updated JSONs (default: *_updated.json).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite original JSON files.")

    args = ap.parse_args()

    paths = load_paths(args.json_paths, args.txt)
    if not paths:
        print("[error] No JSON paths found.")
        return

    for path in paths:
        print(f"[info] Processing {path}")

        # --- Load JSON ONCE ---
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # --- Compute metrics (mutates obj['examples'] in-place) ---
        metrics = compute_metrics_from_json(
            obj,
            embed_threshold=args.threshold,
            nli_threshold=args.nli_threshold,
        )

        # --- Attach metrics ---
        obj["metrics"] = metrics

        # --- Determine output path ---
        if args.overwrite:
            out_path = path
        else:
            if args.out_dir is not None:
                os.makedirs(args.out_dir, exist_ok=True)
                out_path = os.path.join(args.out_dir, os.path.basename(path))
            else:
                base, ext = os.path.splitext(path)
                out_path = f"{base}_updated.json"

        # --- Save updated JSON ---
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

        print(f"[info] Saved → {out_path}")


if __name__ == "__main__":
    main()
