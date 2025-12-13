import argparse
import json
import os
from typing import List

from ..eval_metrics import compute_metrics_from_json

""""

Utility to recompute and add NLI-based metrics to existing result JSON files.

Example usage:

1) python recompute_metrics.py --txt paths.txt

2) python recompute_metrics.py results/run1.json results/run2.json

3) python recompute_metrics.py results/*.json --nli_threshold 0.65

4) python recompute_metrics.py results/*.json --overwrite

5) python recompute_metrics.py --txt paths.txt --out_dir updated_results

"""

def load_paths(input_paths: List[str], txt_file: str | None) -> List[str]:
    paths = []

    if txt_file is not None:
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p.endswith(".json"):
                    paths.append(p)

    # add command-line JSON paths
    for p in input_paths:
        if p.endswith(".json"):
            paths.append(p)

    # deduplicate while preserving order
    seen = set()
    final = []
    for p in paths:
        if p not in seen:
            final.append(p)
            seen.add(p)

    return final


def update_json_file(
    path: str,
    metrics: dict,
    out_dir: str | None = None,
    overwrite: bool = False,
):
    # load original json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # insert metrics at top-level:
    data["metrics"] = metrics

    # determine output path
    if overwrite:
        out_path = path
    else:
        if out_dir is None:
            # default: write next to input file with "_updated" suffix
            base, ext = os.path.splitext(path)
            out_path = f"{base}_updated.json"
        else:
            os.makedirs(out_dir, exist_ok=True)
            name = os.path.basename(path)
            out_path = os.path.join(out_dir, name)

    # save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Recompute metrics for result JSON files.")
    ap.add_argument("json_paths", nargs="*", help="Paths to result JSON files.")
    ap.add_argument("--txt", help="Text file containing JSON paths (one per line).")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="Embedding similarity threshold.")
    ap.add_argument("--nli_threshold", type=float, default=None,
                    help="NLI resolution threshold. If None -> use embedding threshold.")
    ap.add_argument("--out_dir", default=None,
                    help="Directory to save updated JSONs (default=create *_updated.json).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite original JSON files instead of creating new ones.")

    args = ap.parse_args()

    # load paths
    paths = load_paths(args.json_paths, args.txt)
    if not paths:
        print("No JSON paths found.")
        return

    for p in paths:
        print(f"[info] Processing {p}")

        # compute metrics using your existing logic
        metrics = compute_metrics_from_json(
            p,
            embed_threshold=args.threshold,
            nli_threshold=args.nli_threshold,
        )

        # write updated json
        out_path = update_json_file(
            p,
            metrics=metrics,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
        )
        print(f"[info] Saved: {out_path}")


if __name__ == "__main__":
    main()

