import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
Usage example:
    python plot_compare_metrics.py \
    --results_root results/results_layer20_question \
    --out compare_layer20_question.pdf \
    --include_regex "str3\.0" \
    --error sem
"""


# ---------- Paper-friendly styling (bigger text) ----------
mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# ---------- Metrics to plot ----------
METRICS = [
    ("Resolved (proxy)", "resolved_proxy_rate"),
    ("Resolved (NLI)",   "nli_resolved_rate"),
    ("#Questions",       "avg_num_questions"),
]


def iter_json_files(results_root: Path, include_regex: Optional[str] = None) -> List[Path]:
    files = sorted([p for p in results_root.rglob("*.json") if p.is_file()])
    if include_regex:
        rgx = re.compile(include_regex)
        files = [p for p in files if rgx.search(str(p))]
    return files


def load_metrics_from_json(path: Path) -> Optional[Dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("metrics"), dict):
        return obj["metrics"]
    return None


def compute_metrics_if_possible(path: Path) -> Optional[Dict]:
    """
    Tries to compute metrics using eval_metrics.compute_metrics_from_json.
    Run from the AmbiK evaluation repo root so the import works.
    """
    try:
        from eval_metrics import compute_metrics_from_json  # type: ignore
    except Exception:
        return None

    try:
        m = compute_metrics_from_json(str(path))
        return m if isinstance(m, dict) else None
    except Exception:
        return None


def extract_metric_vector(m: Dict) -> Optional[List[float]]:
    vals = []
    for _, key in METRICS:
        v = m.get(key, None)
        if not isinstance(v, (int, float)):
            return None
        vals.append(float(v))
    return vals


def is_baseline(path: Path, baseline_tag: str) -> bool:
    return baseline_tag in path.name


def summarize(arr: np.ndarray, err: str = "sem") -> Tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    if err == "none" or arr.shape[0] == 1:
        return mean, np.zeros_like(mean)

    std = arr.std(axis=0, ddof=1)
    if err == "std":
        return mean, std

    # sem
    sem = std / np.sqrt(arr.shape[0])
    return mean, sem


def infer_title(results_root: Path) -> str:
    """
    Best-effort title from your folder/file naming conventions:
      - vocab often appears in folder names
      - model scale appears in filenames
    """
    s = str(results_root)

    vocab = None
    for v in ["clar_vocab2", "question"]:
        if v in s:
            vocab = v
            break
    if vocab is None:
        for part in results_root.parts:
            if "question" in part:
                vocab = "question"
                break
            if "clar" in part:
                vocab = part
                break

    # scan a few filenames for 2b/9b
    model = None
    for p in list(results_root.rglob("*.json"))[:50]:
        m = re.search(r"_([29]b)_", p.name)
        if m:
            model = m.group(1)
            break

    bits = []
    if model:
        bits.append(model)
    if vocab:
        bits.append(f"vocab={vocab}")
    return " · ".join(bits) if bits else results_root.name


def collect_condition_vectors(
    json_paths: List[Path],
    require_metrics: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
      - vectors: shape (n_runs, 3)
      - n_seen: number of json files inspected
      - n_used: number of runs successfully contributed
    """
    vectors: List[List[float]] = []
    n_seen = 0

    for p in json_paths:
        n_seen += 1

        m = load_metrics_from_json(p)
        if m is None:
            m = compute_metrics_if_possible(p)
        if m is None:
            if require_metrics:
                continue
            else:
                continue

        vec = extract_metric_vector(m)
        if vec is None:
            continue

        vectors.append(vec)

    if not vectors:
        raise SystemExit(
            "No usable runs found (missing metrics). Required keys: "
            + ", ".join(k for _, k in METRICS)
            + "\nTip: ensure your run JSONs have top-level 'metrics', or run from repo root so eval_metrics can be imported."
        )

    arr = np.array(vectors, dtype=float)
    return arr, n_seen, len(vectors)


def plot_grouped(
    base_mean: np.ndarray,
    base_err: np.ndarray,
    steer_mean: np.ndarray,
    steer_err: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str,
):
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, base_mean, width, yerr=base_err, capsize=4, label="baseline")
    plt.bar(x + width/2, steer_mean, width, yerr=steer_err, capsize=4, label="steering")

    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if out_path.lower().endswith(".pdf"):
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    print(f"[ok] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="General results folder")
    ap.add_argument("--baseline_tag", default="featNone",
                    help="Substring in baseline filenames (default: featNone)")
    ap.add_argument("--include_regex", default=None,
                    help="Optional regex to restrict which JSONs are considered (matches full path).")
    ap.add_argument("--out", default="compare_metrics.pdf",
                    help="Output path")
    ap.add_argument("--title", default=None,
                    help="Optional title override (default: inferred from path)")
    ap.add_argument("--error", choices=["sem", "std", "none"], default="sem",
                    help="Error bars across runs: sem/std/none")
    args = ap.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    all_jsons = iter_json_files(root, include_regex=args.include_regex)
    if not all_jsons:
        raise SystemExit(f"No JSON files found under: {root}")

    baseline_jsons = [p for p in all_jsons if is_baseline(p, args.baseline_tag)]
    steering_jsons = [p for p in all_jsons if not is_baseline(p, args.baseline_tag)]

    if not baseline_jsons:
        raise SystemExit(f"No baseline JSONs found (tag='{args.baseline_tag}') under: {root}")
    if not steering_jsons:
        raise SystemExit(f"No steering JSONs found (everything except tag='{args.baseline_tag}') under: {root}")

    base_arr, base_seen, base_used = collect_condition_vectors(baseline_jsons)
    steer_arr, steer_seen, steer_used = collect_condition_vectors(steering_jsons)

    base_mean, base_err = summarize(base_arr, err=args.error)
    steer_mean, steer_err = summarize(steer_arr, err=args.error)

    labels = [lab for lab, _ in METRICS]
    title = args.title if args.title is not None else f"Metric comparison · {infer_title(root)}"

    plot_grouped(
        base_mean, base_err,
        steer_mean, steer_err,
        labels=labels,
        out_path=args.out,
        title=title,
    )

    print(
        f"[info] baseline: files={len(baseline_jsons)} inspected={base_seen} used={base_used}\n"
        f"[info] steering:  files={len(steering_jsons)} inspected={steer_seen} used={steer_used}"
    )


if __name__ == "__main__":
    main()
