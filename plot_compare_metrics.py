import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# tqdm (optional)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# Paper-friendly text sizes
mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# Metrics: first two are rates (left axis), last one is count (right axis)
METRICS = [
    ("Resolved (proxy)", ["resolved_proxy_rate"]),                          # left axis
    ("Resolved (NLI)",   ["resolved_proxy_rate_nli", "nli_resolved_rate"]), # left axis
    ("#Questions",       ["avg_num_questions"]),                            # right axis
]


def iter_json_files(results_root: Path, include_regex: Optional[str] = None) -> List[Path]:
    files = sorted([p for p in results_root.rglob("*.json") if p.is_file()])
    if include_regex:
        rgx = re.compile(include_regex)
        files = [p for p in files if rgx.search(str(p))]
    return files


def is_baseline(path: Path, baseline_tag: str) -> bool:
    return baseline_tag in path.name


def load_metrics_from_json(path: Path) -> Optional[Dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("metrics"), dict):
        return obj["metrics"]
    return None


def compute_metrics_if_possible(path: Path) -> Optional[Dict]:
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
    vals: List[float] = []
    for _, keys in METRICS:
        found = None
        for k in keys:
            v = m.get(k, None)
            if isinstance(v, (int, float)):
                found = float(v)
                break
        if found is None:
            return None
        vals.append(found)
    return vals


def summarize(arr: np.ndarray, err: str = "sem") -> Tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)

    if err == "none" or arr.shape[0] <= 1:
        return mean, np.zeros_like(mean)

    std = arr.std(axis=0, ddof=1)
    if err == "std":
        return mean, std

    sem = std / np.sqrt(arr.shape[0])
    return mean, sem


def infer_title(results_root: Path) -> str:
    s = str(results_root)

    vocab = None
    for v in ["clar_vocab2", "question"]:
        if v in s:
            vocab = v
            break

    model = None
    for p in list(results_root.rglob("*.json"))[:80]:
        m = re.search(r"_([29]b)_", p.name)
        if m:
            model = m.group(1)
            break

    parts = []
    if model:
        parts.append(model)
    if vocab:
        parts.append(f"vocab={vocab}")
    return " · ".join(parts) if parts else results_root.name


def collect_condition_vectors(
    json_paths: List[Path],
    desc: str,
) -> Tuple[np.ndarray, int, int, int]:
    vectors: List[List[float]] = []
    n_total = len(json_paths)
    n_skipped = 0

    for p in tqdm(json_paths, desc=desc, unit="file"):
        m = load_metrics_from_json(p)
        if m is None:
            m = compute_metrics_if_possible(p)
        if m is None:
            n_skipped += 1
            continue

        vec = extract_metric_vector(m)
        if vec is None:
            n_skipped += 1
            continue

        vectors.append(vec)

    if not vectors:
        needed = ", ".join(sorted({k for _, keys in METRICS for k in keys}))
        raise SystemExit(
            f"No usable runs found for {desc}.\n"
            f"Expected metric keys (one per metric): {needed}\n"
            "Tip: ensure your run JSONs contain top-level 'metrics', or run from the repo root so "
            "eval_metrics.py is importable."
        )

    arr = np.array(vectors, dtype=float)
    return arr, n_total, arr.shape[0], n_skipped


def plot_grouped_two_axes(
    base_mean: np.ndarray,
    base_err: np.ndarray,
    steer_mean: np.ndarray,
    steer_err: np.ndarray,
    out_path: str,
    title: str,
    error_label: str,
):
    labels = [lab for lab, _ in METRICS]
    x = np.arange(len(labels))
    width = 0.36

    # left axis for rate metrics (0,1), right axis for questions (2)
    rate_idx = [0, 1]
    q_idx = [2]

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    # Choose consistent colors across both axes
    baseline_color = "tab:blue"
    steering_color = "tab:orange"

    # --- left axis bars (rates) ---
    ax_left.bar(
        x[rate_idx] - width / 2,
        base_mean[rate_idx],
        width,
        yerr=base_err[rate_idx],
        capsize=4,
        color=baseline_color,
    )
    ax_left.bar(
        x[rate_idx] + width / 2,
        steer_mean[rate_idx],
        width,
        yerr=steer_err[rate_idx],
        capsize=4,
        color=steering_color,
    )

    ax_left.set_ylabel("Resolution rate")
    ax_left.set_ylim(0.0, 1.05)
    ax_left.grid(True, axis="y", alpha=0.3)

    # --- right axis bars (avg # questions) ---
    ax_right.bar(
        x[q_idx] - width / 2,
        base_mean[q_idx],
        width,
        yerr=base_err[q_idx],
        capsize=4,
        color=baseline_color,
    )
    ax_right.bar(
        x[q_idx] + width / 2,
        steer_mean[q_idx],
        width,
        yerr=steer_err[q_idx],
        capsize=4,
        color=steering_color,
    )

    ax_right.set_ylabel("Average #questions")

    # x labels / title
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_title(title)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=baseline_color, label="baseline"),
        Patch(facecolor=steering_color, label="steering"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    # Save
    if out_path.lower().endswith(".pdf"):
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    print(f"[ok] saved {out_path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="General results folder (scanned recursively for run JSON files)")
    ap.add_argument("--baseline_tag", default="featNone",
                    help="Substring that identifies baseline files in their filename (default: featNone)")
    ap.add_argument("--include_regex", default=None,
                    help="Optional regex to restrict which JSONs are considered (matches full path)")
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

    print(f"[info] found {len(baseline_jsons)} baseline JSON(s), {len(steering_jsons)} steering JSON(s)")

    base_arr, base_total, base_used, base_skipped = collect_condition_vectors(baseline_jsons, desc="Baseline")
    steer_arr, steer_total, steer_used, steer_skipped = collect_condition_vectors(steering_jsons, desc="Steering")

    base_mean, base_err = summarize(base_arr, err=args.error)
    steer_mean, steer_err = summarize(steer_arr, err=args.error)

    title = args.title if args.title is not None else f"Metric comparison · {infer_title(root)}"
    plot_grouped_two_axes(
        base_mean, base_err,
        steer_mean, steer_err,
        out_path=args.out,
        title=title,
        error_label=args.error,
    )

    print(
        f"[info] baseline: total_files={base_total}, used_runs={base_used}, skipped={base_skipped}\n"
        f"[info] steering:  total_files={steer_total}, used_runs={steer_used}, skipped={steer_skipped}"
    )


if __name__ == "__main__":
    main()
