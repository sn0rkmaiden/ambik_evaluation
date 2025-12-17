import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x


RATE_METRICS = [
    ("Resolved (proxy)", ["resolved_proxy_rate"]),
    ("Resolved (NLI)",   ["resolved_proxy_rate_nli", "nli_resolved_rate"]),
]

COUNT_METRIC = ("Avg. #Questions", ["avg_num_questions"])


def iter_json_files(root: Path, include_regex: Optional[str] = None) -> List[Path]:
    files = sorted([p for p in root.rglob("*.json") if p.is_file()])
    if include_regex:
        rgx = re.compile(include_regex)
        files = [p for p in files if rgx.search(str(p))]
    return files


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
        from eval_metrics import compute_metrics_from_json  
    except Exception:
        return None
    try:
        m = compute_metrics_from_json(str(path))
        return m if isinstance(m, dict) else None
    except Exception:
        return None


def parse_alpha_from_path(path: Path, baseline_tag: str = "featNone") -> Optional[float]:
    if baseline_tag and baseline_tag in path.name:
        return 0.0

    s = str(path)
    patterns = [
        r"(?:^|[^a-zA-Z])str([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[^a-zA-Z])alpha[_-]?([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[^a-zA-Z])a([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return float(m.group(1))
    return None


def get_metric_value(m: Dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        v = m.get(k, None)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def mean_and_err(vals: List[float], err: str) -> Tuple[float, float]:
    arr = np.array(vals, dtype=float)
    mu = float(arr.mean())
    if err == "none" or arr.size <= 1:
        return mu, 0.0
    std = float(arr.std(ddof=1))
    if err == "std":
        return mu, std
    return mu, std / np.sqrt(arr.size)  # SEM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--include_regex", default=None)
    ap.add_argument("--baseline_tag", default="featNone")
    ap.add_argument("--error", choices=["sem", "std", "none"], default="sem")
    ap.add_argument("--out", default="ambik_metrics_vs_alpha.pdf")
    ap.add_argument("--title", default="AmbiK: Metrics vs α")
    args = ap.parse_args()

    root = Path(args.results_root)
    files = iter_json_files(root, args.include_regex)
    if not files:
        raise SystemExit(f"No JSON files found under {root}")

    # by_alpha[alpha]["metric"] = [values]
    by_alpha: Dict[float, Dict[str, List[float]]] = {}

    for p in tqdm(files, desc="Processing runs", unit="file"):
        alpha = parse_alpha_from_path(p, args.baseline_tag)
        if alpha is None:
            continue

        m = load_metrics_from_json(p)
        if m is None:
            m = compute_metrics_if_possible(p)
        if m is None:
            continue

        for label, keys in RATE_METRICS:
            v = get_metric_value(m, keys)
            if v is not None:
                by_alpha.setdefault(alpha, {}).setdefault(label, []).append(v)

        label_q, keys_q = COUNT_METRIC
        vq = get_metric_value(m, keys_q)
        if vq is not None:
            by_alpha.setdefault(alpha, {}).setdefault(label_q, []).append(vq)

    if not by_alpha:
        raise SystemExit("No usable runs found (alpha parsing or metrics missing).")

    alphas = sorted(by_alpha.keys())

    # Prepare data
    rate_series = {}
    for label, _ in RATE_METRICS:
        means, errs = [], []
        for a in alphas:
            vals = by_alpha.get(a, {}).get(label, [])
            mu, er = mean_and_err(vals, args.error) if vals else (np.nan, 0.0)
            means.append(mu)
            errs.append(er)
        rate_series[label] = (np.array(means), np.array(errs))

    q_label, _ = COUNT_METRIC
    q_means, q_errs = [], []
    for a in alphas:
        vals = by_alpha.get(a, {}).get(q_label, [])
        mu, er = mean_and_err(vals, args.error) if vals else (np.nan, 0.0)
        q_means.append(mu)
        q_errs.append(er)

    # ---------- Plot ----------
    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    # Left axis: rates
    for label, (means, errs) in rate_series.items():
        ax_left.plot(alphas, means, marker="o", label=label)
        if args.error != "none":
            ax_left.errorbar(alphas, means, yerr=errs, fmt="none", capsize=4)

    ax_left.set_ylabel("Resolution rate")
    ax_left.set_ylim(0.0, 1.05)

    # Right axis: question count
    ax_right.plot(alphas, q_means, marker="s", linestyle="--", color="black", label=q_label)
    if args.error != "none":
        ax_right.errorbar(alphas, q_means, yerr=q_errs, fmt="none", capsize=4, color="black")

    ax_right.set_ylabel("Average #questions")

    ax_left.set_xlabel("steering strength")
    ax_left.grid(True, alpha=0.3)

    # Unified legend
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")

    plt.title(args.title)
    plt.tight_layout()

    if args.out.lower().endswith(".pdf"):
        plt.savefig(args.out, bbox_inches="tight")
    else:
        plt.savefig(args.out, dpi=300, bbox_inches="tight")

    print(f"[ok] saved {args.out}")
    print("[info] counts per alpha:")
    for a in alphas:
        counts = {k: len(v) for k, v in by_alpha[a].items()}
        print(f"  alpha={a}: {counts}")


if __name__ == "__main__":
    main()
