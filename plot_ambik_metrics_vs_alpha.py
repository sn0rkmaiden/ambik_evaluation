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
python plot_ambik_metrics_vs_alpha.py \
  --results_root all_results \
  --out images/ambik_metrics_vs_alpha.pdf \
  --error sem \
  --title "AmbiK · gemma-2-9b-it · metrics vs steering strength"

"""

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


METRICS = [
    ("resolved_proxy_rate", ["resolved_proxy_rate"]),
    ("nli_resolved_rate",   ["resolved_proxy_rate_nli", "nli_resolved_rate"]),
    ("avg_num_questions",   ["avg_num_questions"]),
]


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
        from eval_metrics import compute_metrics_from_json  # type: ignore
    except Exception:
        return None
    try:
        m = compute_metrics_from_json(str(path))
        return m if isinstance(m, dict) else None
    except Exception:
        return None


def parse_alpha_from_path(path: Path, baseline_tag: str = "featNone") -> Optional[float]:
    """
    Supports patterns like:
      - str3.0 / str15 / str1
      - alpha0.5 / alpha_0.5
      - a0.5
    Baseline (featNone) -> alpha=0.0
    """
    s = str(path)

    if baseline_tag and baseline_tag in path.name:
        return 0.0

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
    return mu, std / np.sqrt(arr.size)  # sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True, help="Folder with AmbiK run JSONs")
    ap.add_argument("--include_regex", default=None, help="Optional regex to restrict files (matches full path)")
    ap.add_argument("--baseline_tag", default="featNone", help="Substring identifying baseline filenames")
    ap.add_argument("--error", choices=["sem", "std", "none"], default="sem")
    ap.add_argument("--out", default="ambik_metrics_vs_alpha.pdf")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    root = Path(args.results_root)
    files = iter_json_files(root, include_regex=args.include_regex)
    if not files:
        raise SystemExit(f"No JSON files found under: {root}")

    by_alpha: Dict[float, Dict[str, List[float]]] = {}

    for p in tqdm(files, desc="Reading runs", unit="file"):
        alpha = parse_alpha_from_path(p, baseline_tag=args.baseline_tag)
        if alpha is None:
            continue

        m = load_metrics_from_json(p)
        if m is None:
            m = compute_metrics_if_possible(p)
        if m is None:
            continue

        for metric_label, keys in METRICS:
            v = get_metric_value(m, keys)
            if v is None:
                continue
            by_alpha.setdefault(alpha, {}).setdefault(metric_label, []).append(v)

    if not by_alpha:
        raise SystemExit("No usable runs found. Check your folder and filename patterns for alpha/featNone.")

    alphas = sorted(by_alpha.keys())

    series = {}
    for metric_label, _ in METRICS:
        means = []
        errs = []
        for a in alphas:
            vals = by_alpha.get(a, {}).get(metric_label, [])
            if not vals:
                means.append(np.nan)
                errs.append(0.0)
            else:
                mu, er = mean_and_err(vals, args.error)
                means.append(mu)
                errs.append(er)
        series[metric_label] = (np.array(means, dtype=float), np.array(errs, dtype=float))

    plt.figure(figsize=(10, 5))
    for metric_label, (means, errs) in series.items():
        plt.plot(alphas, means, marker="o", label=metric_label)
        if args.error != "none":
            plt.errorbar(alphas, means, yerr=errs, fmt="none", capsize=4)

    plt.xlabel("Steering strength")
    plt.ylabel("Metric value")
    if args.title is None:
        plt.title("AmbiK: metrics vs steering strength")
    else:
        plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out.lower().endswith(".pdf"):
        plt.savefig(args.out, bbox_inches="tight")
    else:
        plt.savefig(args.out, dpi=300, bbox_inches="tight")

    print(f"[ok] saved {args.out}")
    print("[info] counts per alpha (per metric):")
    for a in alphas:
        counts = {k: len(by_alpha.get(a, {}).get(k, [])) for k, _ in METRICS}
        print(f"  alpha={a}: {counts}")


if __name__ == "__main__":
    main()
