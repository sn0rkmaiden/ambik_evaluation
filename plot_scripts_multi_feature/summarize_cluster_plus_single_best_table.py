#!/usr/bin/env python3
"""
Summarize cluster-steering + (best) single-feature steering results as a strength-dynamics table.

Table format (as requested)
---------------------------
Model & Base & Layer & Cluster & Str1 & Str3 & Str5 & Str10 \\
\\midrule
2B & 0.080 & 12 & 0 & 0.100 & 0.130 & 0.100 & 0.000 \\

Where:
- "Cluster" is a numeric cluster id for cluster-steering runs (taken directly from the filename).
- Additional rows for single-feature steering are added as:
    Cluster = best_clar   (best single-feature run from clar_vocab2 / results_2b_clar)
    Cluster = best_question (best single-feature run from question_vocab / results_2b_question)
  These rows show, for each strength column, the BEST metric achieved across all single-feature runs
  (max over features / seeds) at that strength, for that model+layer+vocab.

Inputs
------
You can pass either directories or .zip files:
- --cluster_results : results from cluster steering (e.g. results_multi.zip)
- --single_results  : results from single-feature steering (e.g. results_without_rand_with_combined.zip)

The script excludes the *combined vocab* folder from single-feature results by default.

Metric definition
-----------------
For each JSON:
  metric(run) = mean over examples[*][metric_field]
  - booleans treated as 0/1
  - missing/non-numeric skipped

Baseline
--------
Baseline is computed per model from runs detected as baseline:
  filename contains baseline/non_steered/featnone
  OR (no cluster id AND no detectable strength)
Baseline value = agg(baseline-run metrics), with --agg (default mean).

Notes
-----
- If your cluster results contain multiple clustering configs per layer (different thr/nclust),
  cluster ids can collide. Use filters (--filter_layer/--filter_thr/--filter_nclust) to select one config.

How to run:
python plot_scripts_multi_feature/summarize_cluster_plus_single_best_table.py \
  --cluster_results all_results/combined/20_layer/results \
  --single_results results_without_rand_with_combined.zip \
  --out_tex result_plots_multi_feature/table.tex \
  --out_csv result_plots_multi_feature/table.csv \
  --metric_field resolved_proxy \
  --strengths 1,3,5,10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ------------------------------- Regexes -------------------------------

_STRENGTH_RE = re.compile(r"(?:^|[_-])str(?P<val>-?\d+(?:\.\d+)?)", re.IGNORECASE)
_MODEL_RE = re.compile(r"(?:^|[_/])(?P<size>2b|9b)(?:[^a-z0-9]|$)", re.IGNORECASE)
_CLUSTER_RE = re.compile(r"(?:^|[_-])cluster(?P<cid>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_LAYER_L_RE = re.compile(r"(?:^|[_-])l(?P<layer>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_LAYER_WORD_RE = re.compile(r"(?:^|[/_ -])layer[_-]?(?P<layer>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_NCLUST_RE = re.compile(r"(?:^|[_-])nclust(?P<n>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_THR_RE = re.compile(r"(?:^|[_-])thr(?P<thr>\d+(?:\.\d+)?)", re.IGNORECASE)
_SAE_LAYER_RE = re.compile(r"blocks\.(?P<layer>\d+)\.", re.IGNORECASE)

# single-feature steering: infer vocab from path
def infer_vocab_from_path(path: str) -> Optional[str]:
    lp = path.lower()
    if "clar_vocab2" in lp or "results_2b_clar" in lp:
        return "clar"
    if "question_vocab" in lp or "results_2b_question" in lp:
        return "question"
    if "combined" in lp:
        return "combined"
    return None


# ------------------------------- Helpers -------------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def infer_strength_from_filename(name: str) -> Optional[float]:
    m = _STRENGTH_RE.search(name)
    return _safe_float(m.group("val")) if m else None


def infer_model_size_from_path_or_json(path: str, run: Dict[str, Any]) -> str:
    model_name = (((run.get("run_info") or {}).get("model") or {}).get("model_name"))
    if isinstance(model_name, str):
        ln = model_name.lower()
        if "2b" in ln and "9b" not in ln:
            return "2B"
        if "9b" in ln:
            return "9B"
    m = _MODEL_RE.search(path.lower())
    return m.group("size").upper() if m else "UNK"


def infer_cluster_id_from_filename(path: str) -> Optional[int]:
    m = _CLUSTER_RE.search(Path(path).name)
    return int(m.group("cid")) if m else None


def infer_layer_from_anywhere(path: str, run: Dict[str, Any]) -> Optional[int]:
    """
    Priority:
      1) filename "l<digits>"
      2) path segment containing "layer<digits>"
      3) steering_cfg.sae_id like "blocks.<layer>.hook_resid_post"
    """
    name = Path(path).name
    m = _LAYER_L_RE.search(name)
    if m:
        return int(m.group("layer"))
    m = _LAYER_WORD_RE.search(path)
    if m:
        return int(m.group("layer"))

    ri = run.get("run_info") or {}
    scfg = ri.get("steering_cfg")
    if isinstance(scfg, dict):
        sae_id = scfg.get("sae_id")
        if isinstance(sae_id, str):
            m = _SAE_LAYER_RE.search(sae_id)
            if m:
                return int(m.group("layer"))
    return None


def infer_nclust_from_filename(path: str) -> Optional[int]:
    m = _NCLUST_RE.search(Path(path).name)
    return int(m.group("n")) if m else None


def infer_thr_from_filename(path: str) -> Optional[float]:
    m = _THR_RE.search(Path(path).name)
    return _safe_float(m.group("thr")) if m else None


def extract_strength(run: Dict[str, Any], path: str) -> Optional[float]:
    ri = run.get("run_info") or {}

    # 1) steering_cfg.strength (single-feature runs)
    scfg = ri.get("steering_cfg")
    if isinstance(scfg, dict):
        v = _safe_float(scfg.get("strength"))
        if v is not None:
            return v

    # 2) model.steering_strength (cluster/multi-feature runs)
    model = ri.get("model")
    if isinstance(model, dict):
        v = _safe_float(model.get("steering_strength"))
        if v is not None:
            return v

    # 3) filename fallback
    return infer_strength_from_filename(Path(path).name)


def is_baseline_run(run: Dict[str, Any], path: str) -> bool:
    name = Path(path).name.lower()
    if "baseline" in name or "non_steered" in name or "featnone" in name:
        return True

    cid = infer_cluster_id_from_filename(path)
    st = extract_strength(run, path)
    return (cid is None and st is None)


def compute_metric(run: Dict[str, Any], metric_field: str) -> float:
    examples = run.get("examples") or []
    if not isinstance(examples, list) or not examples:
        return float("nan")

    vals: List[float] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        v = ex.get(metric_field)
        if isinstance(v, bool):
            vals.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))

    return (sum(vals) / len(vals)) if vals else float("nan")


def agg(values: Sequence[float], how: str) -> float:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan")
    if how == "mean":
        return sum(clean) / len(clean)
    if how == "max":
        return max(clean)
    if how == "median":
        return statistics.median(clean)
    raise ValueError(how)


# ------------------------------- IO -------------------------------

@dataclass
class Row:
    model: str
    layer: Optional[int]
    cluster: Optional[int]        # numeric cluster id (cluster runs)
    strength: Optional[float]     # None for baseline
    metric: float
    path: str
    thr: Optional[float] = None
    nclust: Optional[int] = None
    vocab: Optional[str] = None   # single-feature only
    is_single: bool = False       # single-feature only


def iter_json_blobs(results_path: str) -> Iterable[Tuple[str, bytes]]:
    p = Path(results_path)
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".json"):
                    yield name, zf.read(name)
    else:
        for fp in p.rglob("*.json"):
            try:
                yield str(fp), fp.read_bytes()
            except Exception:
                continue


def load_rows(
    results_path: str,
    metric_field: str,
    exclude_combined_vocab: bool,
) -> List[Row]:
    out: List[Row] = []
    for vpath, blob in iter_json_blobs(results_path):
        try:
            run = json.loads(blob.decode("utf-8"))
        except Exception:
            continue

        # Optional exclusion for combined vocab (single-feature zip contains a combined_vocab folder)
        if exclude_combined_vocab and "combined_vocab" in vpath.lower():
            continue

        model = infer_model_size_from_path_or_json(vpath, run)
        layer = infer_layer_from_anywhere(vpath, run)
        cid = infer_cluster_id_from_filename(vpath)
        thr = infer_thr_from_filename(vpath)
        ncl = infer_nclust_from_filename(vpath)

        base = is_baseline_run(run, vpath)
        strength = None if base else extract_strength(run, vpath)

        metric_val = compute_metric(run, metric_field)

        ri = run.get("run_info") or {}
        scfg = ri.get("steering_cfg")
        is_single = isinstance(scfg, dict) and ("feature" in scfg)  # single-feature steering runs
        vocab = infer_vocab_from_path(vpath) if is_single else None
        if exclude_combined_vocab and vocab == "combined":
            continue

        out.append(Row(
            model=model, layer=layer, cluster=cid, strength=strength, metric=metric_val, path=vpath,
            thr=thr, nclust=ncl, vocab=vocab, is_single=is_single
        ))
    return out


# ------------------------------- Table building -------------------------------

def fmt_strength(s: float) -> str:
    return f"Str{int(s)}" if abs(s - round(s)) < 1e-9 else f"Str{s:g}"


def fmt_val(v: float) -> str:
    return "-" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.3f}"


def to_latex(header: List[str], rows: List[List[str]], caption: str, label: str) -> str:
    cols = "l" * len(header)
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append(f"\\begin{{tabular}}{{{cols}}}")
    out.append("\\toprule")
    out.append(" & ".join(header) + " \\\\")
    out.append("\\midrule")
    for r in rows:
        out.append(" & ".join(r) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def parse_strengths(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_results", required=True, help="Folder or zip with cluster-steering results JSONs")
    ap.add_argument("--single_results", required=True, help="Folder or zip with single-feature results JSONs")
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--metric_field", default="resolved_proxy")
    ap.add_argument("--agg", choices=["mean", "max", "median"], default="mean",
                    help="Aggregation for baselines and for multiple runs in the same cell (except single-best cells).")
    ap.add_argument("--strengths", default="1,3,5,10")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:cluster_vs_single_strength")

    # Filters for cluster runs (recommended if multiple configs exist)
    ap.add_argument("--filter_model", choices=["2B", "9B", "UNK"], default=None)
    ap.add_argument("--filter_layer", type=int, default=None)
    ap.add_argument("--filter_nclust", type=int, default=None)
    ap.add_argument("--filter_thr", type=float, default=None)

    ap.add_argument("--exclude_combined_vocab", action="store_true",
                    help="Exclude combined vocab folder from single-feature results (default ON).")
    ap.set_defaults(exclude_combined_vocab=True)

    ap.add_argument("--include_single_best", action="store_true",
                    help="Add rows best_clar and best_question from single-feature steering (default ON).")
    ap.set_defaults(include_single_best=True)

    args = ap.parse_args()

    strengths = parse_strengths(args.strengths)

    # Load rows
    rows_cluster = load_rows(args.cluster_results, metric_field=args.metric_field, exclude_combined_vocab=False)
    rows_single = load_rows(args.single_results, metric_field=args.metric_field, exclude_combined_vocab=args.exclude_combined_vocab)

    all_rows = rows_cluster + rows_single

    # Baseline per model (from BOTH sources)
    baseline_by_model: Dict[str, List[float]] = {}
    for r in all_rows:
        if r.strength is None:
            baseline_by_model.setdefault(r.model, []).append(r.metric)
    baseline_val_by_model = {m: agg(vs, args.agg) for m, vs in baseline_by_model.items()}

    # ---------------- Cluster table rows ----------------
    # Filter cluster rows
    cluster_cells: Dict[Tuple[str, int, int, float], List[float]] = {}
    for r in rows_cluster:
        if r.strength is None:
            continue
        if r.cluster is None:
            continue
        if r.layer is None:
            continue

        if args.filter_model and r.model != args.filter_model:
            continue
        if args.filter_layer is not None and r.layer != args.filter_layer:
            continue
        if args.filter_nclust is not None and r.nclust != args.filter_nclust:
            continue
        if args.filter_thr is not None:
            if r.thr is None or abs(r.thr - args.filter_thr) > 1e-9:
                continue

        cluster_cells.setdefault((r.model, r.layer, r.cluster, float(r.strength)), []).append(r.metric)

    cluster_keys = sorted({(m, layer, cid) for (m, layer, cid, _s) in cluster_cells.keys()},
                          key=lambda x: (x[0], x[1], x[2]))

    # ---------------- Single-feature best rows ----------------
    # For each (model, layer, vocab, strength), pick the BEST metric across runs (max over features/seeds).
    single_best: Dict[Tuple[str, int, str, float], float] = {}
    if args.include_single_best:
        for r in rows_single:
            if not r.is_single:
                continue
            if r.strength is None or r.layer is None or r.vocab is None:
                continue
            if r.vocab == "combined":
                continue
            if args.filter_model and r.model != args.filter_model:
                continue
            if args.filter_layer is not None and r.layer != args.filter_layer:
                continue

            key = (r.model, r.layer, r.vocab, float(r.strength))
            prev = single_best.get(key, float("nan"))
            if math.isnan(prev) or (not math.isnan(r.metric) and r.metric > prev):
                single_best[key] = r.metric

    # Build table rows
    header = ["Model", "Base", "Layer", "Cluster"] + [fmt_strength(s) for s in strengths]
    table_rows: List[List[str]] = []

    # Add cluster rows
    for (model, layer, cid) in cluster_keys:
        base = baseline_val_by_model.get(model, float("nan"))
        row = [model, fmt_val(base), str(layer), str(cid)]
        for st in strengths:
            vals = cluster_cells.get((model, layer, cid, float(st)), [])
            row.append(fmt_val(agg(vals, args.agg)))
        table_rows.append(row)

    # Add best single-feature rows (one per vocab) AFTER clusters for that model+layer.
    if args.include_single_best:
        # Determine which (model, layer) pairs exist in cluster table (or single table if cluster empty)
        model_layers = sorted({(m, int(layer)) for (m, layer, _cid) in cluster_keys} |
                              {(r.model, r.layer) for r in rows_single if r.layer is not None},
                              key=lambda x: (x[0], x[1]))

        for (model, layer) in model_layers:
            for vocab in ["clar", "question"]:
                # Check if any strength exists for this row
                exists = any((model, layer, vocab, float(st)) in single_best for st in strengths)
                if not exists:
                    continue
                base = baseline_val_by_model.get(model, float("nan"))
                row = [model, fmt_val(base), str(layer), f"best_{vocab}"]
                for st in strengths:
                    v = single_best.get((model, layer, vocab, float(st)), float("nan"))
                    row.append(fmt_val(v))
                table_rows.append(row)

    caption = args.caption or f"AmbiK {args.metric_field}: cluster steering vs best single-feature steering"
    tex = to_latex(header, table_rows, caption=caption, label=args.label)
    Path(args.out_tex).write_text(tex, encoding="utf-8")

    if args.out_csv:
        write_csv(args.out_csv, header, table_rows)

    # preview
    print("\n".join(["\t".join(header)] + ["\t".join(r) for r in table_rows[:15]]))


if __name__ == "__main__":
    main()
