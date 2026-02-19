#!/usr/bin/env python3
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

_STRENGTH_RE = re.compile(r"(?:^|[_-])str(?P<val>-?\d+(?:\.\d+)?)", re.IGNORECASE)
_MODEL_RE = re.compile(r"(?:^|[_/])(?P<size>2b|9b)(?:[^a-z0-9]|$)", re.IGNORECASE)
_CLUSTER_RE = re.compile(r"(?:^|[_-])cluster(?P<cid>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_LAYER_RE = re.compile(r"(?:^|[_-])l(?P<layer>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_NCLUST_RE = re.compile(r"(?:^|[_-])nclust(?P<n>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_THR_RE = re.compile(r"(?:^|[_-])thr(?P<thr>\d+(?:\.\d+)?)", re.IGNORECASE)

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

def infer_layer_from_filename(path: str) -> Optional[int]:
    m = _LAYER_RE.search(Path(path).name)
    return int(m.group("layer")) if m else None

def infer_nclust_from_filename(path: str) -> Optional[int]:
    m = _NCLUST_RE.search(Path(path).name)
    return int(m.group("n")) if m else None

def infer_thr_from_filename(path: str) -> Optional[float]:
    m = _THR_RE.search(Path(path).name)
    return _safe_float(m.group("thr")) if m else None

def extract_strength(run: Dict[str, Any], path: str) -> Optional[float]:
    ri = run.get("run_info") or {}
    scfg = ri.get("steering_cfg")
    if isinstance(scfg, dict):
        v = _safe_float(scfg.get("strength"))
        if v is not None:
            return v
    model = ri.get("model")
    if isinstance(model, dict):
        v = _safe_float(model.get("steering_strength"))
        if v is not None:
            return v
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

@dataclass
class ResultRow:
    model: str
    cluster_id: Optional[int]
    layer: Optional[int]
    nclust: Optional[int]
    thr: Optional[float]
    strength: Optional[float]
    metric: float
    path: str

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

def load_results(
    results_path: str,
    metric_field: str,
    only_cluster_runs: bool,
    filter_model: Optional[str],
    filter_layer: Optional[int],
    filter_nclust: Optional[int],
    filter_thr: Optional[float],
) -> List[ResultRow]:
    rows: List[ResultRow] = []
    for vpath, blob in iter_json_blobs(results_path):
        try:
            run = json.loads(blob.decode("utf-8"))
        except Exception:
            continue

        model = infer_model_size_from_path_or_json(vpath, run)
        cid = infer_cluster_id_from_filename(vpath)
        layer = infer_layer_from_filename(vpath)
        ncl = infer_nclust_from_filename(vpath)
        thr = infer_thr_from_filename(vpath)

        base = is_baseline_run(run, vpath)
        strength = None if base else extract_strength(run, vpath)

        if filter_model and model != filter_model:
            continue

        if cid is not None:
            if filter_layer is not None and layer != filter_layer:
                continue
            if filter_nclust is not None and ncl != filter_nclust:
                continue
            if filter_thr is not None and (thr is None or abs(thr - filter_thr) > 1e-9):
                continue

        if only_cluster_runs and (cid is None) and (not base):
            continue

        rows.append(ResultRow(
            model=model, cluster_id=cid, layer=layer, nclust=ncl, thr=thr,
            strength=strength, metric=compute_metric(run, metric_field), path=vpath
        ))
    return rows

def fmt_strength(s: float) -> str:
    return f"Str{int(s)}" if abs(s - round(s)) < 1e-9 else f"Str{s:g}"

def fmt_val(v: float) -> str:
    return "-" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.3f}"

def cluster_display(
    cid: int,
    cluster_id_offset: int,
    cluster_prefix: str,
) -> str:
    shown = cid + cluster_id_offset
    return f"{cluster_prefix}{shown}" if cluster_prefix else str(shown)

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
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--metric_field", default="resolved_proxy")
    ap.add_argument("--agg", choices=["mean", "max", "median"], default="mean")
    ap.add_argument("--strengths", default="1,3,5,10")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:cluster_strength_dynamics")

    ap.add_argument("--only_cluster_runs", action="store_true")
    ap.set_defaults(only_cluster_runs=True)

    ap.add_argument("--filter_model", choices=["2B", "9B", "UNK"], default=None)
    ap.add_argument("--filter_layer", type=int, default=None)
    ap.add_argument("--filter_nclust", type=int, default=None)
    ap.add_argument("--filter_thr", type=float, default=None)

    # NEW: control how cluster IDs are printed
    ap.add_argument("--cluster_id_offset", type=int, default=0,
                    help="Add this offset to the cluster id read from the filename. Default 0 (no change).")
    ap.add_argument("--cluster_prefix", type=str, default="cluster",
                    help="Prefix for cluster display. Use '' to show only the number.")

    args = ap.parse_args()

    strengths = parse_strengths(args.strengths)
    rows = load_results(
        args.results,
        metric_field=args.metric_field,
        only_cluster_runs=args.only_cluster_runs,
        filter_model=args.filter_model,
        filter_layer=args.filter_layer,
        filter_nclust=args.filter_nclust,
        filter_thr=args.filter_thr,
    )

    # Baseline per model
    baseline_by_model: Dict[str, List[float]] = {}
    for r in rows:
        if r.strength is None:
            baseline_by_model.setdefault(r.model, []).append(r.metric)
    baseline_val_by_model = {m: agg(vs, args.agg) for m, vs in baseline_by_model.items()}

    # Group by (Model, Layer, Cluster, Strength)
    grouped: Dict[Tuple[str, int, int, float], List[float]] = {}
    for r in rows:
        if r.cluster_id is None or r.strength is None or r.layer is None:
            continue
        grouped.setdefault((r.model, r.layer, r.cluster_id, float(r.strength)), []).append(r.metric)

    keys = sorted({(m, layer, cid) for (m, layer, cid, _s) in grouped.keys()},
                  key=lambda x: (x[0], x[1], x[2]))

    header = ["Model", "Base", "Layer", "Cluster"] + [fmt_strength(s) for s in strengths]
    table_rows: List[List[str]] = []

    for (model, layer, cid) in keys:
        base = baseline_val_by_model.get(model, float("nan"))
        cluster_str = cluster_display(cid, args.cluster_id_offset, args.cluster_prefix)
        row = [model, fmt_val(base), str(layer), cluster_str]
        for st in strengths:
            vals = grouped.get((model, layer, cid, float(st)), [])
            row.append(fmt_val(agg(vals, args.agg)))
        table_rows.append(row)

    caption = args.caption or f"AmbiK {args.metric_field} vs steering strength (cluster steering)"
    tex = to_latex(header, table_rows, caption=caption, label=args.label)
    Path(args.out_tex).write_text(tex, encoding="utf-8")

    if args.out_csv:
        write_csv(args.out_csv, header, table_rows)

    print("\n".join(["\t".join(header)] + ["\t".join(r) for r in table_rows[:10]]))

if __name__ == "__main__":
    main()
