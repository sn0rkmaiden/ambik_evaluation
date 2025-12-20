#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ----------------------------
# 0) Import your repo metric code
# ----------------------------

def import_repo_metrics(repo_root: Path):
    """
    Ensure repo_root is on sys.path, then import your metric function.
    """
    sys.path.insert(0, str(repo_root.resolve()))
    try:
        from eval_metrics import compute_metrics_from_json  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import eval_metrics.compute_metrics_from_json.\n"
            f"Run this script from your repo root or pass --repo_root.\n"
            f"Import error: {e}"
        )
    return compute_metrics_from_json


# ----------------------------
# 1) Reading + metadata inference
# ----------------------------

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def has_metrics(obj: Dict[str, Any]) -> bool:
    m = obj.get("metrics")
    return isinstance(m, dict) and len(m) > 0

def infer_vocab(path: Path) -> str:
    s = str(path).replace("\\", "/").lower()
    if "/clar_vocab2/" in s or "/clar_vocab/" in s:
        return "clar_vocab"
    if "/question_vocab/" in s:
        return "question_vocab"
    return "unknown"

def infer_layer(obj: Dict[str, Any], path: Path) -> Optional[int]:
    run_info = obj.get("run_info", {})
    steering_cfg = run_info.get("steering_cfg", {}) if isinstance(run_info, dict) else {}
    sae_id = steering_cfg.get("sae_id") if isinstance(steering_cfg, dict) else None
    if isinstance(sae_id, str):
        m = re.search(r"blocks\.(\d+)\.", sae_id)
        if m:
            return int(m.group(1))

    # fallback to path hints
    s = str(path).lower()
    m2 = re.search(r"layer(\d+)", s)
    return int(m2.group(1)) if m2 else None

def infer_model(obj: Dict[str, Any], path: Path) -> str:
    run_info = obj.get("run_info", {})
    model = run_info.get("model", {}) if isinstance(run_info, dict) else {}
    name = ""
    if isinstance(model, dict):
        name = str(model.get("model_name", "")).lower()
    if not name:
        name = str(path).lower()

    if "9b" in name:
        return "9B"
    if "2b" in name:
        return "2B"
    return "unknown"

def infer_condition(path: Path) -> str:
    fn = path.name.lower()
    if "baseline" in fn:
        return "baseline"
    if "steered_rand" in fn or "rand" in fn:
        return "steering (random)"
    if "steered_eval" in fn or "eval" in fn:
        return "steering (eval)"
    if "steered2" in fn:
        return "steering (variant2)"
    if "steered" in fn:
        return "steering"
    return "unknown"

def parse_feat_strength(obj: Dict[str, Any], path: Path) -> Tuple[Optional[int], Optional[float]]:
    run_info = obj.get("run_info", {})
    model = run_info.get("model", {}) if isinstance(run_info, dict) else {}
    feat = None
    strength = None
    if isinstance(model, dict):
        if isinstance(model.get("steering_feature"), (int, float)):
            feat = int(model["steering_feature"])
        if isinstance(model.get("steering_strength"), (int, float)):
            strength = float(model["steering_strength"])

    fn = path.name
    m1 = re.search(r"feat(\d+)", fn)
    m2 = re.search(r"str([0-9]+(?:\.[0-9]+)?)", fn)
    if feat is None and m1:
        feat = int(m1.group(1))
    if strength is None and m2:
        strength = float(m2.group(1))
    return feat, strength


# ----------------------------
# 2) Metric extraction for table rows
# ----------------------------

def mean_of_key(examples: List[Dict[str, Any]], key: str) -> float:
    vals = []
    for ex in examples:
        v = ex.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else float("nan")

def clarif_rate(examples: List[Dict[str, Any]]) -> float:
    vals = []
    for ex in examples:
        nq = ex.get("num_questions")
        if isinstance(nq, (int, float)):
            vals.append(1.0 if float(nq) > 0 else 0.0)
    return sum(vals) / len(vals) if vals else float("nan")

def ambik_acc_proxy(examples: List[Dict[str, Any]]) -> float:
    vals = []
    for ex in examples:
        rp = ex.get("resolved_proxy")
        if isinstance(rp, bool):
            vals.append(1.0 if rp else 0.0)
        elif isinstance(rp, (int, float)):
            vals.append(float(rp))
    return sum(vals) / len(vals) if vals else float("nan")

def extract_table_metrics(obj: Dict[str, Any]) -> Dict[str, float]:
    """
    Uses (a) repo-computed metrics if present and (b) example fields for the
    similarity columns so they match what you see per example.
    """
    examples = obj.get("examples", [])
    metrics = obj.get("metrics", {})

    out: Dict[str, float] = {
        "clarif_rate": clarif_rate(examples),
        "best_q_sim": mean_of_key(examples, "model_question_best_similarity"),
        "nli": mean_of_key(examples, "model_question_best_nli_similarity"),
        "ambik_acc": ambik_acc_proxy(examples),
    }

    if isinstance(metrics, dict) and metrics:
        # if your repo provides these, keep them for selection and/or accuracy
        for k in [
            "resolved_proxy_rate",
            "overall_weighted_score",
            "overall_similarity_nli",
            "resolved_proxy_rate_nli",
            "overall_weighted_score_nli",
        ]:
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                out[k] = float(v)

        # accuracy: prefer repo aggregate if present
        if isinstance(metrics.get("resolved_proxy_rate"), (int, float)):
            out["ambik_acc"] = float(metrics["resolved_proxy_rate"])

        # optional: if you want NLI column to be the repo aggregate:
        # if isinstance(metrics.get("overall_similarity_nli"), (int, float)):
        #     out["nli"] = float(metrics["overall_similarity_nli"])

    return out


# ----------------------------
# 3) Recompute missing metrics using repo code
# ----------------------------

def recompute_metrics_if_needed(
    compute_metrics_from_json,
    in_path: Path,
    out_path: Path,
    *,
    embed_threshold: float,
    nli_threshold: Optional[float],
    overwrite: bool,
) -> Tuple[Dict[str, Any], bool]:
    """
    Returns (obj, changed). Writes updated JSON if changed.
    """
    obj = read_json(in_path)

    if has_metrics(obj) and not overwrite:
        # still write it through if out_path differs
        if out_path != in_path:
            write_json(out_path, obj)
        return obj, False

    # call YOUR repo code
    metrics = compute_metrics_from_json(
        obj,
        embed_threshold=embed_threshold,
        nli_threshold=nli_threshold,
    )
    obj["metrics"] = metrics
    write_json(out_path, obj)
    return obj, True


# ----------------------------
# 4) Build tables
# ----------------------------

def pick_best_run(df: pd.DataFrame, select_by: str) -> pd.DataFrame:
    """
    One row per (vocab, layer, model, condition).
    Robust when select_by is missing/NaN for some groups (falls back to ambik_acc).
    """
    group_cols = ["vocab", "layer", "model", "condition"]

    if select_by not in df.columns:
        select_by = "ambik_acc"

    fallback = "ambik_acc" if "ambik_acc" in df.columns else None

    idx = df.groupby(group_cols, dropna=False)[select_by].idxmax()
    idx_valid = idx.dropna().astype(int)
    best = df.loc[idx_valid].copy()

    missing_groups = set(idx.index) - set(best.set_index(group_cols).index)
    if missing_groups:
        df_missing = df.set_index(group_cols).loc[list(missing_groups)].reset_index()
        if fallback and fallback in df_missing.columns:
            idx2 = df_missing.groupby(group_cols, dropna=False)[fallback].idxmax()
            idx2_valid = idx2.dropna().astype(int)
            best2 = df_missing.loc[idx2_valid].copy()
        else:
            best2 = df_missing.groupby(group_cols, dropna=False).head(1).copy()
        best = pd.concat([best, best2], ignore_index=True)

    return best.sort_values(group_cols, na_position="last").reset_index(drop=True)

def format_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    order = ["baseline", "steering", "steering (eval)", "steering (random)", "steering (variant2)", "unknown"]
    df = df.copy()
    df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)
    df = df.sort_values(["condition", "model"], na_position="last")

    out = df[["condition", "model", "clarif_rate", "best_q_sim", "nli", "ambik_acc"]].copy()
    for c in ["clarif_rate", "best_q_sim", "nli", "ambik_acc"]:
        out[c] = out[c].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    return out

def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"{body}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


# ----------------------------
# 5) Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="Path to ambik_evaluation repo root (contains eval_metrics.py)")
    ap.add_argument("--results_dir", type=str, required=True, help="Input results directory to scan recursively")
    ap.add_argument("--out_results_dir", type=str, required=True,
                    help="Where to write JSONs with metrics (can be same as results_dir for in-place)")

    ap.add_argument("--embed_threshold", type=float, default=0.75, help="Passed to compute_metrics_from_json")
    ap.add_argument("--nli_threshold", type=float, default=None, help="Passed to compute_metrics_from_json (None uses repo default)")

    ap.add_argument("--overwrite_existing_metrics", action="store_true",
                    help="Recompute even if obj['metrics'] exists")

    ap.add_argument("--tables_out_dir", type=str, default="paper_tables", help="Where to write .tex tables and CSVs")
    ap.add_argument("--select_by", type=str, default="overall_weighted_score_nli",
                    help="Metric used to pick best run per (vocab,layer,model,condition)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    compute_metrics_from_json = import_repo_metrics(repo_root)

    results_dir = Path(args.results_dir)
    out_results_dir = Path(args.out_results_dir)
    out_results_dir.mkdir(parents=True, exist_ok=True)

    # 5.1 Recompute metrics (only missing by default)
    total = 0
    changed = 0

    updated_rows: List[Dict[str, Any]] = []

    json_files = list(results_dir.rglob("*.json"))

    for in_path in tqdm(
        json_files,
        desc="Recomputing metrics & collecting rows",
        unit="file",
    ):
        total += 1
        rel = in_path.relative_to(results_dir)
        out_path = out_results_dir / rel

        obj, did_change = recompute_metrics_if_needed(
            compute_metrics_from_json,
            in_path,
            out_path,
            embed_threshold=args.embed_threshold,
            nli_threshold=args.nli_threshold,
            overwrite=args.overwrite_existing_metrics,
        )
        if did_change:
            changed += 1

        vocab = infer_vocab(out_path)
        layer = infer_layer(obj, out_path)
        model = infer_model(obj, out_path)
        condition = infer_condition(out_path)
        feat, strength = parse_feat_strength(obj, out_path)
        mets = extract_table_metrics(obj)

        updated_rows.append({
            "vocab": vocab,
            "layer": layer,
            "model": model,
            "condition": condition,
            "feature": feat,
            "strength": strength,
            "file": str(out_path),
            **mets,
        })


    if not updated_rows:
        raise RuntimeError(f"No JSON files found under {results_dir}")

    print(f"[metrics] scanned: {total} | recomputed/written: {changed} | skipped-with-metrics: {total - changed}")

    # 5.3 Build tables
    tables_out = Path(args.tables_out_dir)
    tables_out.mkdir(parents=True, exist_ok=True)

    raw = pd.DataFrame(updated_rows)
    raw.to_csv(tables_out / "raw_runs.csv", index=False)

    best = pick_best_run(raw, select_by=args.select_by)
    best.to_csv(tables_out / "best_runs.csv", index=False)

    # One .tex per (vocab, layer)
    for (vocab, layer), sub in best.groupby(["vocab", "layer"], dropna=False):
        layer_str = "none" if pd.isna(layer) else str(int(layer))
        caption = f"AmbiK results for {vocab} (SAE layer {layer_str}). Best run per condition selected by {args.select_by}."
        label = f"tab:ambik_{vocab}_layer{layer_str}"

        latex_df = format_for_latex(sub)
        tex = to_latex_table(latex_df, caption=caption, label=label)

        (tables_out / f"table_ambik_{vocab}_layer{layer_str}.tex").write_text(tex, encoding="utf-8")

    print(f"[tables] wrote: {tables_out.resolve()}")
    print(f"[results_with_metrics] wrote: {out_results_dir.resolve()}")


if __name__ == "__main__":
    main()
