from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def infer_vocab(path: Path) -> str:
    s = str(path).replace("\\", "/").lower()
    if "/clar_vocab2/" in s:
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
    # fallback: from folder name like results_layer20...
    s = str(path).lower()
    m2 = re.search(r"layer(\d+)", s)
    return int(m2.group(1)) if m2 else None

def infer_model(obj: Dict[str, Any], path: Path) -> str:
    run_info = obj.get("run_info", {})
    model = run_info.get("model", {}) if isinstance(run_info, dict) else {}
    name = ""
    if isinstance(model, dict):
        name = str(model.get("model_name", "")).lower()
    # fallback: file path
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
# Helpers: computing metrics
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
    Table columns (you can rename later):
      - clarif_rate
      - best_q_sim
      - nli
      - ambik_acc

    Preference:
      - use obj["metrics"] if present (your repo adds NLI fields there)
      - otherwise compute from per-example fields
    """
    examples = obj.get("examples", [])
    metrics = obj.get("metrics", {})

    # Always compute clarif_rate and best_q_sim from examples (consistent across all files)
    out = {
        "clarif_rate": clarif_rate(examples),
        "best_q_sim": mean_of_key(examples, "model_question_best_similarity"),
        # NLI column: mean per-example NLI similarity (closest analog to Best-Q sim)
        "nli": mean_of_key(examples, "model_question_best_nli_similarity"),
        "ambik_acc": ambik_acc_proxy(examples),
    }

    # If top-level metrics exist, optionally override accuracy and provide selection keys
    if isinstance(metrics, dict) and metrics:
        if isinstance(metrics.get("resolved_proxy_rate"), (int, float)):
            out["ambik_acc"] = float(metrics["resolved_proxy_rate"])

        # Add extra fields we can use for selecting “best run” per condition
        for k in ["overall_weighted_score", "overall_weighted_score_nli", "overall_similarity_nli", "resolved_proxy_rate_nli"]:
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                out[k] = float(v)

    return out


# ----------------------------
# Collect all runs
# ----------------------------

def collect_runs(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in results_dir.rglob("*.json"):
        obj = read_json(p)

        vocab = infer_vocab(p)
        layer = infer_layer(obj, p)
        model = infer_model(obj, p)
        condition = infer_condition(p)
        feat, strength = parse_feat_strength(obj, p)

        m = extract_table_metrics(obj)

        rows.append({
            "vocab": vocab,
            "layer": layer,
            "model": model,
            "condition": condition,
            "feature": feat,
            "strength": strength,
            "file": str(p),
            **m,
        })

    if not rows:
        raise RuntimeError(f"No JSON files found under {results_dir}")
    return pd.DataFrame(rows)


# ----------------------------
# Table building
# ----------------------------

def pick_best_run(df: pd.DataFrame, select_by: str) -> pd.DataFrame:
    """
    One row per (vocab, layer, model, condition).
    select_by can be: overall_weighted_score_nli | overall_weighted_score | ambik_acc | nli | best_q_sim
    """
    group_cols = ["vocab", "layer", "model", "condition"]

    if select_by not in df.columns:
        # fallback safely
        select_by = "ambik_acc"

    # idxmax per group
    idx = df.groupby(group_cols, dropna=False)[select_by].idxmax()
    best = df.loc[idx].copy().reset_index(drop=True)
    return best


def format_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    # Order conditions like in paper tables
    order = ["baseline", "steering", "steering (eval)", "steering (random)", "steering (variant2)", "unknown"]
    df = df.copy()
    df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)
    df = df.sort_values(["condition", "model"], na_position="last")

    # Keep only “paper columns”
    out = df[["condition", "model", "clarif_rate", "best_q_sim", "nli", "ambik_acc"]].copy()

    # formatting
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help="Path to results/ folder")
    ap.add_argument("--out_dir", type=str, default="paper_tables", help="Output folder")
    ap.add_argument("--select_by", type=str, default="overall_weighted_score_nli",
                    help="Metric to choose best run per condition (e.g., overall_weighted_score_nli, ambik_acc)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = collect_runs(results_dir)
    raw.to_csv(out_dir / "raw_runs.csv", index=False)

    best = pick_best_run(raw, select_by=args.select_by)
    best.to_csv(out_dir / "best_runs.csv", index=False)

    # Make one table per (vocab, layer)
    for (vocab, layer), sub in best.groupby(["vocab", "layer"], dropna=False):
        layer_str = "none" if pd.isna(layer) else str(int(layer))
        caption = f"AmbiK results for {vocab} (SAE layer {layer_str}). Best run per condition selected by {args.select_by}."
        label = f"tab:ambik_{vocab}_layer{layer_str}"

        latex_df = format_for_latex(sub)
        tex = to_latex_table(latex_df, caption=caption, label=label)

        (out_dir / f"table_ambik_{vocab}_layer{layer_str}.tex").write_text(tex, encoding="utf-8")

    print(f"Wrote tables to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
