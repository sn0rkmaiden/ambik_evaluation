#!/usr/bin/env python3
import argparse, json, os, sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from eval_metrics import compute_metrics_from_json  # recomputes similarity with best_match_score


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _extract_examples_and_meta(obj: Any, path: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (examples, run_info_or_None).
    Expects the wrapped schema: {"run_info": {...}, "examples": [...]}
    """
    run_info = None
    if isinstance(obj, dict) and "examples" in obj:
        examples = obj["examples"]
        run_info = obj.get("run_info")
    else:
        raise ValueError(f"Unrecognized results schema in {path}")

    # If still no run_info, try to infer from the first example
    if run_info is None and examples:
        ex0 = examples[0]
        model_used = ex0.get("model_used") if isinstance(ex0.get("model_used"), dict) else None
        run_info = {
            "model": model_used or {},
            "steering_used": ex0.get("steering_used", False),
            "steering": ex0.get("steering") if ex0.get("steering_used") else None,
            "dataset_csv": None,
            "mode": None,
            "seed": None,
            "num_examples": len(examples),
            "output_json": path,
        }
    return examples, run_info

def _model_and_steer_from_runinfo(run_info: Optional[Dict[str, Any]]) -> Tuple[str, bool, Dict[str, Any]]:
    model_name = "unknown"
    steering_used = False
    steering_cfg: Dict[str, Any] = {}
    if run_info:
        m = run_info.get("model") or {}
        model_name = m.get("model_name") or m.get("wrapper_class") or "unknown"
        steering_used = bool(run_info.get("steering_used", False))
        steering_cfg = run_info.get("steering") or {}
    return model_name, steering_used, steering_cfg

def _round_or_none(x, nd=4):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None

def _row_from_file(path: str, threshold: float) -> Dict[str, Any]:
    obj = _load_json(path)
    examples, run_info = _extract_examples_and_meta(obj, path)

    # compute metrics fresh
    metrics = compute_metrics_from_json(path, embed_threshold=threshold)

    model_name, steering_used, steering_cfg = _model_and_steer_from_runinfo(run_info)
    num_examples = (run_info or {}).get("num_examples") or len(examples)
    mode = (run_info or {}).get("mode")
    seed = (run_info or {}).get("seed")
    dataset_csv = (run_info or {}).get("dataset_csv")

    # Pick a few steering fields if present
    steer_feature = steering_cfg.get("feature")
    steer_strength = steering_cfg.get("strength")
    steer_per_turn = steering_cfg.get("compute_max_per_turn")
    steer_max_act = steering_cfg.get("max_act")
    sae_id = steering_cfg.get("sae_id")

    # Flatten key metrics for a nice table row
    row = OrderedDict()
    row["run_file"] = os.path.basename(path)
    row["model"] = model_name
    row["steering_used"] = steering_used
    row["feature"] = steer_feature
    row["strength"] = steer_strength
    row["compute_max_per_turn"] = steer_per_turn
    row["max_act"] = steer_max_act
    row["sae_id"] = sae_id
    row["mode"] = mode
    row["seed"] = seed
    row["dataset"] = os.path.basename(dataset_csv) if dataset_csv else None
    row["n_examples"] = num_examples
    row["resolved_proxy_rate"] = _round_or_none(metrics.get("resolved_proxy_rate"), 4)
    row["avg_num_questions"] = _round_or_none(metrics.get("avg_num_questions"), 3)
    row["necessity_precision"] = _round_or_none(metrics.get("necessity_precision"), 4)
    row["necessity_recall"] = _round_or_none(metrics.get("necessity_recall"), 4)
    row["resolved_dialog_rate"] = _round_or_none(metrics.get("resolved_dialog_rate"), 4)
    row["overall_weighted_score"] = _round_or_none(metrics.get("overall_weighted_score"), 4)
    return row

def compare_runs_to_dataframe(paths: List[str], threshold: float = 0.75) -> pd.DataFrame:
    """Build a pandas DataFrame summarizing multiple results JSONs."""
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            rows.append(_row_from_file(p, threshold=threshold))
        except Exception as e:
            print(f"[warn] Failed on {p}: {e}", file=sys.stderr)

    if not rows:
        raise RuntimeError("No rows produced. Check input files/schema.")

    # Stable column order
    cols = [
        "run_file","model","steering_used","feature","strength","compute_max_per_turn","max_act","sae_id",
        "mode","seed","dataset","n_examples",
        "resolved_proxy_rate","avg_num_questions","necessity_precision","necessity_recall",
        "resolved_dialog_rate","overall_weighted_score"
    ]
    df = pd.DataFrame(rows)
    # ensure column order even if some fields are missing in a row
    df = df.reindex(columns=cols)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="Paths to wrapped results JSON files ({run_info, examples})")
    ap.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for proxy resolution")
    ap.add_argument("--out_csv", default=None, help="Optional path to write a CSV")
    ap.add_argument("--print", action="store_true", help="Print DataFrame as a pretty table to stdout")
    args = ap.parse_args()

    df = compare_runs_to_dataframe(args.results, threshold=args.threshold)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[info] wrote CSV -> {args.out_csv}")

    # For terminals (non-notebook), print a readable table
    if args.print or not sys.stdout.isatty():
        # widen columns a bit for readability
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.to_string(index=False))

if __name__ == "__main__":
    main()
