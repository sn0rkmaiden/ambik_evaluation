#!/usr/bin/env python3
import argparse, json, os, sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from eval_metrics import compute_metrics_from_json 

def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # integers that can be missing → nullable Int64
    for c in ["feature", "n_examples", "seed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # booleans that can be missing → nullable boolean
    for c in ["steering_used", "compute_max_per_turn"]:
        if c in df.columns:
            df[c] = df[c].astype("boolean")

    # floats (keep NaN for undefined metrics)
    for c in [
        "strength", "max_act",
        "resolved_proxy_rate", "avg_num_questions",
        "necessity_precision", "necessity_recall",
        "resolved_dialog_rate", "overall_weighted_score",
        "nli_resolved_rate", "nli_overall_similarity", "overall_weighted_score_nli",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


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
        steering_cfg = (
            run_info.get("steering")      # old key, if you ever used it
            or run_info.get("steering_cfg")
            or {}
        )
    return model_name, steering_used, steering_cfg

def _round_or_none(x, nd=4):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None

def _row_from_file(path: str, threshold: float, nli_threshold: float | None) -> Dict[str, Any]:
    obj = _load_json(path)
    examples, run_info = _extract_examples_and_meta(obj, path)

    # compute metrics fresh
    metrics = compute_metrics_from_json(path, embed_threshold=threshold, nli_threshold=nli_threshold)

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
    row["dataset"] = os.path.basename(dataset_csv) if dataset_csv else None
    row["steering_used"] = steering_used
    row["sae_id"] = sae_id
    row["mode"] = mode
    row["seed"] = seed
    row["compute_max_per_turn"] = steer_per_turn
    row["n_examples"] = num_examples
    row["threshold"] = threshold
    row["nli_threshold"] = nli_threshold
    row["feature"] = steer_feature
    row["max_act"] = steer_max_act 
    row["strength"] = steer_strength
    row["resolved_proxy_rate"] = _round_or_none(metrics.get("resolved_proxy_rate"), 4)
    row["avg_num_questions"] = _round_or_none(metrics.get("avg_num_questions"), 3)
    row["necessity_precision"] = _round_or_none(metrics.get("necessity_precision"), 4)
    row["necessity_recall"] = _round_or_none(metrics.get("necessity_recall"), 4)
    row["resolved_dialog_rate"] = _round_or_none(metrics.get("resolved_dialog_rate"), 4)
    row["overall_weighted_score"] = _round_or_none(metrics.get("overall_weighted_score"), 4)
    row["nli_resolved_rate"] = _round_or_none(metrics.get("resolved_proxy_rate_nli"), 4)
    row["nli_overall_similarity"] = _round_or_none(metrics.get("overall_similarity_nli"), 4)
    row["overall_weighted_score_nli"] = _round_or_none(metrics.get("overall_weighted_score_nli"), 4)
    return row

def compare_runs_to_dataframe(paths: List[str], threshold: float = 0.75, nli_threshold: float | None = None) -> pd.DataFrame:
    """Build a pandas DataFrame summarizing multiple results JSONs."""
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            rows.append(_row_from_file(p, threshold=threshold, nli_threshold=nli_threshold))
        except Exception as e:
            print(f"[warn] Failed on {p}: {e}", file=sys.stderr)

    if not rows:
        raise RuntimeError("No rows produced. Check input files/schema.")

    cols = [
         "run_file", "model", "dataset", "sae_id", "mode", "seed", "compute_max_per_turn", "steering_used", "n_examples", "threshold", "nli_threshold",
           "feature", "strength",  "max_act", "resolved_proxy_rate", "nli_resolved_rate", "avg_num_questions", "necessity_precision", "necessity_recall", 
           "resolved_dialog_rate", "overall_weighted_score", "overall_weighted_score_nli", "nli_overall_similarity",   
    ]
    df = pd.DataFrame(rows)
    # ensure column order even if some fields are missing in a row
    df = df.reindex(columns=cols)
    df = coerce_dtypes(df)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        nargs="*",
        default=[],
        help="List of results JSON files (can be combined with --results_file)",
    )
    ap.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Optional text file with one path per line; only lines ending in .json are used",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for proxy resolution",
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Optional path to write a CSV",
    )
    ap.add_argument(
        "--print",
        action="store_true",
        help="Print DataFrame as a pretty table to stdout",
    )
    ap.add_argument(
        "--nli_threshold",
        type=float,
        default=None,
        help="Threshold for NLI-based resolution (defaults to --threshold)."
    )

    args = ap.parse_args()

    # Collect paths from CLI and optional file
    all_paths: List[str] = []

    # 1) Directly passed JSON files
    if args.results:
        all_paths.extend(args.results)

    # 2) From a text file (only *.json lines)
    if args.results_file:
        if not os.path.exists(args.results_file):
            print(f"[error] results_file not found: {args.results_file}", file=sys.stderr)
            sys.exit(1)
        with open(args.results_file, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue  # skip empty/comment lines
                if raw.endswith(".json"):
                    all_paths.append(raw)
                else:
                    # Explicitly ignore non-json lines
                    print(
                        f"[warn] Skipping non-JSON entry from results_file: {raw}",
                        file=sys.stderr,
                    )

    if not all_paths:
        print(
            "[error] No result files provided. Use --results and/or --results_file.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = compare_runs_to_dataframe(all_paths, threshold=args.threshold, nli_threshold=args.nli_threshold)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[info] wrote CSV -> {args.out_csv}")

    # For terminals (non-notebook), print a readable table
    if args.print or not sys.stdout.isatty():
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()

