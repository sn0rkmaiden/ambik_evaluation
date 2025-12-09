"""
Offline metrics for clarifying questions.

Usage (example):
    python all_metrics.py \
        results/ambik_eval_baseline.json \
        results/ambik_eval_steered.json \
        --q_thresholds 0.75 0.65 \
        --qa_threshold 0.7

    python all_metrics.py \
        --dir results \
        --pattern "ambik_eval_*.json" \
        --q_thresholds 0.75 0.65 \
        --qa_threshold 0.7

"""

import argparse
import json
import glob
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from text_matching import best_match_score  # your existing lexical+embedding sim


# ---------- loading ----------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_examples(obj: Any) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - list of examples, or
      - {"run_info": {...}, "examples": [...]}
    Returns the examples list.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "examples" in obj:
        return obj["examples"]
    raise ValueError("Unrecognized results schema (expected list or {run_info, examples}).")

def _safe_list(xs):
    if xs is None:
        return []
    if isinstance(xs, list):
        return xs
    # some older runs might have a single string instead of list
    return [xs]


def per_example_scores(ex: Dict[str, Any]) -> Tuple[float, float, bool]:
    """
    Compute S_Q_max and S_QA_max for a single example.

    Returns:
        (S_Q_max, S_QA_max, has_question)
    """
    gold_q = (ex.get("gold_question") or "").strip()
    gold_a = (ex.get("gold_answer") or "").strip()
    model_qs = _safe_list(ex.get("model_questions"))

    if not model_qs:
        return 0.0, 0.0, False

    # best similarity question and gold question
    best_sq = 0.0
    # best similarity (Q_model, A_gold) and (Q_gold, A_gold)
    best_sqa = 0.0

    pair_gold = f"Q: {gold_q}\nA: {gold_a}" if (gold_q or gold_a) else ""

    for q in model_qs:
        q = (q or "").strip()
        if not q:
            continue

        # question-only similarity
        try:
            sq = float(best_match_score(q, gold_q))
        except Exception:
            sq = 0.0

        if sq > best_sq:
            best_sq = sq

        # question + answer similarity
        if pair_gold:
            pair_model = f"Q: {q}\nA: {gold_a}"
            try:
                sqa = float(best_match_score(pair_model, pair_gold))
            except Exception:
                sqa = 0.0
            if sqa > best_sqa:
                best_sqa = sqa

    return best_sq, best_sqa, True


def compute_answer_aware_metrics(
    path: str,
    q_thresholds: List[float],
    qa_threshold: float,
) -> Dict[str, Any]:
    """
    Off-line metrics for a single results JSON.

    q_thresholds: thresholds for question-only proxy (e.g. [0.75, 0.65])
    qa_threshold: threshold for answer-aware compatibility (e.g. 0.7)
    """
    obj = _load_json(path)
    examples = _extract_examples(obj)

    n = len(examples)
    if n == 0:
        raise ValueError(f"No examples in {path}")

    sum_sq = 0.0
    sum_sqa = 0.0
    count_with_q = 0
    count_with_q_and_ans = 0

    resolved_counts = {thr: 0 for thr in q_thresholds}
    answer_compatible_count = 0

    per_cat = defaultdict(lambda: {"n": 0, "sum_sq": 0.0, "sum_sqa": 0.0, "has_q": 0})

    for ex in examples:
        amb_type = ex.get("ambiguity_type", "unknown")
        S_Q, S_QA, has_q = per_example_scores(ex)

        if has_q:
            count_with_q += 1
            sum_sq += S_Q
            per_cat[amb_type]["sum_sq"] += S_Q
            per_cat[amb_type]["has_q"] += 1

        if S_QA > 0.0:
            count_with_q_and_ans += 1
            sum_sqa += S_QA
            per_cat[amb_type]["sum_sqa"] += S_QA

        per_cat[amb_type]["n"] += 1

        # resolved_proxy at multiple thresholds
        for thr in q_thresholds:
            if S_Q >= thr:
                resolved_counts[thr] += 1

        # answer-compatible flag
        if S_QA >= qa_threshold:
            answer_compatible_count += 1

    # aggregate
    metrics: Dict[str, Any] = {}
    metrics["n_examples"] = n
    metrics["q_thresholds"] = q_thresholds
    metrics["qa_threshold"] = qa_threshold

    # overall averages (only where a question was asked / S_QA defined)
    metrics["mean_S_Q_over_asked"] = (sum_sq / count_with_q) if count_with_q > 0 else None
    metrics["mean_S_QA_over_QA"] = (sum_sqa / count_with_q_and_ans) if count_with_q_and_ans > 0 else None

    # resolved_proxy rates
    metrics["resolved_proxy_rates"] = {
        str(thr): resolved_counts[thr] / n for thr in q_thresholds
    }

    # answer-compatibility rate (over all examples)
    metrics["answer_compatible_rate"] = answer_compatible_count / n

    # per-category averages
    per_cat_out = {}
    for cat, stats in per_cat.items():
        m = {}
        m["n"] = stats["n"]
        if stats["has_q"] > 0:
            m["mean_S_Q"] = stats["sum_sq"] / stats["has_q"]
        else:
            m["mean_S_Q"] = None
        if stats["sum_sqa"] > 0 and stats["n"] > 0:
            # average over examples where S_QA > 0
            m["mean_S_QA_raw"] = stats["sum_sqa"] / stats["n"]
        else:
            m["mean_S_QA_raw"] = None
        per_cat_out[cat] = m

    metrics["per_category"] = per_cat_out

    return metrics

def main():
    ap = argparse.ArgumentParser()
    # make results optional: we can also use --dir
    ap.add_argument(
        "results",
        nargs="*",
        help="Paths to results JSON files (list or {run_info, examples})",
    )
    ap.add_argument(
        "--dir",
        default=None,
        help="Directory to scan for JSON files (optional).",
    )
    ap.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern inside --dir (default: *.json).",
    )
    ap.add_argument(
        "--q_thresholds",
        type=float,
        nargs="+",
        default=[0.75, 0.65],
        help="Thresholds for question-only proxy resolution",
    )
    ap.add_argument(
        "--qa_threshold",
        type=float,
        default=0.7,
        help="Threshold for answer-aware compatibility",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON metrics",
    )
    args = ap.parse_args()

    # collect all file paths
    paths = list(args.results)
    if args.dir is not None:
        pattern = os.path.join(args.dir, args.pattern)
        dir_paths = sorted(glob.glob(pattern))
        paths.extend(dir_paths)

    # deduplicate, preserve order
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if not unique_paths:
        print("No input files: provide positional paths or --dir/--pattern.", file=sys.stderr)
        sys.exit(1)

    for path in unique_paths:
        try:
            m = compute_answer_aware_metrics(path, args.q_thresholds, args.qa_threshold)
        except Exception as e:
            print(f"[warn] Failed on {path}: {e}", file=sys.stderr)
            continue

        print(f"\n=== {os.path.basename(path)} ===")
        if args.pretty:
            print(json.dumps(m, indent=2))
        else:
            print(f"n_examples: {m['n_examples']}")
            print("resolved_proxy_rates:")
            for thr_str, rate in m["resolved_proxy_rates"].items():
                print(f"  τ={thr_str}: {rate:.3f}")
            print(f"answer_compatible_rate (qa τ={m['qa_threshold']}): {m['answer_compatible_rate']:.3f}")
            print(f"mean_S_Q_over_asked: {m['mean_S_Q_over_asked']}")
            print(f"mean_S_QA_over_QA: {m['mean_S_QA_over_QA']}")


if __name__ == "__main__":
    main()

