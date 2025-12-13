import json
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Any
from text_matching import best_match_score, nli_question_similarity

def _load_results(obj_or_path):
    """Accept a path or already-loaded JSON. Return the examples list."""
    if isinstance(obj_or_path, str):
        with open(obj_or_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = obj_or_path
    # list OR {"run_info": {}, "examples": []}
    return obj["examples"] if isinstance(obj, dict) and "examples" in obj else obj

def compute_metrics_from_json(
    json_path,
    embed_threshold: float = 0.75,
    brevity_max: int = 1,
    nli_threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Recomputes per-example similarity with best_match_score and derives metrics.

    embed_threshold: score cutoff to consider the proxy 'resolved' (embedding metric).
    brevity_max:     how many questions count as 'brief' for brevity_score.
    nli_threshold:   cutoff for NLI-based resolution; defaults to embed_threshold if None.
    """
    if nli_threshold is None:
        nli_threshold = embed_threshold

    data: List[Dict[str, Any]] = _load_results(json_path)

    counts = Counter()
    sim_sums = defaultdict(float)       # sum of best sims (only where nq > 0)
    sim_counts = defaultdict(int)       # count of rows contributing to sim_sums

    nli_sim_sums = defaultdict(float)      
    nli_sim_counts = defaultdict(int)

    num_questions_hist = Counter()

    resolved_proxy_count = 0
    resolved_dialog_count = 0
    nli_resolved_count = 0
    dialog_total = 0

    # “Necessity” bookkeeping (only 'preferences' should ask)
    necessity_tp = necessity_fp = necessity_fn = 0

    total = len(data)

    for ex in data:
        cat = ex.get("ambiguity_type", "unknown")
        counts[cat] += 1

        model_questions = ex.get("model_questions") or []
        if not isinstance(model_questions, list):
            model_questions = []

        gold_q = ex.get("gold_question", "") or ""

        # --- Recompute best similarity using best_match_score ---
        best_sim = 0.0
        for q in model_questions:
            s = best_match_score(q or "", gold_q or "", threshold=embed_threshold)
            if s > best_sim:
                best_sim = s

        ex["model_question_best_similarity"] = best_sim
        ex["resolved_proxy"] = bool(best_sim >= float(embed_threshold))

        best_nli_sim = 0.0
        for q in model_questions:
            s_nli = nli_question_similarity(q or "", gold_q or "")
            if s_nli > best_nli_sim:
                best_nli_sim = s_nli

        ex["model_question_best_nli_similarity"] = best_nli_sim
        ex["resolved_nli"] = bool(best_nli_sim >= float(nli_threshold))


        nq = len(model_questions)
        num_questions_hist[nq] += 1

        if nq > 0:
            sim_sums[cat] += best_sim
            sim_counts[cat] += 1

            nli_sim_sums[cat] += best_nli_sim       
            nli_sim_counts[cat] += 1 

        if ex["resolved_proxy"]:
            resolved_proxy_count += 1

        if ex["resolved_nli"]:                     
            nli_resolved_count += 1

        # Dialog resolution if present
        dialog = ex.get("dialog")
        if isinstance(dialog, dict) and dialog.get("resolved_dialog") is not None:
            dialog_total += 1
            if dialog.get("resolved_dialog"):
                resolved_dialog_count += 1

        # Only 'preferences' should ask a question
        asked = 1 if nq > 0 else 0
        if cat == "preferences" and asked == 1:
            necessity_tp += 1
        if cat != "preferences" and asked == 1:
            necessity_fp += 1
        if cat == "preferences" and asked == 0:
            necessity_fn += 1

    # Aggregation
    per_cat_sim = {
        c: (sim_sums[c] / sim_counts[c]) if sim_counts[c] > 0 else None
        for c in counts.keys()
    }

    per_cat_sim_nli = {                            
        c: (nli_sim_sums[c] / nli_sim_counts[c]) if nli_sim_counts[c] > 0 else None
        for c in counts.keys()
    }

    avg_num_questions = sum(k * v for k, v in num_questions_hist.items()) / max(1, total)
    necessity_precision = (
        necessity_tp / (necessity_tp + necessity_fp) if (necessity_tp + necessity_fp) > 0 else None
    )
    necessity_recall = (
        necessity_tp / (necessity_tp + necessity_fn) if (necessity_tp + necessity_fn) > 0 else None
    )
    resolved_proxy_rate = resolved_proxy_count / total if total > 0 else 0.0
    resolved_dialog_rate = resolved_dialog_count / dialog_total if dialog_total > 0 else None
    nli_resolved_rate = nli_resolved_count / total if total > 0 else 0.0 

    # overall weighted score
    # similarity only over rows where the model asked >=1 question
    total_q_rows = sum(1 for ex in data if len(ex.get("model_questions") or []) > 0)
    overall_similarity = (
        (sum(ex.get("model_question_best_similarity", 0.0) for ex in data if len(ex.get("model_questions") or []) > 0)
         / max(1, total_q_rows))
    )

    overall_similarity_nli = (                          
        (sum(ex.get("model_question_best_nli_similarity", 0.0) for ex in data
             if len(ex.get("model_questions") or []) > 0)
         / max(1, total_q_rows))
    )

    brevity_score = sum(1 for ex in data if len(ex.get("model_questions") or []) <= brevity_max) / max(1, total)

    # necessity_score == recall on the 'preferences' class
    denom_pref = max(1, sum(1 for ex in data if ex.get("ambiguity_type") == "preferences"))
    necessity_score = necessity_tp / denom_pref

    overall_weighted = 0.5 * necessity_score + 0.4 * overall_similarity + 0.1 * brevity_score

    overall_weighted_nli = (                            
        0.5 * necessity_score + 0.4 * overall_similarity_nli + 0.1 * brevity_score
    )

    metrics = {
        "total": total,
        "counts_per_category": dict(counts),
        "per_category_similarity": per_cat_sim,
        "per_category_similarity_nli": per_cat_sim_nli,
        "num_questions_hist": dict(num_questions_hist),
        "avg_num_questions": avg_num_questions,
        "necessity_precision": necessity_precision,
        "necessity_recall": necessity_recall,
        "resolved_proxy_rate": resolved_proxy_rate,
        "resolved_dialog_rate": resolved_dialog_rate,
        "overall_weighted_score": overall_weighted,
        "resolved_proxy_rate_nli": nli_resolved_rate,          
        "overall_similarity_nli": overall_similarity_nli,      
        "overall_weighted_score_nli": overall_weighted_nli,    
        "nli_threshold": nli_threshold,             
    }
    return metrics


def print_metrics(metrics):
    print(json.dumps(metrics, indent=2))
