import json
from collections import Counter, defaultdict
from text_matching import best_match_score


def compute_metrics_from_json(json_path, embed_threshold=0.75, brevity_max=1):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counts = Counter()
    sim_sums = defaultdict(float)
    sim_counts = defaultdict(int)
    num_questions_hist = Counter()
    resolved_proxy_count = 0
    resolved_dialog_count = 0
    dialog_total = 0
    necessity_tp = necessity_fp = necessity_fn = 0
    total = len(data)
    for ex in data:
        cat = ex.get('ambiguity_type','unknown')
        counts[cat]+=1
        nq = ex.get('num_questions',0)
        num_questions_hist[nq]+=1
        best_sim = ex.get('model_question_best_similarity', 0.0)
        if nq>0:
            sim_sums[cat]+=best_sim
            sim_counts[cat]+=1
        if ex.get('resolved_proxy', False):
            resolved_proxy_count+=1
        dialog = ex.get('dialog')
        if dialog and dialog.get('resolved_dialog') is not None:
            dialog_total += 1
            if dialog.get('resolved_dialog'):
                resolved_dialog_count += 1
        # necessity: only 'preferences' should ask question
        asked = 1 if nq>0 else 0
        if cat=='preferences' and asked==1:
            necessity_tp += 1
        if cat!='preferences' and asked==1:
            necessity_fp += 1
        if cat=='preferences' and asked==0:
            necessity_fn += 1
    # compute stats
    per_cat_sim = {c: (sim_sums[c]/sim_counts[c]) if sim_counts[c]>0 else None for c in counts.keys()}
    avg_num_questions = sum(k*v for k,v in num_questions_hist.items()) / max(1, total)
    necessity_precision = necessity_tp / (necessity_tp + necessity_fp) if (necessity_tp + necessity_fp)>0 else None
    necessity_recall = necessity_tp / (necessity_tp + necessity_fn) if (necessity_tp + necessity_fn)>0 else None
    resolved_proxy_rate = resolved_proxy_count/total if total>0 else 0.0
    resolved_dialog_rate = resolved_dialog_count/dialog_total if dialog_total>0 else None
    # weighted score per example (approx): we'll compute average using per-example components
    # For simplicity here compute aggregate score: necessity as fraction correct, similarity as avg across all questions, brevity as fraction with <= brevity_max
    necessity_score = (necessity_tp) / max(1, sum(1 for ex in data if ex.get('ambiguity_type')=='preferences'))
    overall_similarity = sum(ex.get('model_question_best_similarity',0.0) for ex in data if ex.get('num_questions',0)>0)
    overall_similarity /= max(1, sum(1 for ex in data if ex.get('num_questions',0)>0))
    brevity_score = sum(1 for ex in data if ex.get('num_questions',0) <= brevity_max) / max(1, total)
    overall_weighted = 0.5*necessity_score + 0.4*overall_similarity + 0.1*brevity_score
    metrics = {
        'total': total,
        'counts_per_category': dict(counts),
        'per_category_similarity': per_cat_sim,
        'num_questions_hist': dict(num_questions_hist),
        'avg_num_questions': avg_num_questions,
        'necessity_precision': necessity_precision,
        'necessity_recall': necessity_recall,
        'resolved_proxy_rate': resolved_proxy_rate,
        'resolved_dialog_rate': resolved_dialog_rate,
        'overall_weighted_score': overall_weighted
    }
    return metrics

def print_metrics(metrics):
    print(json.dumps(metrics, indent=2))
