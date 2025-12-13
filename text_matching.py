
import re
import traceback
import sys
from typing import Tuple
import torch

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    NLI_MODEL_NAME = "roberta-large-mnli"
    _NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    _NLI_MODEL.eval()

    _NLI_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _NLI_MODEL.to(_NLI_DEVICE)

    _NLI_LABEL2ID = _NLI_MODEL.config.label2id
    _ENTAIL_ID = (
        _NLI_LABEL2ID.get("ENTAILMENT")
        or _NLI_LABEL2ID.get("entailment")
        or 2
    )
except Exception as e:
    print(f"[WARN] Failed to load NLI model: {repr(e)}", file=sys.stderr)
    traceback.print_exc()
    print("[WARN] NLI model is None — NLI similarity will always be 0.0", file=sys.stderr)
    _NLI_MODEL = None
    _NLI_TOKENIZER = None
    _NLI_DEVICE = None
    _ENTAIL_ID = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from numpy.linalg import norm
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"[WARN] Failed to load SentenceTransformer: {repr(e)}", file=sys.stderr)
    traceback.print_exc()
    print("[WARN] EMBED_MODEL is None — semantic similarity will always be 0.0", file=sys.stderr)
    EMBED_MODEL = None
    np = None

def normalize_text(s: str) -> str:
    if s is None:
        return ''
    return re.sub(r'\s+', ' ', s.strip().lower())

def exact_contains_match(a: str, b: str) -> Tuple[bool, float]:
    if not a or not b:
        return False, 0.0
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if a_n == b_n:
        return True, 1.0
    if a_n in b_n or b_n in a_n:
        return True, 0.95
    return False, 0.0

def embedding_similarity(a: str, b: str) -> float:
    if EMBED_MODEL is None:
        return 0.0
    v1 = EMBED_MODEL.encode([a])[0]
    v2 = EMBED_MODEL.encode([b])[0]
    # cosine
    return float(np.dot(v1, v2) / (norm(v1)*norm(v2)))

def best_match_score(model_out: str, gold: str, threshold: float = 0.75, return_pass: bool = False):
    """
    Returns a similarity score in [0,1]. If return_pass=True, also returns a boolean flag
    indicating score >= threshold.

    Logic:
      1) exact/contains -> 1.0 / 0.95
      2) else embedding cosine if available
      3) else 0.0
    """
    # Fast lexical path
    match, score = exact_contains_match(model_out, gold)
    if match:
        return (score, score >= threshold) if return_pass else score

    # Semantic path
    sim = 0.0
    try:
        sim = embedding_similarity(model_out, gold)
    except Exception:
        # If EMBED_MODEL is None or encode fails, keep sim=0.0
        pass

    return (sim, sim >= threshold) if return_pass else sim

def nli_entailment_prob(premise: str, hypothesis: str) -> float:
    """
    Return P(entailment | premise, hypothesis) from the NLI model.
    If the NLI model is not available, returns 0.0.
    """
    if _NLI_MODEL is None or _NLI_TOKENIZER is None or _ENTAIL_ID is None:
        return 0.0

    premise = premise or ""
    hypothesis = hypothesis or ""
    if not premise or not hypothesis:
        return 0.0

    inputs = _NLI_TOKENIZER(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(_NLI_DEVICE)

    with torch.no_grad():
        logits = _NLI_MODEL(**inputs).logits
        probs = logits.softmax(dim=-1).squeeze(0)

    return float(probs[_ENTAIL_ID].item())


def nli_question_similarity(a: str, b: str) -> float:
    """
    Symmetric NLI-based similarity between two questions in [0,1].

    1) If lexically equal / one contains the other -> reuse exact_contains_match.
    2) Otherwise, use 0.5 * (P(E | a→b) + P(E | b→a)).
    3) If no NLI model is available, returns 0.0 (see warnings).
    """
    if not a or not b:
        return 0.0

    # Fast lexical check for trivial matches
    match, score = exact_contains_match(a, b)
    if match:
        return score

    p1 = nli_entailment_prob(a, b)
    p2 = nli_entailment_prob(b, a)
    return 0.5 * (p1 + p2)
