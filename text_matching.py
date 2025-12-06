
import re
import traceback
import sys
from typing import Tuple
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

