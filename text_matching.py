
import re
from typing import Tuple
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from numpy.linalg import norm
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
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

def best_match_score(model_out: str, gold: str, threshold=0.75) -> float:
    # exact/contains first
    match, score = exact_contains_match(model_out, gold)
    if match:
        return score
    # embedding fallback
    try:
        sim = embedding_similarity(model_out, gold)
        return sim
    except Exception:
        return 0.0
