from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 1) TF-IDF cosine similarity (ολόκληρου κειμένου)
def cosine_similarity(a: str, b: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform([a, b]).toarray()
    va, vb = X[0], X[1]
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)

# 2) Απλό n-gram overlap (Precision/Recall/F1) για unigrams+bigrams
def ngram_counts(s: str, n: int = 1) -> dict[str, int]:
    toks = s.lower().split()
    grams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    d: dict[str, int] = {}
    for g in grams:
        d[g] = d.get(g, 0) + 1
    return d

def prf_overlap(pred: str, ref: str, n: int = 1) -> tuple[float, float, float]:
    p = ngram_counts(pred, n)
    r = ngram_counts(ref, n)
    inter = sum(min(p.get(k, 0), r.get(k, 0)) for k in p.keys())
    p_sum = sum(p.values()) or 1e-9
    r_sum = sum(r.values()) or 1e-9
    precision = inter / p_sum
    recall = inter / r_sum
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)
