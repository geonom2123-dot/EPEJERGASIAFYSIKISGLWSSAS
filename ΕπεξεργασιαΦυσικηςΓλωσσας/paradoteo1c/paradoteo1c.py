# paradoteo1c.py
from __future__ import annotations
import re
import numpy as np
from pathlib import Path
from typing import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- I/O ----------
def read_texts(data_dir: str | Path) -> dict[str, str]:
    p = Path(data_dir)
    texts = {}
    for f in p.glob("*.txt"):
        texts[f.stem] = f.read_text(encoding="utf-8").strip()
    return texts

# ---------- basic text utils ----------
def sent_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]

def cleanup(text: str) -> str:
    text = re.sub(r"\s+([,.;:?!—])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

# ---------- metrics (ό,τι χρειάζεται για το C) ----------
def cosine_similarity(a: str, b: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform([a, b]).toarray()
    va, vb = X[0], X[1]
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)

def _ngram_counts(s: str, n: int = 1) -> dict[str, int]:
    toks = re.findall(r"[A-Za-z]+", s.lower())
    grams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    d: dict[str, int] = {}
    for g in grams:
        d[g] = d.get(g, 0) + 1
    return d

def prf_overlap(pred: str, ref: str, n: int = 1) -> tuple[float, float, float]:
    p = _ngram_counts(pred, n)
    r = _ngram_counts(ref, n)
    inter = sum(min(p.get(k, 0), r.get(k, 0)) for k in p.keys())
    p_sum = sum(p.values()) or 1e-9
    r_sum = sum(r.values()) or 1e-9
    precision = inter / p_sum
    recall = inter / r_sum
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)

def keyword_preservation(text: str, ref: str, keywords: Iterable[str]) -> float:
    def norm(s: str) -> set[str]:
        s = re.sub(r"[^a-zA-Z\s]", " ", s).lower()
        return set(s.split())
    T = norm(text); R = norm(ref); K = {k.lower() for k in keywords}
    K_in_ref = K & R
    if not K_in_ref:
        return 1.0
    preserved = len(K_in_ref & T) / len(K_in_ref)
    return float(preserved)

def sentence_coherence(text: str) -> float:
    sents = sent_split(text)
    if len(sents) < 2:
        return 1.0
    vec = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform(sents).toarray()
    sims = []
    for i in range(len(sents)-1):
        a, b = X[i], X[i+1]
        denom = (np.linalg.norm(a)*np.linalg.norm(b)) or 1e-9
        sims.append(float(np.dot(a, b)/denom))
    return float(np.mean(sims))

_VOWELS = "aeiouy"
def _count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w: return 0
    syll, prev_v = 0, False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_v:
            syll += 1
        prev_v = is_v
    if w.endswith("e") and syll > 1:
        syll -= 1
    return max(syll, 1)

def flesch_reading_ease(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    if not words or not sentences:
        return 100.0
    syl = sum(_count_syllables(w) for w in words)
    W = len(words); S = len(sentences)
    ASL = W / S; ASW = syl / W
    return float(206.835 - 1.015*ASL - 84.6*ASW)

def sentence_length_stats(text: str) -> tuple[float, float]:
    sents = sent_split(text)
    lens = [len(re.findall(r"[A-Za-z]+", s)) for s in sents]
    if not lens:
        return 0.0, 0.0
    arr = np.array(lens, dtype=float)
    return float(arr.mean()), float(arr.std())

_POSITIVE_LEX = {
    "thank","appreciate","support","celebrate","safe","safety","prosperity",
    "cooperation","acceptance","grateful","goals","future","great","kindly"
}
def positive_tone_ratio(text: str) -> float:
    toks = re.findall(r"[A-Za-z]+", text.lower())
    if not toks: return 0.0
    cnt = sum(1 for t in toks if t in _POSITIVE_LEX)
    return float(cnt/len(toks))
