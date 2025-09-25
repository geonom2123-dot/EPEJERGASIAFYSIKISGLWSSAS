from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---------------------------
# I/O + βασικά helpers
# ---------------------------
def read_file(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()

def tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())

def vocab_size(text: str, min_count: int = 2) -> int:
    freqs: Dict[str, int] = {}
    for t in tokens(text):
        freqs[t] = freqs.get(t, 0) + 1
    vocab = {w for w, c in freqs.items() if c >= min_count}
    return len(vocab)

def window_cooc(tokens_list: List[str], window: int = 4, min_count: int = 2) -> Tuple[Dict[str,int], np.ndarray]:
    # λεξιλόγιο με κατώφλι συχνότητας
    freqs: Dict[str,int] = {}
    for t in tokens_list:
        freqs[t] = freqs.get(t, 0) + 1
    vocab = {w for w, c in freqs.items() if c >= min_count}
    idx = {w:i for i,w in enumerate(sorted(vocab))}
    V = len(idx)
    C = np.zeros((V, V), dtype=np.float32)
    n = len(tokens_list)
    for i, w in enumerate(tokens_list):
        if w not in idx: continue
        wi = idx[w]
        j0 = max(0, i-window); j1 = min(n, i+window+1)
        for j in range(j0, j1):
            if j == i: continue
            u = tokens_list[j]
            if u not in idx: continue
            C[wi, idx[u]] += 1.0
    inv_idx = {i:w for w,i in idx.items()}
    return inv_idx, C

def ppmi_matrix(C: np.ndarray, k_smoothing: float = 1.0) -> np.ndarray:
    C = C.copy().astype(np.float64)
    total = C.sum() + 1e-9
    row = C.sum(axis=1, keepdims=True) + 1e-9
    col = C.sum(axis=0, keepdims=True) + 1e-9
    P = (C * total) / (row @ col)
    P /= k_smoothing
    with np.errstate(divide='ignore'):
        ppmi = np.maximum(np.log(P), 0.0)
    return ppmi.astype(np.float32)

def glove_style_embeddings(text: str, dim: int, window: int = 4, min_count: int = 2) -> Tuple[Dict[str,np.ndarray], Dict[str,int]]:
    toks = tokens(text)
    inv_idx, C = window_cooc(toks, window=window, min_count=min_count)
    V = len(inv_idx)
    if V < dim + 1:
        raise ValueError(f"SVD dim={dim} is too large for vocab size V={V}. Choose dim ≤ V-1.")
    X = ppmi_matrix(C)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    E = svd.fit_transform(X)  # (V, dim)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    E = E / norms
    emb = {inv_idx[i]: E[i] for i in range(V)}
    return emb, {w:i for i,w in inv_idx.items()}

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / ((np.linalg.norm(u)+1e-9)*(np.linalg.norm(v)+1e-9)))

# ---------------------------
# Σύγκριση "πριν vs μετά"
# ---------------------------
def compare_embeddings(emb_before: Dict[str,np.ndarray], emb_after: Dict[str,np.ndarray], top_k: int = 1000) -> pd.DataFrame:
    common = sorted(set(emb_before.keys()) & set(emb_after.keys()))
    if top_k: common = common[:top_k]
    rows = [(w, cosine(emb_before[w], emb_after[w])) for w in common]
    return pd.DataFrame(rows, columns=["word", "cosine_before_after"]).sort_values("cosine_before_after", ascending=False)

# ---------------------------
# Stopwords + επιλογή λέξεων
# ---------------------------
STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "our","we","you","he","she","they","his","her","their","i","me","my","us"
}

def top_words_by_freq(text: str, k: int = 30, stop: list[str] | None = None, keep: list[str] | None = None) -> list[str]:
    toks = tokens(text)
    stop = set(stop or [])
    keep = set((keep or []))
    freqs: dict[str,int] = {}
    for t in toks:
        if t in keep or t not in stop:
            freqs[t] = freqs.get(t, 0) + 1
    # προτεραιότητα σε keep (αν υπάρχουν), μετά συχνότητα
    base = [w for w in keep if w in freqs]
    rest = [w for w,_ in sorted(freqs.items(), key=lambda x: (-x[1], x[0])) if w not in base]
    out = base + rest
    return out[:k]

# ---------------------------
# Οπτικοποιήσεις (PCA / t-SNE) με βέλη Before→After
# ---------------------------
def plot_embeddings_2d(
    emb_a: dict[str,np.ndarray],
    emb_b: dict[str,np.ndarray],
    words: list[str],
    title: str,
    out_png: Path,
    method: str = "pca"
):
    # κράτησε μόνο λέξεις που υπάρχουν και στα δύο και δεν είναι stopwords
    labels = [w for w in words if (w in emb_a and w in emb_b and w not in STOPWORDS)]
    if not labels:
        return

    A = np.vstack([emb_a[w] for w in labels])
    B = np.vstack([emb_b[w] for w in labels])
    X = np.vstack([A, B])  # πρώτα όλα τα before, μετά όλα τα after

    # διάσταση 2 με PCA ή t-SNE
    if method.lower() == "tsne":
        n = len(labels)
        perp = max(5, min(30, n//2 if n > 6 else max(2, n-1)))
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init="random", learning_rate="auto")
    else:
        reducer = PCA(n_components=2, random_state=42)

    Y = reducer.fit_transform(X)
    Ya, Yb = Y[:len(labels)], Y[len(labels):]

    plt.figure(figsize=(9,7))
    # before/after σημεία
    plt.scatter(Ya[:,0], Ya[:,1], marker='o', alpha=0.7, label='Before')
    plt.scatter(Yb[:,0], Yb[:,1], marker='x', alpha=0.9, label='After')

    # βέλη Before -> After
    for i in range(len(labels)):
        dx, dy = (Yb[i,0]-Ya[i,0]), (Yb[i,1]-Ya[i,1])
        plt.arrow(Ya[i,0], Ya[i,1], dx, dy, length_includes_head=True,
                  head_width=0.12, alpha=0.45, linewidth=1.0, color="gray")

    # ετικέτες μόνο στο "after" για καθαρότητα
    for i, w in enumerate(labels):
        plt.text(Yb[i,0], Yb[i,1], w, fontsize=9)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
