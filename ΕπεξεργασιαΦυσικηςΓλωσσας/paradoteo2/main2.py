from __future__ import annotations
from pathlib import Path
import pandas as pd

from paradoteo2 import (
    read_file, vocab_size, glove_style_embeddings,
    compare_embeddings, plot_embeddings_2d, top_words_by_freq, STOPWORDS
)

# ---- ΡΥΘΜΙΣΕΙΣ PATHS ----
BASE = Path(__file__).parent
ROOT = BASE.parent
DATA_DIR = ROOT / "data" / "raw"
RECON_DIR = ROOT / "paradoteo1c" / "outputs"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Φόρτωση κειμένων ----
TEXT1 = read_file(DATA_DIR / "text1.txt")
TEXT2 = read_file(DATA_DIR / "text2.txt")
ORIGINAL = (TEXT1 + "\n\n" + TEXT2).strip()

R_RULE  = read_file(RECON_DIR / "reconstructed_rule_based.txt")
R_SKNN  = read_file(RECON_DIR / "reconstructed_sklearn_nn.txt")
R_TORCH = read_file(RECON_DIR / "reconstructed_torch_guided.txt")

# ---- Κοινές υπερ-παράμετροι ----
MIN_COUNT = 2
WINDOW = 4
MAX_DIM = 50

# Λέξεις που θέλουμε να κρατάμε πάντα στα plots (αν υπάρχουν)
KEYWORDS_KEEP = [
    "professor","springer","proceedings","doctor","contract","review",
    "submission","autumn","updates","reviewer","editor","acknowledgments",
    "festival","dragon","boat","celebrate","cooperation","acceptance","message"
]

def run():
    # Υπολογισμός κοινής διάστασης με βάση το μικρότερο λεξιλόγιο
    V_orig  = vocab_size(ORIGINAL, min_count=MIN_COUNT)
    V_rule  = vocab_size(R_RULE,   min_count=MIN_COUNT)
    V_sknn  = vocab_size(R_SKNN,   min_count=MIN_COUNT)
    V_torch = vocab_size(R_TORCH,  min_count=MIN_COUNT)
    V_min = min(V_orig, V_rule, V_sknn, V_torch)
    DIM = max(10, min(MAX_DIM, V_min - 1))
    print(f"[Info] Vocab sizes -> orig={V_orig}, rule={V_rule}, sknn={V_sknn}, torch={V_torch} | Using DIM={DIM}")

    # Χτίσιμο embeddings με ΙΔΙΑ DIM/WINDOW/MIN_COUNT παντού
    print("• Χτίζω ενσωματώσεις (PPMI+SVD) για original και reconstructions...")
    emb_orig, _ = glove_style_embeddings(ORIGINAL, dim=DIM, window=WINDOW, min_count=MIN_COUNT)
    emb_rule, _ = glove_style_embeddings(R_RULE,   dim=DIM, window=WINDOW, min_count=MIN_COUNT)
    emb_sknn, _ = glove_style_embeddings(R_SKNN,   dim=DIM, window=WINDOW, min_count=MIN_COUNT)
    emb_torch,_ = glove_style_embeddings(R_TORCH,  dim=DIM, window=WINDOW, min_count=MIN_COUNT)

    # Σύγκριση before/after
    print("• Υπολογίζω cosine(before, after) ανά λέξη...")
    df_rule  = compare_embeddings(emb_orig, emb_rule)
    df_sknn  = compare_embeddings(emb_orig, emb_sknn)
    df_torch = compare_embeddings(emb_orig, emb_torch)

    df_rule.to_csv(OUT_DIR / "word_cosine_rule_based.csv", index=False)
    df_sknn.to_csv(OUT_DIR / "word_cosine_sklearn_nn.csv", index=False)
    df_torch.to_csv(OUT_DIR / "word_cosine_torch_guided.csv", index=False)

    def summary(df: pd.DataFrame, name: str) -> str:
        return (f"{name}: mean={df.cosine_before_after.mean():.3f} | "
                f"median={df.cosine_before_after.median():.3f} | "
                f"p10={df.cosine_before_after.quantile(0.10):.3f} | "
                f"p90={df.cosine_before_after.quantile(0.90):.3f} | n={len(df)}")

    rep = "\n".join([
        "=== Word-level Before/After Cosine (PPMI+SVD) ===",
        summary(df_rule,  "Rule-based    "),
        summary(df_sknn,  "sklearn NN    "),
        summary(df_torch, "Torch-guided  "),
    ])
    print(rep)
    (OUT_DIR / "report_paradoteo2.txt").write_text(rep, encoding="utf-8")

    # Οπτικοποιήσεις για Α & Β (PCA και t-SNE) με stopword filtering + βέλη
    topA = top_words_by_freq(TEXT1, k=25, stop=list(STOPWORDS), keep=KEYWORDS_KEEP)
    topB = top_words_by_freq(TEXT2, k=25, stop=list(STOPWORDS), keep=KEYWORDS_KEEP)

    for name, emb_after in [
        ("Rule-based", emb_rule),
        ("sklearn NN", emb_sknn),
        ("Torch-guided", emb_torch),
    ]:
        plot_embeddings_2d(emb_orig, emb_after, topA,
                           title=f"A: {name} (PCA)",
                           out_png=OUT_DIR / f"A_{name.replace(' ','_')}_PCA.png",
                           method="pca")
        plot_embeddings_2d(emb_orig, emb_after, topA,
                           title=f"A: {name} (t-SNE)",
                           out_png=OUT_DIR / f"A_{name.replace(' ','_')}_tSNE.png",
                           method="tsne")
        plot_embeddings_2d(emb_orig, emb_after, topB,
                           title=f"B: {name} (PCA)",
                           out_png=OUT_DIR / f"B_{name.replace(' ','_')}_PCA.png",
                           method="pca")
        plot_embeddings_2d(emb_orig, emb_after, topB,
                           title=f"B: {name} (t-SNE)",
                           out_png=OUT_DIR / f"B_{name.replace(' ','_')}_tSNE.png",
                           method="tsne")

    print(f"\n✓ Αποτελέσματα & εικόνες στον φάκελο: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    run()
