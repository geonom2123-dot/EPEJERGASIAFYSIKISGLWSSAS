from __future__ import annotations
from pathlib import Path
from utils import read_texts
from rule_based import build_pipeline as build_rule_based
from sklearn_nn import build_pipeline as build_sklearn_nn
from torch_rules import build_pipeline as build_torch_rules
from evaluation import cosine_similarity, prf_overlap

OUT_DIR = Path("outputs")

def run_all():
    texts = read_texts("data/raw")
    assert "text1" in texts and "text2" in texts, "Χρειάζομαι data/raw/text1.txt και data/raw/text2.txt"
    full_text = (texts["text1"] + "\n\n" + texts["text2"]).strip()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Rule-based
    rb = build_rule_based()
    out_rb = rb.correct_text(full_text)
    (OUT_DIR / "reconstructed_rule_based.txt").write_text(out_rb, encoding="utf-8")

    # 2) sklearn NN paraphraser
    nn = build_sklearn_nn(threshold=0.30)
    out_nn = nn.rewrite_text(full_text)
    (OUT_DIR / "reconstructed_sklearn_nn.txt").write_text(out_nn, encoding="utf-8")

    # 3) PyTorch classifier-guided
    tg = build_torch_rules()
    out_tg = tg.rewrite_text(full_text)
    (OUT_DIR / "reconstructed_torch_guided.txt").write_text(out_tg, encoding="utf-8")

    # --- Απλή αξιολόγηση σε σχέση με το αρχικό ---
    cos_rb = cosine_similarity(full_text, out_rb)
    cos_nn = cosine_similarity(full_text, out_nn)
    cos_tg = cosine_similarity(full_text, out_tg)

    # n-gram overlap (unigram & bigram) έναντι original (ενδεικτικά)
    u_rb = prf_overlap(out_rb, full_text, n=1)
    b_rb = prf_overlap(out_rb, full_text, n=2)
    u_nn = prf_overlap(out_nn, full_text, n=1)
    b_nn = prf_overlap(out_nn, full_text, n=2)
    u_tg = prf_overlap(out_tg, full_text, n=1)
    b_tg = prf_overlap(out_tg, full_text, n=2)

    report = f"""=== Reconstruction Report ===
Cosine (TF-IDF 1-2gram) vs Original:
 - Rule-based:          {cos_rb:.3f}
 - sklearn NN:          {cos_nn:.3f}
 - Torch-guided Rules:  {cos_tg:.3f}

Unigram Overlap P/R/F1 vs Original:
 - Rule-based:          P={u_rb[0]:.3f} R={u_rb[1]:.3f} F1={u_rb[2]:.3f}
 - sklearn NN:          P={u_nn[0]:.3f} R={u_nn[1]:.3f} F1={u_nn[2]:.3f}
 - Torch-guided Rules:  P={u_tg[0]:.3f} R={u_tg[1]:.3f} F1={u_tg[2]:.3f}

Bigram Overlap P/R/F1 vs Original:
 - Rule-based:          P={b_rb[0]:.3f} R={b_rb[1]:.3f} F1={b_rb[2]:.3f}
 - sklearn NN:          P={b_nn[0]:.3f} R={b_nn[1]:.3f} F1={b_nn[2]:.3f}
 - Torch-guided Rules:  P={b_tg[0]:.3f} R={b_tg[1]:.3f} F1={b_tg[2]:.3f}
"""
    print(report)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")

if __name__ == "__main__":
    run_all()
