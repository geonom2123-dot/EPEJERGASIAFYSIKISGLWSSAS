# main1c.py (header)
from __future__ import annotations
from pathlib import Path
import sys

# Δείξε στον Python πού είναι ο φάκελος του ερωτήματος B
BASE_DIR = Path(__file__).parent           # .../paradoteo1c
ROOT_DIR = BASE_DIR.parent                 # .../ΕπεξεργασιαΦυσικηςΓλωσσας
B_DIR = ROOT_DIR / "paradoteo1b"           # .../paradoteo1b

from paradoteo1c import (
    read_texts, cosine_similarity, prf_overlap, keyword_preservation,
    sentence_coherence, flesch_reading_ease, sentence_length_stats, positive_tone_ratio
)

# Πρόσθεσε τον φάκελο B στο sys.path για να κάνουμε import τα .py του
if str(B_DIR) not in sys.path:
    sys.path.insert(0, str(B_DIR))

# ΤΩΡΑ μπορούμε να κάνουμε import ΚΑΝΟΝΙΚΑ τα αρχεία .py του B χωρίς __init__.py
from rule_based import build_pipeline as build_rule_based
from sklearn_nn import build_pipeline as build_sklearn_nn
from torch_rules import build_pipeline as build_torch_rules


# Ρυθμίσεις paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "outputs"

KEYWORDS = [
    "dragon","boat","festival","chinese","professor","springer","proceedings",
    "doctor","contract","review","submission","autumn","updates","reviewer",
    "editor","acknowledgments","acceptance","cooperation"
]

def fmt_metrics(name: str, out: str, ref: str) -> str:
    cos = cosine_similarity(ref, out)
    uP,uR,uF = prf_overlap(out, ref, n=1)
    bP,bR,bF = prf_overlap(out, ref, n=2)
    kp = keyword_preservation(out, ref, KEYWORDS)
    coh = sentence_coherence(out)
    fre = flesch_reading_ease(out)
    sl_mean, sl_std = sentence_length_stats(out)
    tone = positive_tone_ratio(out)
    return (f"{name:>18} | cos={cos:.3f} | uniF1={uF:.3f} | biF1={bF:.3f} | "
            f"KP={kp:.2f} | Coh={coh:.3f} | Flesch={fre:.1f} | SentLen={sl_mean:.1f}±{sl_std:.1f} | Tone={tone:.3f}")

def run_all():
    print("Δεδομένα:", DATA_DIR.resolve())
    texts = read_texts(DATA_DIR)
    assert "text1" in texts and "text2" in texts, \
        "Βάλε data/raw/text1.txt και data/raw/text2.txt"
    full_text = (texts["text1"] + "\n\n" + texts["text2"]).strip()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Rule-based
    rb = build_rule_based()
    out_rb = rb.correct_text(full_text)
    (OUT_DIR / "reconstructed_rule_based.txt").write_text(out_rb, encoding="utf-8")

    # 2) sklearn NN
    nn = build_sklearn_nn(threshold=0.30)
    out_nn = nn.rewrite_text(full_text)
    (OUT_DIR / "reconstructed_sklearn_nn.txt").write_text(out_nn, encoding="utf-8")

    # 3) Torch-guided
    tg = build_torch_rules()
    out_tg = tg.rewrite_text(full_text)
    (OUT_DIR / "reconstructed_torch_guided.txt").write_text(out_tg, encoding="utf-8")

    # ====== Report (ό,τι είχες ήδη + extended) ======
    report = f"""=== Reconstruction Report ===
Cosine (TF-IDF 1-2gram) vs Original:
 - Rule-based:          {cosine_similarity(full_text, out_rb):.3f}
 - sklearn NN:          {cosine_similarity(full_text, out_nn):.3f}
 - Torch-guided Rules:  {cosine_similarity(full_text, out_tg):.3f}

Unigram Overlap P/R/F1 vs Original:
 - Rule-based:          P={prf_overlap(out_rb, full_text, 1)[0]:.3f} R={prf_overlap(out_rb, full_text, 1)[1]:.3f} F1={prf_overlap(out_rb, full_text, 1)[2]:.3f}
 - sklearn NN:          P={prf_overlap(out_nn, full_text, 1)[0]:.3f} R={prf_overlap(out_nn, full_text, 1)[1]:.3f} F1={prf_overlap(out_nn, full_text, 1)[2]:.3f}
 - Torch-guided Rules:  P={prf_overlap(out_tg, full_text, 1)[0]:.3f} R={prf_overlap(out_tg, full_text, 1)[1]:.3f} F1={prf_overlap(out_tg, full_text, 1)[2]:.3f}

Bigram Overlap P/R/F1 vs Original:
 - Rule-based:          P={prf_overlap(out_rb, full_text, 2)[0]:.3f} R={prf_overlap(out_rb, full_text, 2)[1]:.3f} F1={prf_overlap(out_rb, full_text, 2)[2]:.3f}
 - sklearn NN:          P={prf_overlap(out_nn, full_text, 2)[0]:.3f} R={prf_overlap(out_nn, full_text, 2)[1]:.3f} F1={prf_overlap(out_nn, full_text, 2)[2]:.3f}
 - Torch-guided Rules:  P={prf_overlap(out_tg, full_text, 2)[0]:.3f} R={prf_overlap(out_tg, full_text, 2)[1]:.3f} F1={prf_overlap(out_tg, full_text, 2)[2]:.3f}
"""

    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(report)

    # Extended Comparison (συνοπτική γραμμή ανά pipeline)
    print("=== Extended Comparison ===")
    print(fmt_metrics("Rule-based", out_rb, full_text))
    print(fmt_metrics("sklearn NN", out_nn, full_text))
    print(fmt_metrics("Torch-guided", out_tg, full_text))

if __name__ == "__main__":
    run_all()
