from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import sent_split, join_sentences, cleanup  # προσαρμοσμένο στα δικά σου paths

@dataclass
class Pair:
    src: str
    tgt: str

class NNParaphraser:
    def __init__(self, pairs: List[Pair], threshold: float = 0.35):
        self.pairs = pairs
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
        # Κάνε dense από την αρχή για συνέπεια
        self.X = self.vectorizer.fit_transform([p.src for p in self.pairs]).toarray()
        self.Xnorm = np.linalg.norm(self.X, axis=1)

    def _best_match(self, s: str) -> Tuple[int, float]:
        # Dense vector για την είσοδο
        v = self.vectorizer.transform([s]).toarray()  # shape: (1, d)
        vnorm = float(np.linalg.norm(v))
        if vnorm == 0.0:
            return 0, 0.0
        # (n_pairs, d) @ (d, 1) -> (n_pairs, 1) dense
        sims_num = (self.X @ v.T).ravel()
        denom = self.Xnorm * vnorm
        denom[denom == 0.0] = 1e-9
        sim = sims_num / denom
        idx = int(np.argmax(sim))
        return idx, float(sim[idx])

    def rewrite_sentence(self, s: str) -> str:
        idx, sim = self._best_match(s)
        if sim >= self.threshold:
            return cleanup(self.pairs[idx].tgt)
        return cleanup(s)

    def rewrite_text(self, text: str) -> str:
        sents = sent_split(text)
        return join_sentences([self.rewrite_sentence(s) for s in sents])

def default_pairs() -> List[Pair]:
    return [
        Pair("Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
             "Today is the Dragon Boat Festival in Chinese culture; we celebrate it, wishing safety and prosperity for everyone."),
        Pair("Thank your message to show our words to the doctor, as his next contract checking, to all of us.",
             "Thank you for your message to the doctor regarding his upcoming contract review on our behalf."),
        Pair("I am very appreciated the full support of the professor, for our Springer proceedings publication",
             "I greatly appreciate the professor’s full support for our publication in the Springer proceedings."),
        Pair("During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?",
             "During our final discussion, I told him about the new submission—the one we had been waiting for since last autumn—but the updates were confusing because they did not include the full feedback from the reviewer or the editor."),
        Pair("they really tried best for paper and cooperation",
             "they really did their best on the paper and the collaboration."),
        Pair("let us make sure all are safe and celebrate the outcome with strong coffee and future targets",
             "let us ensure everyone is safe and celebrate the outcome—with strong coffee and clear future goals."),
    ]

def build_pipeline(threshold: float = 0.35) -> NNParaphraser:
    return NNParaphraser(default_pairs(), threshold=threshold)
