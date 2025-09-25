from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import sent_split, join_sentences, cleanup

@dataclass
class Labeled:
    text: str
    label: int  # 0: festival, 1: submission

class TinyClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.linear(x)

def default_trainset() -> List[Labeled]:
    return [
        # festival
        Labeled("dragon boat festival chinese culture celebrate", 0),
        Labeled("professor support springer proceedings publication", 0),
        Labeled("message to doctor contract review", 0),
        # submission
        Labeled("final discussion new submission waiting since last autumn updates", 1),
        Labeled("updates confusing not include full feedback reviewer editor", 1),
        Labeled("acknowledgments section edit before sending again", 1),
    ]

def build_vectorizer(train: List[Labeled]) -> TfidfVectorizer:
    vec = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
    vec.fit([x.text for x in train])
    return vec

def train_classifier(vec: TfidfVectorizer, train: List[Labeled], epochs: int = 200, lr: float = 0.1) -> TinyClassifier:
    X = vec.transform([x.text for x in train]).toarray().astype(np.float32)
    y = np.array([x.label for x in train], dtype=np.int64)

    model = TinyClassifier(in_dim=X.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model

def pred_label(model: TinyClassifier, vec: TfidfVectorizer, s: str) -> int:
    v = vec.transform([s]).toarray().astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(v))
        return int(torch.argmax(logits, dim=1).item())

def rules_festival(s: str) -> str:
    s = re.sub(r"\bour dragon boat festival\b", "the Dragon Boat Festival", s, flags=re.I)
    s = re.sub(r",\s*in our Chinese culture,?", " in Chinese culture", s, flags=re.I)
    s = re.sub(r"\bI am very appreciated\b", "I greatly appreciate", s, flags=re.I)
    return s

def rules_submission(s: str) -> str:
    s = re.sub(r"\bfinal discuss\b", "final discussion", s, flags=re.I)
    s = re.sub(r"\bupdates was\b", "updates were", s, flags=re.I)
    s = re.sub(r"\bit not included\b", "they did not include", s, flags=re.I)
    s = re.sub(r"\bfrom reviewer\b", "from the reviewer", s, flags=re.I)
    s = re.sub(r"\bwaiting\b(?!\s+for)", "waiting for", s, flags=re.I)
    return s

class TorchGuidedRewriter:
    def __init__(self):
        train = default_trainset()
        self.vec = build_vectorizer(train)
        self.clf = train_classifier(self.vec, train)

    def rewrite_sentence(self, s: str) -> str:
        label = pred_label(self.clf, self.vec, s)
        out = rules_festival(s) if label == 0 else rules_submission(s)
        return cleanup(out)

    def rewrite_text(self, text: str) -> str:
        sents = sent_split(text)
        return join_sentences([self.rewrite_sentence(s) for s in sents])

def build_pipeline() -> TorchGuidedRewriter:
    return TorchGuidedRewriter()

