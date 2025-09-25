from __future__ import annotations
from pathlib import Path
import re

def read_texts(raw_dir: str | Path) -> dict[str, str]:
    raw = Path(raw_dir)
    texts = {}
    for p in raw.glob("*.txt"):
        texts[p.stem] = p.read_text(encoding="utf-8").strip()
    return texts

def sent_split(text: str) -> list[str]:
    # Απλό split προτάσεων (αρκετό για τα συγκεκριμένα κείμενα)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]

def join_sentences(sents: list[str]) -> str:
    out = " ".join(sents)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out

def cleanup(text: str) -> str:
    text = re.sub(r"\s+([,.;:?!—])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


