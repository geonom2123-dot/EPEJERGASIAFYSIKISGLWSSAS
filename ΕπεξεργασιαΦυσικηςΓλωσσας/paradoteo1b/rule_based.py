from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Pattern, List
from utils import sent_split, join_sentences, cleanup

@dataclass
class Rule:
    pattern: Pattern[str]
    replacement: str

class Corrector:
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def correct_sentence(self, text: str) -> str:
        for r in self.rules:
            text = r.pattern.sub(r.replacement, text)
        return cleanup(text)

    def correct_text(self, text: str) -> str:
        sents = sent_split(text)
        return join_sentences([self.correct_sentence(s) for s in sents])

# Rules σχεδιασμένα για τα συγκεκριμένα κείμενα (ήπια, ερμηνεύσιμα)
def build_rules_festival() -> List[Rule]:
    return [
        Rule(re.compile(r"\bour dragon boat festival\b", re.I), "the Dragon Boat Festival"),
        Rule(re.compile(r",\s*in our Chinese culture,?", re.I), " in Chinese culture"),
        Rule(re.compile(r"\bto celebrate it with all safe and great in our lives\b", re.I),
             " we celebrate it, wishing safety and prosperity for everyone"),
        Rule(re.compile(r"\bThank your message\b", re.I), "Thank you for your message"),
        Rule(re.compile(r"\bcontract checking\b", re.I), "contract review"),
        Rule(re.compile(r"\bapproved message\b", re.I), "approval"),
        Rule(re.compile(r"\bI am very appreciated\b", re.I), "I greatly appreciate"),
        Rule(re.compile(r"\bSpringer proceedings publication\b", re.I),
             "publication in the Springer proceedings"),
    ]

def build_rules_submission() -> List[Rule]:
    return [
        Rule(re.compile(r"\bfinal discuss\b", re.I), "final discussion"),
        Rule(re.compile(r"\bwe were waiting\b", re.I), "we had been waiting"),
        Rule(re.compile(r"\bwaiting\b(?!\s+for)", re.I), "waiting for"),
        Rule(re.compile(r"\bupdates was\b", re.I), "updates were"),
        Rule(re.compile(r"\bit not included\b", re.I), "they did not include"),
        Rule(re.compile(r"\bfrom reviewer\b", re.I), "from the reviewer"),
        Rule(re.compile(r"\bmaybe editor\b\??", re.I), "or the editor"),
        Rule(re.compile(r"\bas\b(?=\s+(it|they)\b)", re.I), "because"),
        Rule(re.compile(r"\bbit delay\b", re.I), "a slight delay"),
        Rule(re.compile(r"\bcame finally\b", re.I), "finally came"),
    ]

def build_pipeline() -> Corrector:
    rules = build_rules_festival() + build_rules_submission()
    return Corrector(rules)


