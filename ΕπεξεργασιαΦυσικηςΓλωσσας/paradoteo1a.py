from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Pattern

counter=0

# -------------------------
# Δομές & βασικός διορθωτής
# -------------------------
@dataclass
class Rule:
    pattern: Pattern[str]
    replacement: str

class Corrector:
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def correct(self, text: str, passes: int = 2) -> str:
        # εφαρμόζουμε τους κανόνες σε 2 περάσματα για να "δέσουν" αλυσιδωτές αλλαγές
        for _ in range(passes):
            for r in self.rules:
                text = r.pattern.sub(r.replacement, text)
        # γενικοί καθαρισμοί στίξης / κενών
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"\s+([,.;:?!—])", r"\1", text)
        return text

# -----------------------------------
# Προτάσεις (από τα δικά σου κείμενα)
# -----------------------------------
SENTENCE_1 = (
    "Today is our dragon boat festival, in our Chinese culture, "
    "to celebrate it with all safe and great in our lives."
)

SENTENCE_2 = (
    "During our final discuss, I told him about the new submission — "
    "the one we were waiting since last autumn, but the updates was "
    "confusing as it not included the full feedback from reviewer or maybe editor?"
)

# --------------------------
# Rules για την Πρόταση 1
# --------------------------
def build_corrector_sentence_1() -> Corrector:
    rules = [
        Rule(re.compile(r"\bour dragon boat festival\b", re.I), "the Dragon Boat Festival"),
        Rule(re.compile(r",\s*in our Chinese culture,?", re.I), " in Chinese culture"),
        Rule(re.compile(r"\bto celebrate it with all safe and great in our lives\b", re.I),
             " we celebrate it, wishing safety and prosperity for everyone"),
        # καθαρισμοί
        Rule(re.compile(r"\s*—\s*"), "—"),
        Rule(re.compile(r"\s*-\s*"), "—"),
    ]
    return Corrector(rules)

# --------------------------
# Rules για την Πρόταση 2
# --------------------------
def build_corrector_sentence_2() -> Corrector:
    rules = [
        # λεξικο-γραμματικά
        Rule(re.compile(r"\bfinal discuss\b", re.I), "final discussion"),
        Rule(re.compile(r"\bwe were waiting\b", re.I), "we had been waiting"),
        Rule(re.compile(r"\bwaiting\b(?!\s+for)", re.I), "waiting for"),
        Rule(re.compile(r"\bupdates was\b", re.I), "updates were"),
        Rule(re.compile(r"\bit not included\b", re.I), "they did not include"),
        Rule(re.compile(r"\bfrom reviewer\b", re.I), "from the reviewer"),
        Rule(re.compile(r"\bmaybe editor\b\??", re.I), "or the editor"),
        Rule(re.compile(r"\bas\b(?=\s+(it|they)\b)", re.I), "because"),
        # στίξη/κενά
        Rule(re.compile(r"\s*—\s*"), "—"),
        Rule(re.compile(r"\s*-\s*"), "—"),
    ]
    return Corrector(rules)

# --------------------------
# main: menu 1 ή 2
# --------------------------
if __name__ == "__main__":
    c1 = build_corrector_sentence_1()
    c2 = build_corrector_sentence_2()

    while True:
        if(counter>=1):
            print("Πατήστε έντερ για συνέχεια")
            input()
        print("\nΠρόταση 1 (αδιόρθωτη):")
        print(SENTENCE_1)
        print("\nΠρόταση 2 (αδιόρθωτη):")
        print(SENTENCE_2)
        
            
        counter +=1 
        choice = input("\nΔιάλεξε ποια πρόταση να διορθωθεί (1 ή 2) ή 'n' για έξοδο: ").strip().lower()

        if choice == "1":
            print("\nΔιορθωμένη πρόταση 1:")
            print(c1.correct(SENTENCE_1))
        elif choice == "2":
            print("\nΔιορθωμένη πρόταση 2:")
            print(c2.correct(SENTENCE_2))
        elif choice == "n":
            print("Τερματισμός προγράμματος.")
            break
        else:
            print("Μη έγκυρη επιλογή. Δώσε 1, 2 ή n.")
