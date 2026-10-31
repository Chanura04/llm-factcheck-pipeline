import sys, os
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker

def main():
    passages_dict = read_passages(os.path.join(PROJECT_ROOT, "data", "passages_bm25_ChatGPT_humfacts.jsonl"))
    examples = read_fact_examples(os.path.join(PROJECT_ROOT, "data", "dev_labeled_ChatGPT.jsonl"), passages_dict)
    stem_checker = WordRecallThresholdFactChecker(use_lemmatization=False)
    lem_checker = WordRecallThresholdFactChecker(use_lemmatization=True)
    stem_scores, lem_scores, labels = [], [], []
    for ex in examples:
        st = stem_checker._preprocess(ex.fact)
        lt = lem_checker._preprocess_with_lemm(ex.fact)
        stem_max = max((stem_checker._recall(st, stem_checker._preprocess(p["text"])) for p in ex.passages), default=0.0)
        lem_max = max((lem_checker._recall(lt, lem_checker._preprocess_with_lemm(p["text"])) for p in ex.passages), default=0.0)
        stem_scores.append(stem_max)
        lem_scores.append(lem_max)
        labels.append(1 if ex.label == "S" else 0)
    print(f"{'Threshold':<12} {'Stem Acc':<10} {'Lem Acc':<10} {'Winner'}")
    print("-" * 45)
    for t in np.arange(0.50, 0.75, 0.05):
        s_preds = [1 if s >= t else 0 for s in stem_scores]
        l_preds = [1 if l >= t else 0 for l in lem_scores]
        s_acc = sum(p == l for p, l in zip(s_preds, labels)) / len(labels)
        l_acc = sum(p == l for p, l in zip(l_preds, labels)) / len(labels)
        winner = "STEM" if s_acc > l_acc else ("LEM" if l_acc > s_acc else "TIE")
        print(f"{t:<12.2f} {s_acc*100:<10.2f}% {l_acc*100:<10.2f}% {winner}")

if __name__ == "__main__":
    main()