# scratch/probe_real_data.py
import sys, os, re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker


def main():
    passages_dict = read_passages(os.path.join(PROJECT_ROOT, "data", "passages_bm25_ChatGPT_humfacts.jsonl"))
    examples = read_fact_examples(os.path.join(PROJECT_ROOT, "data", "dev_labeled_ChatGPT.jsonl"), passages_dict)

    checker = WordRecallThresholdFactChecker(use_lemmatization=False)

    facts_with_diff = 0
    for ex in examples:
        s = checker._preprocess(ex.fact)
        l = checker._preprocess_with_lemm(ex.fact)
        if s != l:
            facts_with_diff += 1

    print(f"Total examples: {len(examples)}")
    print(f"Facts where stem set != lem set: {facts_with_diff} ({facts_with_diff / len(examples) * 100:.1f}%)")
    if facts_with_diff > 0:
        print(">> SUCCESS: Lemmatization is actively changing the token sets on real data.")
    else:
        print(">> WARNING: Token sets are identical. Check NLTK wordnet download.")


if __name__ == "__main__":
    main()