import sys, os, csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker


def main():
    passages_dict = read_passages(os.path.join(PROJECT_ROOT, "data", "passages_bm25_ChatGPT_humfacts.jsonl"))
    examples = read_fact_examples(os.path.join(PROJECT_ROOT, "data", "dev_labeled_ChatGPT.jsonl"), passages_dict)

    stem_checker = WordRecallThresholdFactChecker(use_lemmatization=False)
    lem_checker = WordRecallThresholdFactChecker(use_lemmatization=True)
    threshold = stem_checker.THRESHOLD

    disagreements = []

    for ex in examples:
        st_tokens = stem_checker._preprocess(ex.fact)
        lm_tokens = lem_checker._preprocess_with_lemm(ex.fact)

        st_score = max((stem_checker._recall(st_tokens, stem_checker._preprocess(p["text"])) for p in ex.passages),
                       default=0.0)
        lm_score = max(
            (lem_checker._recall(lm_tokens, lem_checker._preprocess_with_lemm(p["text"])) for p in ex.passages),
            default=0.0)

        st_pred = "S" if st_score > threshold else "NS"
        lm_pred = "S" if lm_score > threshold else "NS"
        gold = "NS" if ex.label == "IR" else ex.label

        if st_pred != lm_pred:
            disagreements.append({
                "fact": ex.fact,
                "gold": gold,
                "stem_pred": st_pred,
                "lem_pred": lm_pred,
                "stem_score": round(st_score, 4),
                "lem_score": round(lm_score, 4)
            })

    out_path = os.path.join(PROJECT_ROOT, "results", "final_disagreements.tsv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if disagreements:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=disagreements[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(disagreements)
        print(f"Saved {len(disagreements)} disagreements to {out_path}")
    else:
        print("0 disagreements found. The methods are identical on this dataset.")


if __name__ == "__main__":
    main()