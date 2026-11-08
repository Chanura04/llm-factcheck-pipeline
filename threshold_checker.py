import numpy as np
from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker

# Load data
passages_dict = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages_dict)

checker = WordRecallThresholdFactChecker()
methods = [
    ("Recall", checker._recall),
    ("F1", checker._f1),
    ("Jaccard", checker._jaccard),
    ("Cosine", checker._cosine),
]

# One overlap score per claim: the best passage, for every scoring method.
score_lists = {name: [] for name, _ in methods}
labels = []
for ex in examples:
    fact_tokens = checker._preprocess(ex.fact)
    passage_sets = [checker._preprocess(p["text"]) for p in ex.passages]
    for name, score_fn in methods:
        if not fact_tokens or not passage_sets:
            best_score = 0.0
        else:
            best_score = max(score_fn(fact_tokens, tokens) for tokens in passage_sets)
        score_lists[name].append(best_score)
    # Ground truth: 1 for 'S', 0 for 'NS' (and IR is treated as NS)
    labels.append(1 if ex.label == "S" else 0)


n = len(labels)
print(f"{'Method':<12} {'Threshold':<12} {'Accuracy'}")
print("-" * 36)
for name, _ in methods:
    scores = score_lists[name]
    best_threshold = 0.0
    best_accuracy = -1.0
    for t in np.arange(0.01, 0.99, 0.01):
        correct = sum((s > t) == lab for s, lab in zip(scores, labels))
        acc = correct / n if n else 0.0
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(t)
    print(f"{name:<12} {best_threshold:<12.2f} {best_accuracy * 100:.2f}%")
