import numpy as np
from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker

# Load data
passages_dict = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages_dict)

checker = WordRecallThresholdFactChecker()

#  Pre-calculate overlap scores for all examples (to make the sweep fast)
scores = []
labels = []
for ex in examples:
    fact_tokens = checker._preprocess(ex.fact)
    best_score = max(
        (
            checker._recall(fact_tokens, checker._preprocess(p["text"]))
            for p in ex.passages
        ),
        default=0.0,
    )
    scores.append(best_score)
    # Ground truth: 1 for 'S', 0 for 'NS' (and IR is treated as NS)
    labels.append(1 if ex.label == "S" else 0)

# 3. Sweep thresholds from 0.10 to 0.90 in steps of 0.05
best_threshold = 0.5
best_accuracy = 0.0

print(f"{'Threshold':<12} {'Accuracy':<10}")
print("-" * 22)
for t in np.arange(0.10, 0.95, 0.05):
    preds = [1 if s >= t else 0 for s in scores]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"{t:<12.2f} {acc:<10.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = t

print(
    f"\nOptimal Threshold: {best_threshold:.2f} with Accuracy: {best_accuracy * 100:.2f}%"
)
