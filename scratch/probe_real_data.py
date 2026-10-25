# scratch/probe_real_data.py
import sys, os, re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker

PASSAGES_PATH = os.path.join(PROJECT_ROOT, "data", "passages_bm25_ChatGPT_humfacts.jsonl")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "dev_labeled_ChatGPT.jsonl")
BASELINE_FILE = os.path.join(PROJECT_ROOT, "results", "day1_baseline_stem.txt")
OUT_TXT = os.path.join(PROJECT_ROOT, "results", "day1.5_probe.txt")
OUT_TSV = os.path.join(PROJECT_ROOT, "results", "day1.5_disagreements.tsv")


def _best_score(preprocess_fn, fact, passages, recall_fn):
    fact_tokens = preprocess_fn(fact)
    best = 0.0
    for p in passages:
        passage_tokens = preprocess_fn(p.get("text", ""))
        best = max(best, recall_fn(fact_tokens, passage_tokens))
    return best


def _parse_baseline_accuracy(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.search(r"Accuracy:\s*\d+/\d+\s*=\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    passages_dict = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, passages_dict)
    N = len(examples)

    checker = WordRecallThresholdFactChecker(use_lemmatization=False)
    threshold = checker.THRESHOLD

    facts_with_diff = []
    total_stem_only = 0
    total_lem_only = 0
    for ex in examples:
        s = checker._preprocess(ex.fact)
        l = checker._preprocess_with_lemm(ex.fact)
        if s != l:
            facts_with_diff.append((ex.fact, sorted(s - l), sorted(l - s)))
        total_stem_only += len(s - l)
        total_lem_only += len(l - s)

    rows = []
    for ex in examples:
        stem_score = _best_score(checker._preprocess, ex.fact, ex.passages, checker._recall)
        lem_score = _best_score(checker._preprocess_with_lemm, ex.fact, ex.passages, checker._recall)
        stem_pred = "S" if stem_score > threshold else "NS"
        lem_pred = "S" if lem_score > threshold else "NS"
        gold = "NS" if ex.label == "IR" else ex.label
        rows.append((ex.fact, gold, stem_pred, lem_pred, stem_score, lem_score))

    stem_correct = sum(1 for r in rows if r[2] == r[1])
    lem_correct = sum(1 for r in rows if r[3] == r[1])
    stem_acc = stem_correct / N if N else 0.0
    lem_acc = lem_correct / N if N else 0.0

    disagreements = [r for r in rows if r[2] != r[3]]
    stem_s_lem_ns = sum(1 for r in disagreements if r[2] == "S" and r[3] == "NS")
    stem_ns_lem_s = sum(1 for r in disagreements if r[2] == "NS" and r[3] == "S")

    baseline_acc = _parse_baseline_accuracy(BASELINE_FILE)

    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 78)
    out("DAY 1.5 PROBE — DOES LEMMATIZATION WORK ON THE REAL DEV SET?")
    out("=" * 78)
    out(f"Examples      : {N}")
    out(f"Threshold     : {threshold}\n")

    out("-" * 78)
    out("A. TOKEN-LEVEL DIFF")
    out("-" * 78)
    out(f"Facts where stem set != lem set : {len(facts_with_diff):>4} / {N}  ({len(facts_with_diff) / N * 100:.1f}%)")
    out(f"Facts where stem set == lem set : {N - len(facts_with_diff):>4} / {N}  ({(N - len(facts_with_diff)) / N * 100:.1f}%)")
    out(f"Total tokens in stem only       : {total_stem_only}")
    out(f"Total tokens in lem only        : {total_lem_only}\n")

    for i, (fact, stem_only, lem_only) in enumerate(facts_with_diff[:5], 1):
        out(f"  [{i}] {fact!r}")
        out(f"       stem_only = {stem_only}")
        out(f"       lem_only  = {lem_only}\n")

    out("-" * 78)
    out("B. PREDICTION PREVIEW")
    out("-" * 78)
    out(f"Stemming      accuracy : {stem_correct}/{N} = {stem_acc * 100:6.2f}%")
    out(f"Lemmatization accuracy : {lem_correct}/{N} = {lem_acc * 100:6.2f}%")
    out(f"Δ (lem - stem)         : {(lem_acc - stem_acc) * 100:+.2f} pp\n")

    out("-" * 78)
    out("C. WHERE THE TWO PATHS DISAGREE")
    out("-" * 78)
    out(f"Total disagreements : {len(disagreements)} / {N}")
    out(f"  stem=S, lem=NS    : {stem_s_lem_ns}")
    out(f"  stem=NS, lem=S    : {stem_ns_lem_s}\n")

    for fact, gold, sp, lp, ss, ls in disagreements:
        out(f"  {fact!r}")
        out(f"    gold={gold}  stem={sp} ({ss:.3f})  lem={lp} ({ls:.3f})\n")

    out("-" * 78)
    out("D. CONSISTENCY vs DAY 1 BASELINE")
    out("-" * 78)
    if baseline_acc is None:
        out(f"Baseline file not found: {BASELINE_FILE}")
        out("(Run: python factchecking_main.py --mode word_overlap > results/day1_baseline_stem.txt)")
    else:
        out(f"Baseline accuracy : {baseline_acc * 100:6.2f}%")
        out(f"Manual stem acc.  : {stem_acc * 100:6.2f}%")
        if abs(baseline_acc - stem_acc) < 1e-9:
            out(">> MATCH — predict() still uses stemming, nothing was broken.")
        else:
            out(">> MISMATCH — investigate before continuing to Day 2.")
    out("=" * 78)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    with open(OUT_TSV, "w") as f:
        f.write("fact\tgold\tstem_pred\tlem_pred\tstem_score\tlem_score\n")
        for fact, gold, sp, lp, ss, ls in disagreements:
            fact_clean = fact.replace("\t", " ").replace("\n", " ")
            f.write(f"{fact_clean}\t{gold}\t{sp}\t{lp}\t{ss:.4f}\t{ls:.4f}\n")
    print(f"\n[OK] Full probe saved to {OUT_TXT}")
    print(f"[OK] Disagreement table saved to {OUT_TSV}")


if __name__ == "__main__":
    main()