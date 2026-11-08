"""
ablation_experiments.py
=======================
Produces the two sets of metrics requested for the report:

A. PRUNING ABLATION  (Entailment §2.2)
   • No-pruning baseline: accuracy, macro-F1, runtime
   • Pruning system:      accuracy, macro-F1, runtime
   • Similarity metric used by the sieve: reported explicitly (Recall)
   • Average candidates removed per claim

B. PREPROCESSING ABLATION  (§2.1.9–§2.1.10)
   Four granularity conditions × title on/off = 8 rows:
     1. Whole passage  / no title
     2. Whole passage  / with title
     3. Single sentence / no title
     4. Single sentence / with title
     5. Sentence + 2-sent window / no title   ← final system
     6. Sentence + 2-sent window / with title ← final system
   Reports: accuracy, precision-S, recall-S, F1-S, macro-F1,
            avg candidates / claim, runtime

Outputs
-------
  eda_plots/pruning_ablation.json
  eda_plots/pruning_ablation.png
  eda_plots/preprocessing_ablation_full.json
  eda_plots/preprocessing_ablation_full.png

Usage
-----
  python ablation_experiments.py                  # CPU, NLI from cache if present
  python ablation_experiments.py --cuda           # GPU
  python ablation_experiments.py --max_examples 50  # quick smoke-test
"""

import json
import os
import time
import argparse
import numpy as np

import nltk
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factcheck import WordRecallThresholdFactChecker, EntailmentFactChecker, EntailmentModel
from factchecking_main import read_passages, read_fact_examples

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEV_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
OUTPUT_DIR = "eda_plots"
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _summarize(preds, labels):
    """Return full metric dict from parallel pred/label lists."""
    tp = sum(p == "S" and g == "S" for p, g in zip(preds, labels))
    fn = sum(p == "NS" and g == "S" for p, g in zip(preds, labels))
    fp = sum(p == "S" and g == "NS" for p, g in zip(preds, labels))
    tn = sum(p == "NS" and g == "NS" for p, g in zip(preds, labels))
    total = len(labels)
    correct = tp + tn
    acc = correct / total * 100.0 if total else 0.0
    p_s = tp / (tp + fp) if (tp + fp) else 0.0
    r_s = tp / (tp + fn) if (tp + fn) else 0.0
    f1_s = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) else 0.0
    p_ns = tn / (tn + fn) if (tn + fn) else 0.0
    r_ns = tn / (tn + fp) if (tn + fp) else 0.0
    f1_ns = 2 * p_ns * r_ns / (p_ns + r_ns) if (p_ns + r_ns) else 0.0
    macro_f1 = (f1_s + f1_ns) / 2.0
    return dict(
        tp=tp, fn=fn, fp=fp, tn=tn,
        total=total, correct=correct,
        accuracy=acc,
        precision_s=p_s * 100.0,
        recall_s=r_s * 100.0,
        f1_s=f1_s * 100.0,
        precision_ns=p_ns * 100.0,
        recall_ns=r_ns * 100.0,
        f1_ns=f1_ns * 100.0,
        macro_f1=macro_f1 * 100.0,
    )


def _load_model(cuda: bool):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    if cuda and torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def _batch_score(model, tokenizer, premise_list, hyp_list, batch_size=8, cuda=False):
    """Return (p_ent, p_neu, p_con) tuples for each (premise, hyp) pair."""
    results = []
    with torch.inference_mode():
        for start in range(0, len(premise_list), batch_size):
            batch_p = premise_list[start:start + batch_size]
            batch_h = hyp_list[start:start + batch_size]
            inputs = tokenizer(
                batch_p, batch_h,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            if cuda and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            for row in probs:
                results.append((row[0].item(), row[1].item(), row[2].item()))
    return results


# ===========================================================================
# A. PRUNING ABLATION
# ===========================================================================

def _extract_candidates_full(fact, passages, overlap_checker, use_title: bool,
                              use_windows: bool, prune: bool,
                              prune_threshold: float, max_candidates: int):
    """
    Re-implementation of EntailmentFactChecker._extract_candidates with
    controllable title / window / pruning flags.
    Returns (candidate_list, n_before_pruning)
    """
    _ensure_punkt()
    fact_tokens = overlap_checker._preprocess(fact)
    if not fact_tokens:
        return [], 0

    candidates = []
    for passage in passages:
        title = passage.get("title", "").strip() if use_title else ""
        raw_text = passage.get("text", "")
        if not raw_text:
            continue
        clean_text = WordRecallThresholdFactChecker._clean_passage_text(raw_text)
        sentences = nltk.sent_tokenize(clean_text)

        for i, sent in enumerate(sentences):
            s_clean = sent.strip()
            if len(s_clean.split()) < 3:
                continue

            premise = f"{title}. {s_clean}" if title else s_clean
            prem_tokens = overlap_checker._preprocess(premise)
            overlap = overlap_checker._recall(fact_tokens, prem_tokens)
            candidates.append((overlap, premise))

            if use_windows and i > 0:
                win_text = f"{sentences[i-1].strip()} {s_clean}"
                win_premise = f"{title}. {win_text}" if title else win_text
                win_tokens = overlap_checker._preprocess(win_premise)
                win_ov = overlap_checker._recall(fact_tokens, win_tokens)
                candidates.append((win_ov, win_premise))

    n_before = len(candidates)
    if not prune:
        # Return all, sorted by overlap descending, capped at max_candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        kept = [c[1] for c in candidates[:max_candidates]]
        return kept, n_before

    surviving = [c for c in candidates if c[0] >= prune_threshold]
    surviving.sort(key=lambda x: x[0], reverse=True)
    kept = [c[1] for c in surviving[:max_candidates]]
    return kept, n_before


def run_pruning_ablation(examples, cuda: bool = False,
                         prune_threshold: float = 0.2,
                         max_candidates: int = 15,
                         entailment_threshold: float = 0.50):
    """
    Compare two settings:
      - use_pruning=False: pass ALL candidates to DeBERTa (capped at max_candidates)
      - use_pruning=True:  filter by Recall >= prune_threshold first

    Metric used by the sieve: unigram token Recall
      (WordRecallThresholdFactChecker._recall)

    Reports:
      accuracy, macro-F1, runtime,
      avg_candidates_before_pruning, avg_candidates_after_pruning,
      avg_candidates_removed
    """
    print("\n" + "=" * 72)
    print("A.  PRUNING ABLATION  (Entailment candidate sieve)")
    print("    Similarity metric: unigram token Recall (stemmed, no stopwords)")
    print(f"    Prune threshold : >= {prune_threshold}")
    print(f"    Max candidates  : {max_candidates}")
    print("=" * 72)

    model, tokenizer = _load_model(cuda)
    overlap_checker = WordRecallThresholdFactChecker()

    results = {}
    for use_pruning in [False, True]:
        label = "with_pruning" if use_pruning else "no_pruning"
        preds, gold_labels = [], []
        candidates_before_all, candidates_after_all = [], []

        t0 = time.time()
        for ex in examples:
            gold = "NS" if ex.label == "IR" else ex.label

            # -- collect candidates ----------------------------------------
            kept, n_before = _extract_candidates_full(
                ex.fact, ex.passages, overlap_checker,
                use_title=True, use_windows=True,
                prune=use_pruning,
                prune_threshold=prune_threshold,
                max_candidates=max_candidates,
            )
            candidates_before_all.append(n_before)
            candidates_after_all.append(len(kept))

            # -- NLI inference on the kept candidates ----------------------
            if not kept:
                preds.append("NS")
                gold_labels.append(gold)
                continue

            premise_list = kept
            hyp_list = [ex.fact] * len(kept)
            probs = _batch_score(model, tokenizer, premise_list, hyp_list,
                                 cuda=cuda)

            max_entailment = 0.0
            for p_ent, p_neu, p_con in probs:
                if p_ent > p_con and p_ent > max_entailment:
                    max_entailment = p_ent
                if max_entailment >= 0.95:
                    break

            pred = "S" if max_entailment >= entailment_threshold else "NS"
            preds.append(pred)
            gold_labels.append(gold)

        elapsed = time.time() - t0
        metrics = _summarize(preds, gold_labels)
        avg_before = float(np.mean(candidates_before_all)) if candidates_before_all else 0.0
        avg_after = float(np.mean(candidates_after_all)) if candidates_after_all else 0.0
        avg_removed = avg_before - avg_after

        metrics["runtime_s"] = round(elapsed, 2)
        metrics["runtime_per_claim_s"] = round(elapsed / len(examples), 3) if examples else 0.0
        metrics["avg_candidates_before_pruning"] = round(avg_before, 2)
        metrics["avg_candidates_after_pruning"] = round(avg_after, 2)
        metrics["avg_candidates_removed"] = round(avg_removed, 2)
        metrics["similarity_metric"] = "unigram_recall"
        metrics["prune_threshold"] = prune_threshold
        results[label] = metrics

        tag = "NO PRUNING" if not use_pruning else "WITH PRUNING"
        print(f"\n  [{tag}]")
        print(f"    Accuracy   : {metrics['accuracy']:.2f}%  ({metrics['correct']}/{metrics['total']})")
        print(f"    Macro-F1   : {metrics['macro_f1']:.2f}%")
        print(f"    Runtime    : {elapsed:.1f} s  ({metrics['runtime_per_claim_s']:.3f} s/claim)")
        print(f"    Avg cands before pruning : {avg_before:.1f}")
        print(f"    Avg cands after  pruning : {avg_after:.1f}")
        print(f"    Avg cands removed        : {avg_removed:.1f}")
        print(f"    Similarity metric        : Recall  (unigram token recall, stemmed, stopwords removed)")

    delta_acc = results["with_pruning"]["accuracy"] - results["no_pruning"]["accuracy"]
    delta_f1  = results["with_pruning"]["macro_f1"]  - results["no_pruning"]["macro_f1"]
    delta_rt  = results["no_pruning"]["runtime_s"]    - results["with_pruning"]["runtime_s"]
    print(f"\n  Pruning effect:")
    print(f"    Accuracy delta : {delta_acc:+.2f} pp")
    print(f"    Macro-F1 delta : {delta_f1:+.2f} pp")
    print(f"    Speedup        : {delta_rt:+.1f} s faster with pruning")
    print("=" * 72 + "\n")

    results["_meta"] = {
        "similarity_metric": "unigram_recall",
        "similarity_metric_description": (
            "Token-level recall: |fact_tokens ∩ candidate_tokens| / |fact_tokens|. "
            "Uses Porter stemming and English stopword removal, as implemented in "
            "WordRecallThresholdFactChecker._recall."
        ),
        "delta_accuracy_pp": round(delta_acc, 2),
        "delta_macro_f1_pp": round(delta_f1, 2),
        "speedup_s": round(delta_rt, 2),
    }
    return results


# ===========================================================================
# B. PREPROCESSING ABLATION
# ===========================================================================

def _extract_for_condition(fact, passages, overlap_checker,
                           granularity: str, use_title: bool,
                           prune: bool = True,
                           prune_threshold: float = 0.2,
                           max_candidates: int = 15):
    """
    granularity: 'whole_passage' | 'single_sentence' | 'sentence_plus_window'
    """
    _ensure_punkt()
    fact_tokens = overlap_checker._preprocess(fact)
    if not fact_tokens:
        return [], 0

    if granularity == "whole_passage":
        candidates = []
        for passage in passages:
            title = passage.get("title", "").strip() if use_title else ""
            raw_text = passage.get("text", "")
            if not raw_text:
                continue
            clean_text = WordRecallThresholdFactChecker._clean_passage_text(raw_text)
            premise = f"{title}. {clean_text}" if title else clean_text
            prem_tokens = overlap_checker._preprocess(premise)
            overlap = overlap_checker._recall(fact_tokens, prem_tokens)
            candidates.append((overlap, premise))

    else:
        use_windows = (granularity == "sentence_plus_window")
        candidates = []
        for passage in passages:
            title = passage.get("title", "").strip() if use_title else ""
            raw_text = passage.get("text", "")
            if not raw_text:
                continue
            clean_text = WordRecallThresholdFactChecker._clean_passage_text(raw_text)
            sentences = nltk.sent_tokenize(clean_text)
            for i, sent in enumerate(sentences):
                s_clean = sent.strip()
                if len(s_clean.split()) < 3:
                    continue
                premise = f"{title}. {s_clean}" if title else s_clean
                prem_tokens = overlap_checker._preprocess(premise)
                overlap = overlap_checker._recall(fact_tokens, prem_tokens)
                candidates.append((overlap, premise))
                if use_windows and i > 0:
                    win_text = f"{sentences[i-1].strip()} {s_clean}"
                    win_premise = f"{title}. {win_text}" if title else win_text
                    win_tokens = overlap_checker._preprocess(win_premise)
                    win_ov = overlap_checker._recall(fact_tokens, win_tokens)
                    candidates.append((win_ov, win_premise))

    n_before = len(candidates)
    if not prune:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:max_candidates]], n_before

    surviving = [c for c in candidates if c[0] >= prune_threshold]
    surviving.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in surviving[:max_candidates]], n_before


def run_preprocessing_ablation(examples, cuda: bool = False,
                               prune_threshold: float = 0.2,
                               max_candidates: int = 15,
                               entailment_threshold: float = 0.50):
    """
    Eight conditions: 3 granularities × 2 title settings + 2 no-title variants
    Actually: (whole_passage, single_sentence, sentence_plus_window) × (no_title, with_title)

    For whole_passage: pruning is applied at passage level (each passage is one candidate).
    """
    print("\n" + "=" * 72)
    print("B.  PREPROCESSING ABLATION")
    print("    Granularity: whole_passage | single_sentence | sentence+window")
    print("    Title:       with_title | no_title")
    print("    NLI model:   DeBERTa-v3-base-mnli-fever-anli")
    print("=" * 72)

    model, tokenizer = _load_model(cuda)
    overlap_checker = WordRecallThresholdFactChecker()

    conditions = [
        # (granularity,              use_title, label)
        ("whole_passage",            False, "whole_passage__no_title"),
        ("whole_passage",            True,  "whole_passage__with_title"),
        ("single_sentence",          False, "single_sentence__no_title"),
        ("single_sentence",          True,  "single_sentence__with_title"),
        ("sentence_plus_window",     False, "sentence_window__no_title"),
        ("sentence_plus_window",     True,  "sentence_window__with_title"),  # ← final system
    ]

    results = {}
    for granularity, use_title, cond_label in conditions:
        preds, gold_labels = [], []
        cands_per_claim = []

        t0 = time.time()
        for ex in examples:
            gold = "NS" if ex.label == "IR" else ex.label

            kept, n_before = _extract_for_condition(
                ex.fact, ex.passages, overlap_checker,
                granularity=granularity,
                use_title=use_title,
                prune=True,
                prune_threshold=prune_threshold,
                max_candidates=max_candidates,
            )
            cands_per_claim.append(len(kept))

            if not kept:
                preds.append("NS")
                gold_labels.append(gold)
                continue

            probs = _batch_score(model, tokenizer, kept, [ex.fact] * len(kept), cuda=cuda)
            max_entailment = 0.0
            for p_ent, p_neu, p_con in probs:
                if p_ent > p_con and p_ent > max_entailment:
                    max_entailment = p_ent
                if max_entailment >= 0.95:
                    break

            pred = "S" if max_entailment >= entailment_threshold else "NS"
            preds.append(pred)
            gold_labels.append(gold)

        elapsed = time.time() - t0
        metrics = _summarize(preds, gold_labels)
        metrics["runtime_s"] = round(elapsed, 2)
        metrics["runtime_per_claim_s"] = round(elapsed / len(examples), 3) if examples else 0.0
        metrics["avg_candidates"] = round(float(np.mean(cands_per_claim)), 2) if cands_per_claim else 0.0
        metrics["granularity"] = granularity
        metrics["use_title"] = use_title
        results[cond_label] = metrics

        is_final = (granularity == "sentence_plus_window" and use_title)
        tag = " ← FINAL SYSTEM" if is_final else ""
        print(f"\n  [{cond_label}]{tag}")
        print(f"    Accuracy : {metrics['accuracy']:.2f}%  ({metrics['correct']}/{metrics['total']})")
        print(f"    Macro-F1 : {metrics['macro_f1']:.2f}%")
        print(f"    F1 (S)   : {metrics['f1_s']:.2f}%")
        print(f"    Runtime  : {elapsed:.1f} s  ({metrics['runtime_per_claim_s']:.3f} s/claim)")
        print(f"    Avg cands: {metrics['avg_candidates']:.1f}")

    print("\n" + "=" * 72)
    print("PREPROCESSING ABLATION SUMMARY TABLE")
    print(f"{'Condition':<40} | {'Acc':>6} | {'MacroF1':>7} | {'F1(S)':>6} | {'Avg Cands':>9} | {'Runtime':>8}")
    print("-" * 90)
    for cond_label, metrics in results.items():
        final_tag = " *" if cond_label == "sentence_window__with_title" else "  "
        print(
            f"  {cond_label:<40}{final_tag} | "
            f"{metrics['accuracy']:>6.2f} | "
            f"{metrics['macro_f1']:>7.2f} | "
            f"{metrics['f1_s']:>6.2f} | "
            f"{metrics['avg_candidates']:>9.1f} | "
            f"{metrics['runtime_s']:>7.1f}s"
        )
    print("  (* = final system)")
    print("=" * 72 + "\n")

    return results


# ===========================================================================
# PLOTS
# ===========================================================================

def _plot_pruning_ablation(results: dict, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available; skipping pruning ablation plot.")
        return

    no_prune = results["no_pruning"]
    with_prune = results["with_pruning"]

    metric_keys = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("f1_s", "F1 (S)"),
        ("recall_s", "Recall (S)"),
        ("precision_s", "Precision (S)"),
    ]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, (ax_bars, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Pruning Ablation: No-Pruning Baseline vs. Recall-Sieve Pruning\n"
        f"(DeBERTa, N={no_prune['total']}, prune τ={results['_meta'].get('prune_threshold', 0.2)}, "
        f"similarity metric: Unigram Token Recall)",
        fontsize=13, fontweight="bold",
    )

    colors_no = "#c0392b"
    colors_yes = "#2c7a3f"

    bars_no = ax_bars.bar(x - width / 2,
                          [no_prune[k] for k, _ in metric_keys],
                          width, label="No Pruning (baseline)",
                          color=colors_no, edgecolor="black", alpha=0.9)
    bars_yes = ax_bars.bar(x + width / 2,
                           [with_prune[k] for k, _ in metric_keys],
                           width, label="With Pruning (Recall sieve)",
                           color=colors_yes, edgecolor="black", alpha=0.9)
    for bar in list(bars_no) + list(bars_yes):
        ax_bars.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f"{bar.get_height():.1f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([label for _, label in metric_keys], fontsize=10)
    ax_bars.set_ylim(0, 100)
    ax_bars.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax_bars.set_title("Metric Comparison", fontsize=12, fontweight="bold")
    ax_bars.legend(fontsize=10)
    ax_bars.grid(axis="y", linestyle=":", alpha=0.6)

    # Right panel: candidate stats + runtime
    ax_right.axis("off")
    rows = [
        ["", "No Pruning", "With Pruning", "Δ"],
        ["Accuracy (%)",
         f"{no_prune['accuracy']:.2f}",
         f"{with_prune['accuracy']:.2f}",
         f"{with_prune['accuracy'] - no_prune['accuracy']:+.2f}"],
        ["Macro F1 (%)",
         f"{no_prune['macro_f1']:.2f}",
         f"{with_prune['macro_f1']:.2f}",
         f"{with_prune['macro_f1'] - no_prune['macro_f1']:+.2f}"],
        ["Runtime (s)",
         f"{no_prune['runtime_s']:.1f}",
         f"{with_prune['runtime_s']:.1f}",
         f"{with_prune['runtime_s'] - no_prune['runtime_s']:+.1f}"],
        ["Avg cands (before)",
         f"{no_prune['avg_candidates_before_pruning']:.1f}",
         f"{with_prune['avg_candidates_before_pruning']:.1f}",
         "—"],
        ["Avg cands (after)",
         f"{no_prune['avg_candidates_after_pruning']:.1f}",
         f"{with_prune['avg_candidates_after_pruning']:.1f}",
         "—"],
        ["Avg cands removed",
         f"{no_prune['avg_candidates_removed']:.1f}",
         f"{with_prune['avg_candidates_removed']:.1f}",
         "—"],
        ["Similarity metric", "Recall", "Recall", "—"],
    ]
    table = ax_right.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.05, 0.96, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    header_color = "#1c2833"
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f2f3f4" if r % 2 == 0 else "#eaecee")
            cell.set_text_props(fontweight="normal")
    ax_right.set_title("Candidate & Timing Summary", fontsize=12, fontweight="bold", pad=10)

    path = os.path.join(output_dir, "pruning_ablation.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Pruning ablation plot saved to: {path}")


def _plot_preprocessing_ablation(results: dict, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available; skipping preprocessing ablation plot.")
        return

    # Ordered conditions
    ordered_keys = [
        "whole_passage__no_title",
        "whole_passage__with_title",
        "single_sentence__no_title",
        "single_sentence__with_title",
        "sentence_window__no_title",
        "sentence_window__with_title",
    ]
    display_names = {
        "whole_passage__no_title":        "Whole passage\n(no title)",
        "whole_passage__with_title":       "Whole passage\n(+ title)",
        "single_sentence__no_title":      "Single sentence\n(no title)",
        "single_sentence__with_title":     "Single sentence\n(+ title)",
        "sentence_window__no_title":      "Sent + window\n(no title)",
        "sentence_window__with_title":     "Sent + window\n(+ title) ★",
    }
    colors = {
        "whole_passage__no_title":        "#7f8c8d",
        "whole_passage__with_title":       "#95a5a6",
        "single_sentence__no_title":      "#2980b9",
        "single_sentence__with_title":     "#1abc9c",
        "sentence_window__no_title":      "#8e44ad",
        "sentence_window__with_title":     "#c0392b",  # final system highlighted
    }

    metrics_to_plot = [
        ("accuracy", "Accuracy (%)"),
        ("macro_f1", "Macro F1 (%)"),
        ("f1_s", "F1 (S) (%)"),
        ("avg_candidates", "Avg Candidates"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Preprocessing Ablation: Granularity × Title-Prefix Effect on DeBERTa Fact-Checking\n"
        f"(★ = final system: sentence + 2-sentence window + title prefix, N={results[ordered_keys[0]]['total']})",
        fontsize=13, fontweight="bold",
    )

    for ax, (metric_key, metric_label) in zip(axes.flat, metrics_to_plot):
        vals = [results[k].get(metric_key, 0.0) for k in ordered_keys]
        bar_colors = [colors[k] for k in ordered_keys]
        bars = ax.bar(range(len(ordered_keys)), vals, color=bar_colors,
                      edgecolor="black", alpha=0.90)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(ordered_keys)))
        ax.set_xticklabels([display_names[k] for k in ordered_keys],
                           fontsize=8.5, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=10, fontweight="bold")
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        # Highlight final system bar
        final_idx = ordered_keys.index("sentence_window__with_title")
        bars[final_idx].set_edgecolor("#ff0000")
        bars[final_idx].set_linewidth(2.5)

    path = os.path.join(output_dir, "preprocessing_ablation_full.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Preprocessing ablation plot saved to: {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Use CUDA/GPU if available")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap the number of examples (for quick tests)")
    parser.add_argument("--prune_threshold", type=float, default=0.2)
    parser.add_argument("--max_candidates", type=int, default=15)
    parser.add_argument("--entailment_threshold", type=float, default=0.50)
    parser.add_argument("--skip_pruning", action="store_true",
                        help="Skip section A and only run preprocessing ablation")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip section B and only run pruning ablation")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    passages_dict = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(DEV_PATH, passages_dict)
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"Loaded {len(examples)} examples.")

    kw = dict(
        prune_threshold=args.prune_threshold,
        max_candidates=args.max_candidates,
        entailment_threshold=args.entailment_threshold,
        cuda=args.cuda,
    )

    # --- A: Pruning ablation ---
    if not args.skip_pruning:
        pruning_results = run_pruning_ablation(examples, **kw)
        pruning_path = os.path.join(OUTPUT_DIR, "pruning_ablation.json")
        with open(pruning_path, "w", encoding="utf-8") as fh:
            json.dump(pruning_results, fh, indent=2)
        print(f"Pruning ablation JSON saved to: {pruning_path}")
        _plot_pruning_ablation(pruning_results, OUTPUT_DIR)

    # --- B: Preprocessing ablation ---
    if not args.skip_preprocessing:
        prep_results = run_preprocessing_ablation(examples, **kw)
        prep_path = os.path.join(OUTPUT_DIR, "preprocessing_ablation_full.json")
        with open(prep_path, "w", encoding="utf-8") as fh:
            json.dump(prep_results, fh, indent=2)
        print(f"Preprocessing ablation JSON saved to: {prep_path}")
        _plot_preprocessing_ablation(prep_results, OUTPUT_DIR)

    print("\nAll ablation experiments complete.")


if __name__ == "__main__":
    main()

