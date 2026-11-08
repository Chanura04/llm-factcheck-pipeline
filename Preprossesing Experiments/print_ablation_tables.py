"""
print_ablation_tables.py
========================
Reads the JSON outputs produced by ablation_experiments.py and prints
ready-to-paste report tables for:

  A. Pruning ablation  (§2.2 of your report)
  B. Preprocessing ablation  (§2.1.9–2.1.10)

Usage:
    python print_ablation_tables.py

Expects (in eda_plots/):
    pruning_ablation.json
    preprocessing_ablation_full.json
"""

import json
import os
import sys

PLOT_DIR = "eda_plots"
PRUNING_JSON  = os.path.join(PLOT_DIR, "pruning_ablation.json")
PREP_JSON     = os.path.join(PLOT_DIR, "preprocessing_ablation_full.json")


def _load(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        print("        Run:  python ablation_experiments.py")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_pruning_table(d: dict):
    no  = d["no_pruning"]
    yes = d["with_pruning"]
    meta = d.get("_meta", {})

    print()
    print("=" * 72)
    print("TABLE A — PRUNING ABLATION  (Entailment §2.2)")
    print(f"  Similarity metric : {meta.get('similarity_metric_description', 'unigram_recall')}")
    print(f"  Prune threshold   : recall >= {yes.get('prune_threshold', 0.2)}")
    print(f"  NLI model         : DeBERTa-v3-base-mnli-fever-anli")
    print(f"  N examples        : {no['total']}")
    print("=" * 72)

    # Row header
    h = f"  {'Metric':<30}  {'No Pruning':>12}  {'With Pruning':>13}  {'Δ':>8}"
    print(h)
    print("  " + "-" * 68)

    def row(label, k_no, k_yes, fmt=".2f", unit="%"):
        v_no  = no.get(k_no,  0.0)
        v_yes = yes.get(k_yes, 0.0)
        delta = v_yes - v_no
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<30}  {v_no:>11{fmt}}{unit}  {v_yes:>12{fmt}}{unit}  {sign}{delta:>7{fmt}}{unit}")

    row("Accuracy",              "accuracy",   "accuracy")
    row("Macro F1",              "macro_f1",   "macro_f1")
    row("F1 (Supported)",        "f1_s",       "f1_s")
    row("Precision (S)",         "precision_s","precision_s")
    row("Recall (S)",            "recall_s",   "recall_s")

    # Non-percentage rows
    def row2(label, k_no, k_yes, fmt=".1f", unit=""):
        v_no  = no.get(k_no,  0.0)
        v_yes = yes.get(k_yes, 0.0)
        delta = v_yes - v_no
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<30}  {v_no:>12{fmt}}{unit}  {v_yes:>12{fmt}}{unit}  {sign}{delta:>7{fmt}}{unit}")

    print("  " + "-" * 68)
    row2("Avg candidates (before)", "avg_candidates_before_pruning", "avg_candidates_before_pruning")
    row2("Avg candidates (after)",  "avg_candidates_after_pruning",  "avg_candidates_after_pruning")
    row2("Avg candidates removed",  "avg_candidates_removed",        "avg_candidates_removed")
    row2("Total runtime (s)",       "runtime_s",   "runtime_s")
    row2("Runtime per claim (s)",   "runtime_per_claim_s", "runtime_per_claim_s", fmt=".3f")

    print("  " + "=" * 68)
    print(f"  Accuracy speedup   : {meta.get('speedup_s', 0):+.1f} s  ({no['runtime_per_claim_s']:.3f} s/claim → {yes['runtime_per_claim_s']:.3f} s/claim)")
    print(f"  Accuracy delta     : {meta.get('delta_accuracy_pp', 0):+.2f} pp")
    print(f"  Macro F1 delta     : {meta.get('delta_macro_f1_pp', 0):+.2f} pp")
    print("=" * 72)


def print_preprocessing_table(d: dict):
    ordered = [
        ("whole_passage__no_title",       "Whole passage, no title"),
        ("whole_passage__with_title",      "Whole passage, + title"),
        ("single_sentence__no_title",     "Single sentence, no title"),
        ("single_sentence__with_title",    "Single sentence, + title"),
        ("sentence_window__no_title",     "Sent + 2-sent window, no title"),
        ("sentence_window__with_title",    "Sent + 2-sent window, + title  ★"),
    ]

    print()
    print("=" * 100)
    print("TABLE B — PREPROCESSING ABLATION  (§2.1.9–§2.1.10)")
    print("  NLI model  : DeBERTa-v3-base-mnli-fever-anli")
    print("  Pruning    : Recall-sieve, τ = 0.20  (applied in all conditions)")
    n = d[ordered[0][0]]["total"] if ordered[0][0] in d else "?"
    print(f"  N examples : {n}")
    print("  ★ = final system")
    print("=" * 100)

    hdr = f"  {'Condition':<40} {'Acc':>7} {'MacroF1':>8} {'F1-S':>7} {'Prec-S':>8} {'Rec-S':>7} {'AvgCands':>9} {'RT(s)':>7}"
    print(hdr)
    print("  " + "-" * 96)

    for key, label in ordered:
        if key not in d:
            print(f"  {label:<40}   MISSING")
            continue
        m = d[key]
        print(
            f"  {label:<40}"
            f" {m['accuracy']:>7.2f}"
            f" {m['macro_f1']:>8.2f}"
            f" {m['f1_s']:>7.2f}"
            f" {m['precision_s']:>8.2f}"
            f" {m['recall_s']:>7.2f}"
            f" {m['avg_candidates']:>9.1f}"
            f" {m['runtime_s']:>7.1f}"
        )
    print("  " + "=" * 96)

    # Key deltas for the report
    if "sentence_window__with_title" in d and "sentence_window__no_title" in d:
        final   = d["sentence_window__with_title"]
        no_title= d["sentence_window__no_title"]
        da = final["accuracy"] - no_title["accuracy"]
        df = final["macro_f1"] - no_title["macro_f1"]
        print(f"\n  Title-prefix effect (window conditions):")
        print(f"    Accuracy : {no_title['accuracy']:.2f}% → {final['accuracy']:.2f}%  ({da:+.2f} pp)")
        print(f"    Macro F1 : {no_title['macro_f1']:.2f}% → {final['macro_f1']:.2f}%  ({df:+.2f} pp)")

    if "whole_passage__with_title" in d and "single_sentence__with_title" in d and "sentence_window__with_title" in d:
        wp = d["whole_passage__with_title"]
        ss = d["single_sentence__with_title"]
        sw = d["sentence_window__with_title"]
        print(f"\n  Granularity effect (all with title):")
        print(f"    Whole passage    : acc={wp['accuracy']:.2f}%  macro-F1={wp['macro_f1']:.2f}%")
        print(f"    Single sentence  : acc={ss['accuracy']:.2f}%  macro-F1={ss['macro_f1']:.2f}%")
        print(f"    + window (final) : acc={sw['accuracy']:.2f}%  macro-F1={sw['macro_f1']:.2f}%")
    print("=" * 100)


def main():
    pruning = _load(PRUNING_JSON)
    prep    = _load(PREP_JSON)

    any_missing = False
    if pruning is None or prep is None:
        any_missing = True

    if pruning:
        print_pruning_table(pruning)

    if prep:
        print_preprocessing_table(prep)

    if any_missing:
        print("\n[ACTION REQUIRED]")
        print("  Run the full experiment first:")
        print("    python ablation_experiments.py")
        print("  Then re-run this script to get the final tables.")
        sys.exit(1)
    else:
        print("\n[OK] Both tables printed. Copy them into your report.")


if __name__ == "__main__":
    main()

