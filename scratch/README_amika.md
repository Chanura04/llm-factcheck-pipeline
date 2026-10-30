# Fact-Checking Pipeline: Stemming vs. Lemmatization

**Author:** Amika Alankara

## Overview

This project extends an existing fact-checking system by introducing a second text-preprocessing approach based on **lemmatization**. The objective is to evaluate whether lemmatization improves classification accuracy compared with the original **stemming** approach.

The updated pipeline supports both methods and evaluates them side by side using the development (dev) dataset. A single command runs both experiments, displays their evaluation metrics, and identifies the method that achieves the higher accuracy.

## Objectives

- Implement a lemmatization-based preprocessing pipeline alongside the existing stemming pipeline.
- Preserve the original word-overlap and recall-based classification logic.
- Evaluate both preprocessing methods using the same development dataset.
- Compare their classification performance using accuracy, precision, recall, and F1-score.
- Investigate disagreements between the two methods through additional diagnostic scripts.

## Implementation

### 1. `factcheck.py`

The fact-checking implementation was extended to support both stemming and lemmatization.

**New preprocessing method: `_preprocess_with_lemm()`**

A new preprocessing method was introduced based on the original `_preprocess()` implementation. The existing preprocessing steps are retained, while the final token transformation is changed from stemming to lemmatization.

| Method | Token transformation |
|---|---|
| Stemming | `self.stemmer.stem(token)` |
| Lemmatization | `self.lemmatizer.lemmatize(token, pos='v')` |

The lemmatizer uses `pos='v'` to specify the verb part of speech. This allows it to reduce verb forms to their base forms. For example, *running* can be transformed into *run*.

**Configurable preprocessing**

A new `use_lemmatization` boolean flag was added to the `WordRecallThresholdFactChecker` constructor.

This allows the classifier to select the appropriate preprocessing method without changing the underlying word-overlap or recall calculation logic.

**Updated prediction logic**

The `predict()` method now selects the preprocessing approach based on the configured flag:

- `False` — Uses the original stemming pipeline.
- `True` — Uses the new lemmatization pipeline.

This keeps both approaches within the same classification framework and makes their results easier to compare.

### 2. `factchecking_main.py`

The main execution script was updated to automate the comparison.

**New function: `compare_preprocessing_methods()`**

This function evaluates the two preprocessing approaches independently:

1. Runs the classifier using stemming.
2. Records the resulting development-set accuracy.
3. Runs the classifier using lemmatization.
4. Records the resulting development-set accuracy.
5. Calculates the accuracy difference in percentage points.
6. Displays a final comparison and identifies the higher-performing method.

**Updated evaluation function: `predict_two_classes()`**

A `return_accuracy` parameter was introduced to allow the evaluation function to return the calculated accuracy. This makes it possible for the comparison function to capture and compare the results programmatically.

**Updated command-line dispatch**

The `word_overlap` execution mode now automatically invokes `compare_preprocessing_methods()` instead of running only one evaluation.

## How to Run

Execute the following command from the project root directory:

```bash
python factchecking_main.py --mode word_overlap
```

No separate command is required to run the two experiments individually.

## Expected Output

The terminal displays the results of both preprocessing experiments.

The output includes:

- **Progress indicators** for the stemming and lemmatization evaluations.
- **Confusion matrices** for both methods.
- **Classification metrics**, including precision, recall, and F1-score.
- **Accuracy comparison** showing the performance of each approach.
- **Final verdict** identifying the method with the higher development-set accuracy.

The final comparison follows this general format:

```text
==================================================
                 FINAL RESULT
==================================================

Stemming Accuracy       : [Calculated Accuracy]
Lemmatization Accuracy  : [Calculated Accuracy]

Accuracy Difference     : [Difference] percentage points

Winner                  : [Higher-Performing Method]

==================================================
```

The actual values are calculated at runtime from the evaluation results.

## Diagnostic and Analysis Scripts

Additional scripts were created in the `scratch/` directory to verify the implementation and investigate the behaviour of the two preprocessing methods.

| Script | Purpose |
|---|---|
| `test_lem.py` | Tests the lemmatizer using sample sentences to verify that token transformations work as expected. |
| `probe_real_data.py` | Checks whether lemmatization produces changes in tokens from the actual dataset. |
| `sweep_stem_vs_lem.py` | Compares the performance of stemming and lemmatization across different decision thresholds, from 0.50 to 0.70. |
| `verify_code_diff.py` | Uses Python's `inspect` module to examine the two preprocessing methods and verify their implementation differences. |
| `diff_analysis.py` | Identifies instances where the two methods produce different predictions and saves them for further investigation. |

### Disagreement Analysis

The `diff_analysis.py` script provides additional insight beyond the overall accuracy comparison.

It identifies facts for which the stemming and lemmatization approaches produce different predictions. These disagreements are exported to:

```text
results/day3_disagreements.tsv
```

This file can be used for qualitative error analysis to investigate why the two preprocessing methods behave differently on particular examples.

## Summary

The original fact-checking pipeline has been extended to support a direct comparison between stemming and lemmatization without changing its core classification logic.

The implementation provides:

- Two interchangeable preprocessing approaches.
- Automated side-by-side evaluation.
- Comparison of classification metrics and development-set accuracy.
- Threshold-based performance analysis.
- Diagnostic tools for investigating preprocessing differences and prediction disagreements.

The final evaluation determines which preprocessing method performs better on the development dataset, while the additional diagnostic scripts help explain the differences in their behaviour.

---

**Author:** Amika Alankara