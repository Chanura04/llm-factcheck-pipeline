# Fact-Checking Pipeline: Stemming vs. Lemmatization

A comparative evaluation of **Porter Stemming** and **WordNet Lemmatization** in an NLP-based fact-checking system.

This implementation introduces a parallel preprocessing pipeline that evaluates both text-normalization techniques against the same development dataset, making it possible to determine which approach delivers better classification accuracy.

---

## 🎯 Objective

The primary objective is to investigate whether **lemmatization can outperform the existing stemming approach** in the fact-checking pipeline.

Two preprocessing methods are evaluated side by side:

- **Stemming:** Uses the Porter Stemmer to reduce words to their root forms.
- **Lemmatization:** Uses the WordNet Lemmatizer to transform words into their meaningful dictionary forms.

Rather than relying on theoretical assumptions, both approaches are evaluated using the same dataset and classification logic.

A single command executes the comparison and produces a final verdict identifying the better-performing method based on development-set accuracy.

---

## 🔧 Implementation Overview

### 1. `factcheck.py`

The main implementation introduces three key modifications.

#### A. New Lemmatization Pipeline

Added a new method:

```python
_preprocess_with_lemm()
```

This method is an exact copy of the original `_preprocess()` implementation, with only the final word-normalization operation changed.

| Existing Pipeline | New Pipeline |
|---|---|
| `self.stemmer.stem(token)` | `self.lemmatizer.lemmatize(token, pos='v')` |
| Porter Stemmer | WordNet Lemmatizer |

**Why `pos='v'`?**

By default, the WordNet Lemmatizer assumes that words are nouns. Consequently, it may fail to normalize verbs correctly.

For example:

| Input | Without POS | With `pos='v'` |
|---|---|---|
| running | running | run |
| playing | playing | play |
| studies | study | study |

Specifying `pos='v'` instructs the lemmatizer to process tokens as verbs, providing a more meaningful comparison with the existing stemming approach.

#### B. Configurable Preprocessing

Introduced a `use_lemmatization` boolean flag in the `WordRecallThresholdFactChecker` class constructor (`__init__`).

This provides a way to select the preprocessing strategy without duplicating the classification implementation.

#### C. Evaluation Monkey Patch

Added:

```python
_patch_main_for_comparison()
```

This function intercepts the evaluation loop in `factchecking_main.py` to execute both preprocessing strategies independently.

**Why use monkey patching?**

It allows the comparison to be integrated without modifying the original evaluation script, keeping the existing implementation intact.

### 2. `factchecking_main.py`

<kbd>UNCHANGED</kbd>

The original `factchecking_main.py` script remains untouched.

All comparison-related modifications are handled through `factcheck.py`, preserving the original evaluation entry point.

---

## 🚀 How to Run

Execute the following command from the project root directory:

```bash
python factchecking_main.py --mode word_overlap
```

No separate commands are required to evaluate the two approaches.

---

## 📊 Expected Output

The comparison produces the following results in the terminal:

**1. Progress Tracking**

Two separate progress bars indicate the evaluation of:
- Porter Stemming
- WordNet Lemmatization

**2. Classification Performance**

Each method produces its own confusion matrix, including:

- Precision
- Recall
- F1-score
- Classification results per class

**3. Final Comparison**

A consolidated `FINAL RESULT` section reports:

| Metric | Description |
|---|---|
| Stemming Accuracy | Accuracy achieved using Porter Stemming |
| Lemmatization Accuracy | Accuracy achieved using WordNet Lemmatization |
| Accuracy Difference | Difference between the two methods in percentage points |
| Final Verdict | Identifies the higher-performing preprocessing method |

The final verdict is determined automatically from the measured development-set accuracy.

---

## 🧪 Supporting Analysis Scripts

Additional experiments and verification scripts are maintained in the `scratch/` directory.

| Script | Purpose |
|---|---|
| `test_lem.py` | Performs sanity checks using sample sentences to verify that lemmatization works as expected. |
| `probe_real_data.py` | Confirms that the lemmatization pipeline produces actual token changes on the real dataset. |
| `sweep_stem_vs_lem.py` | Evaluates whether the winning method changes across different decision thresholds, ranging from 0.50 to 0.70. |
| `verify_code_diff.py` | Uses Python's `inspect` module to verify that only one line differs between the two preprocessing methods. |
| `diff_analysis.py` | Identifies facts where the two methods produce different predictions and exports them for qualitative error analysis. |

### Disagreement Analysis

The `diff_analysis.py` script generates:

```text
results/
└── final_disagreements.tsv
```

This file contains examples where stemming and lemmatization disagree, allowing further investigation into why one preprocessing method may perform better than the other.

---

## 🔍 Experimental Approach

The comparison follows a controlled evaluation strategy:

1. Maintain the original stemming pipeline as the baseline.
2. Introduce lemmatization with minimal changes to the preprocessing logic.
3. Evaluate both approaches using the same development dataset.
4. Compare classification metrics and overall accuracy.
5. Perform threshold-based experiments to examine the consistency of the results.
6. Investigate disagreements to understand differences in prediction behaviour.

Keeping the remaining preprocessing and evaluation logic consistent helps isolate the impact of the word-normalization technique.

---

## 🏁 Conclusion

This experiment provides an empirical comparison between stemming and lemmatization within the existing fact-checking system.

Instead of assuming that one normalization technique is inherently better, the implementation uses measured classification performance to determine which approach is more effective for the given dataset.

**The winning method is determined by the actual development-set evaluation results.**

---

<sub>Experiment: Stemming vs. Lemmatization | NLP Fact-Checking Pipeline</sub>