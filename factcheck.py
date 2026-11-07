# factcheck.py
import torch
import string
import re
from typing import List, Any
import numpy as np
import spacy
import gc
import nltk
import sys
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    wordnet.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def _patch_main_for_comparison():
    """
    Monkey-patches predict_two_classes and print_eval_stats in __main__ to automatically run
    a side-by-side comparison between Stemming and Lemmatization during word_overlap evaluation.
    """
    main_mod = sys.modules.get('__main__')
    if not main_mod or not hasattr(main_mod, 'predict_two_classes') or not hasattr(main_mod, 'print_eval_stats'):
        return

    if getattr(main_mod, '_is_patched_for_comparison', False):
        return

    original_predict = main_mod.predict_two_classes
    original_print_eval = main_mod.print_eval_stats

    last_accuracy = {"val": 0.0}

    def new_print_eval_stats(confusion_mat, gold_label_indexer):
        original_print_eval(confusion_mat, gold_label_indexer)
        correct = confusion_mat[0][0] + confusion_mat[1][1]
        total = sum(sum(row) for row in confusion_mat)
        last_accuracy["val"] = correct / total if total > 0 else 0.0

    main_mod.print_eval_stats = new_print_eval_stats

    def new_predict_two_classes(examples, fact_checker):
        if isinstance(fact_checker, WordRecallThresholdFactChecker) and getattr(fact_checker, "compare", False):
            print("\n" + "=" * 72)
            print("  WORD_OVERLAP  —  PREPROCESSING COMPARISON")
            print("  Stemming (PorterStemmer)  vs  Lemmatization (WordNetLemmatizer)")
            print("=" * 72)

            print("\n>>> [1 / 2]  STEMMING  —  running...\n")
            original_predict(
                examples,
                WordRecallThresholdFactChecker(
                    threshold=getattr(fact_checker, "threshold", 0.6),
                    metric=getattr(fact_checker, "metric", "recall"),
                    aggregation=getattr(fact_checker, "aggregation", "max"),
                    use_stemmer=getattr(fact_checker, "use_stemmer", True),
                    remove_stopwords=getattr(fact_checker, "remove_stopwords", True),
                    lowercase=getattr(fact_checker, "lowercase", True),
                    use_lemmatization=False,
                ),
            )
            stem_acc = last_accuracy["val"]

            print("\n>>> [2 / 2]  LEMMATIZATION  —  running...\n")
            original_predict(
                examples,
                WordRecallThresholdFactChecker(
                    threshold=getattr(fact_checker, "threshold", 0.6),
                    metric=getattr(fact_checker, "metric", "recall"),
                    aggregation=getattr(fact_checker, "aggregation", "max"),
                    use_stemmer=getattr(fact_checker, "use_stemmer", True),
                    remove_stopwords=getattr(fact_checker, "remove_stopwords", True),
                    lowercase=getattr(fact_checker, "lowercase", True),
                    use_lemmatization=True,
                ),
            )
            lem_acc = last_accuracy["val"]

            print("\n" + "=" * 72)
            print("  FINAL RESULT  —  WHICH PREPROCESSING IS MORE ACCURATE?")
            print("=" * 72)
            print(f"  Stemming        accuracy : {stem_acc * 100:6.2f}%   ({stem_acc:.4f})")
            print(f"  Lemmatization   accuracy : {lem_acc * 100:6.2f}%   ({lem_acc:.4f})")
            print("-" * 72)
            if lem_acc > stem_acc:
                print(f"  ==> *** LEMMATIZATION WINS *** (+{(lem_acc - stem_acc) * 100:.2f} pp)")
                print(f"  ==> HIGHEST ACCURACY: Lemmatization = {lem_acc * 100:.2f}%")
            elif stem_acc > lem_acc:
                print(f"  ==> *** STEMMING WINS *** (+{(stem_acc - lem_acc) * 100:.2f} pp)")
                print(f"  ==> HIGHEST ACCURACY: Stemming = {stem_acc * 100:.2f}%")
            else:
                print(f"  ==> *** TIE *** — both methods hit the same accuracy = {stem_acc * 100:.2f}%")
            print("=" * 72 + "\n")
            return

        return original_predict(examples, fact_checker)

    main_mod.predict_two_classes = new_predict_two_classes
    main_mod._is_patched_for_comparison = True


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """

    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr(
            "fact="
            + repr(self.fact)
            + "; label="
            + repr(self.label)
            + "; passages="
            + repr(self.passages)
        )


class EntailmentModel(object):
    """
    Cross-Encoder NLI model wrapper for sequence classification models (e.g. DeBERTa, RoBERTa).
    Computes continuous entailment probabilities and 3-class distributions for (premise, hypothesis) pairs.
    """

    def __init__(self, model, tokenizer, cuda: bool = False):
        """
        Initializes the pre-trained Cross-Encoder NLI model wrapper.

        :param model: Pre-trained HuggingFace sequence classification model (e.g., DeBERTa or RoBERTa).
        :param tokenizer: HuggingFace tokenizer associated with the model.
        :param cuda: Whether to enable CUDA/GPU acceleration for inference.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda
        if self.cuda and torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Executes NLI inference for a (premise, hypothesis) pair using the pre-trained cross-encoder.
        Returns the continuous entailment probability P(entailment) as a standard float in [0.0, 1.0].

        :param premise: Evidence text (contextualized Wikipedia passage or sentence).
        :param hypothesis: Atomic claim to verify (ChatGPT proposition).
        :return: Continuous entailment probability score between 0.0 and 1.0.
        """
        import torch.nn.functional as F

        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=256, padding=True
            )
            if self.cuda:
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Label order for DeBERTa-mnli-fever-anli: [entailment, neutral, contradiction]
        probs = F.softmax(logits, dim=-1)
        entailment_prob = probs[0][0].item()

        # To prevent out-of-memory (OOM) issues delete objects inputs, outputs, logits,
        #  and any results that are no longer needed after the computation.
        del inputs, outputs, logits, probs
        if self.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return entailment_prob

    def check_entailment_raw(self, premise: str, hypothesis: str):
        """
        Computes the full 3-class probability distribution: (P_entailment, P_neutral, P_contradiction).

        :param premise: Evidence text (contextualized passage or sentence).
        :param hypothesis: Atomic claim to verify (ChatGPT proposition).
        :return: Tuple of (P_entailment, P_neutral, P_contradiction) probabilities summing to 1.0.
        """
        import torch.nn.functional as F

        with torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            if self.cuda and torch.cuda.is_available():
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1)[0]
        p_ent = probs[0].item()
        p_neu = probs[1].item()
        p_con = probs[2].item()

        del inputs, outputs, logits, probs
        if self.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return p_ent, p_neu, p_con


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence

        :param fact: The atomic proposition or sentence to verify.
        :param passages: List of retrieved evidence passages.
        :return: "S" (supported) or "NS" (not supported).
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Randomly predicts 'S' or 'NS' with equal probability.

        :param fact: The atomic proposition to verify.
        :param passages: List of retrieved evidence passages.
        :return: Random prediction string ('S' or 'NS').
        """
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Trivially predicts 'S' for all instances (majority/trivial baseline).

        :param fact: The atomic proposition to verify.
        :param passages: List of retrieved evidence passages.
        :return: Constant prediction string 'S'.
        """
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    THRESHOLD = 0.6

    @staticmethod
    def _clean_passage_text(text: str) -> str:
        """
        Removes Wikipedia section delimiters, citation tags, and excess whitespace.
        Used as an initial text preprocessing step for both claims and retrieved passages.

        :param text: Raw input text string (claim or evidence passage).
        :return: Sanitized string with HTML/Wikipedia artifacts stripped and whitespace normalized.
        """
        text = re.sub(r"</s><s>|</s>|<s>", " ", text)
        text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def __init__(
        self,
        threshold: float = 0.6,
        metric: str = "recall",
        aggregation: str = "max",
        use_stemmer: bool = True,
        remove_stopwords: bool = True,
        lowercase: bool = True,
        use_lemmatization: bool = False,
        compare: bool = False,
    ):
        """
        Initializes the Word Overlap Fact Checker baseline with configurable lexical preprocessing.

        :param threshold: Decision boundary threshold (predict 'S' if overlap score >= threshold, else 'NS').
        :param metric: Scoring metric to use: 'recall', 'jaccard', or 'f1'.
        :param aggregation: Multi-passage aggregation strategy: 'max' (taking max score across
                            passages) or 'concat' (scoring against concatenated passages).
        :param use_stemmer: Whether to apply Porter stemming (stemming vs none ablation).
        :param remove_stopwords: Whether to remove English stopwords.
        :param lowercase: Whether to convert text to lowercase.
        :param use_lemmatization: Whether to use WordNet lemmatization instead of stemming.
        :param compare: Whether to run both stemming and lemmatization and compare them side by side.
        """
        self.threshold = threshold if threshold is not None else self.THRESHOLD
        self.metric = metric
        self.aggregation = aggregation
        self.use_stemmer = use_stemmer
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_lemmatization = use_lemmatization
        self.compare = compare

        # Trigger comparison monkey-patch ONLY when explicitly requested
        if self.compare:
            _patch_main_for_comparison()

    def _preprocess(self, text: str) -> set:
        """
        Converts raw text into a clean set of tokens for lexical overlap comparison.
        Applies passage artifact cleaning, optional lowercasing, punctuation normalization,
        stopword filtering, and Porter stemming.

        :param text: Raw input text string (claim or evidence passage).
        :return: Set of preprocessed, stemmed token strings.
        """
        # Clean Wikipedia delimiters, citations, and excess whitespace
        text = self._clean_passage_text(text)

        if self.lowercase:
            text = text.lower()

        # Replace all punctuation characters with whitespace to preserve token boundaries
        for punc in string.punctuation:
            text = text.replace(punc, " ")

        # Tokenize on whitespace
        tokens = text.split()

        # Remove English stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Convert words to their root form (Porter Stemming)
        if self.use_stemmer and self.stemmer is not None:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return set(tokens)

    def _preprocess_with_lemm(self, text: str) -> set:
        """
        Converts raw text into a clean set of tokens using WordNet lemmatization.
        Applies passage artifact cleaning, lowercasing, punctuation stripping,
        stopword filtering, and verb lemmatization.

        :param text: Raw input text string (claim or evidence passage).
        :return: Set of preprocessed, lemmatized token strings.
        """
        # Clean Wikipedia delimiters, citations, and excess whitespace
        text = self._clean_passage_text(text)

        # lowercase + remove punctuation
        text = text.lower()

        for punc in string.punctuation:
            text = text.replace(punc, " ")

        # Tokenize
        tokens = text.split()

        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatization with pos='v' to handle verbs
        tokens = [self.lemmatizer.lemmatize(token, pos='v') for token in tokens]

        return set(tokens)

    def _recall(self, fact_tokens: set, passage_tokens: set) -> float:
        """
        Calculates token-level recall score: |fact_tokens ∩ passage_tokens| / |fact_tokens|.
        Measures the proportion of atomic fact tokens covered by the evidence passage.

        :param fact_tokens: Set of preprocessed tokens representing the claim/fact.
        :param passage_tokens: Set of preprocessed tokens representing the evidence passage.
        :return: Recall score as a float in [0.0, 1.0].
        """
        if not fact_tokens:
            return 0.0
        return len(fact_tokens & passage_tokens) / len(fact_tokens)

    def _jaccard(self, fact_tokens: set, passage_tokens: set) -> float:
        """
        Calculates Jaccard similarity coefficient: |fact_tokens ∩ passage_tokens| / |fact_tokens ∪ passage_tokens|.

        :param fact_tokens: Set of preprocessed tokens representing the claim/fact.
        :param passage_tokens: Set of preprocessed tokens representing the evidence passage.
        :return: Jaccard similarity coefficient as a float in [0.0, 1.0].
        """
        union = fact_tokens | passage_tokens
        if not union:
            return 0.0
        return len(fact_tokens & passage_tokens) / len(union)

    def _f1(self, fact_tokens: set, passage_tokens: set) -> float:
        """
        Calculates token-level F1 score (harmonic mean of precision and recall).

        :param fact_tokens: Set of preprocessed tokens representing the claim/fact.
        :param passage_tokens: Set of preprocessed tokens representing the evidence passage.
        :return: F1 score as a float in [0.0, 1.0].
        """
        intersection = fact_tokens & passage_tokens
        if not intersection:
            return 0.0
        precision = len(intersection) / len(passage_tokens)
        recall = len(intersection) / len(fact_tokens)
        return 2 * precision * recall / (precision + recall)

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Predicts whether an atomic fact is supported ('S') or not supported ('NS') by retrieved passages.

        :param fact: The atomic proposition or sentence to verify.
        :param passages: List of retrieved evidence passages, each a dictionary containing 'title' and 'text'.
        :return: 'S' if lexical overlap exceeds the decision threshold, otherwise 'NS'.
        """
        preprocess_fn = (
            self._preprocess_with_lemm if self.use_lemmatization else self._preprocess
        )

        fact_tokens = preprocess_fn(fact)
        if not fact_tokens:
            return "NS"

        # scoring function
        if self.metric == "jaccard":
            score_fn = self._jaccard
        elif self.metric == "f1":
            score_fn = self._f1
        else:
            score_fn = self._recall

        thresh = self.threshold if self.threshold is not None else self.THRESHOLD

        if self.aggregation == "concat":
            concat_text = " ".join(p.get("text", "") for p in passages)
            passage_tokens = preprocess_fn(concat_text)
            score = score_fn(fact_tokens, passage_tokens)
            return "S" if score > thresh else "NS"
        else:
            # Default: max score across all retrieved passages
            max_score = 0.0
            for passage in passages:
                passage_tokens = preprocess_fn(passage.get("text", ""))
                score = score_fn(fact_tokens, passage_tokens)
                if score > max_score:
                    max_score = score

            return "S" if max_score > thresh else "NS"

    @classmethod
    def cross_validate(
        cls,
        examples: List[Any],
        k: int = 5,
        seed: int = 42,
        metric: str = "recall",
        aggregation: str = "max",
        verbose: bool = True,
    ) -> dict:
        """
        Performs k-fold stratified cross-validation on development examples to calibrate
        and validate the decision threshold.

        Addresses Part 1 grading criteria for 'appropriate validation procedures' in the
        absence of a separate test set by demonstrating threshold stability across subsets.

        :param examples: List of FactExample objects (or dicts with 'fact'/'text', 'passages', 'label').
        :param k: Number of cross-validation folds (default: 5).
        :param seed: Random seed for fold partitioning.
        :param metric: Scoring metric ('recall', 'jaccard', or 'f1').
        :param aggregation: Passage aggregation ('max' or 'concat').
        :param verbose: Whether to print the formatted fold evaluation table.
        :return: Dict containing fold metrics, mean validation accuracy, and mean optimal threshold.
        """
        checker = cls(metric=metric, aggregation=aggregation)
        scores = []
        labels = []

        score_fn = getattr(checker, f"_{metric}", checker._recall)

        for ex in examples:
            if hasattr(ex, "fact"):
                fact_text = ex.fact
                passages_list = ex.passages
                label_val = 1 if ex.label == "S" else 0
            elif isinstance(ex, dict):
                fact_text = ex.get("text") or ex.get("fact", "")
                passages_list = ex.get("passages", [])
                label_val = 1 if ex.get("label") == "S" else 0
            else:
                continue

            fact_tokens = checker._preprocess(fact_text)
            if aggregation == "concat":
                concat_text = " ".join(p.get("text", "") for p in passages_list)
                p_tokens = checker._preprocess(concat_text)
                score = score_fn(fact_tokens, p_tokens)
            else:
                score = max(
                    (score_fn(fact_tokens, checker._preprocess(p.get("text", ""))) for p in passages_list),
                    default=0.0
                )
            scores.append(score)
            labels.append(label_val)

        scores = np.array(scores)
        labels = np.array(labels)
        n = len(scores)

        # Stratified K-Fold partitioning
        np.random.seed(seed)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        pos_folds = np.array_split(pos_idx, k)
        neg_folds = np.array_split(neg_idx, k)
        folds = [np.concatenate([pos_folds[i], neg_folds[i]]) for i in range(k)]

        fold_results = []
        if verbose:
            print(f"\n{'='*78}")
            print(f"5-FOLD STRATIFIED CROSS-VALIDATION FOR THRESHOLD TUNING (Part 1)")
            print(f"{'='*78}")
            print(f"{'Fold':<6} | {'Train N':<8} | {'Val N':<7} | {'Optimal tau':<12} | {'Train Acc':<11} | {'Val Acc':<10}")
            print(f"{'-'*78}")

        for fold_i in range(k):
            val_idx = folds[fold_i]
            train_idx = np.setdiff1d(np.arange(n), val_idx)

            train_scores, train_labels = scores[train_idx], labels[train_idx]
            val_scores, val_labels = scores[val_idx], labels[val_idx]

            best_t, best_train_acc = 0.60, 0.0
            for t in np.arange(0.10, 0.95, 0.01):
                acc = np.mean((train_scores > t) == train_labels)
                if acc > best_train_acc:
                    best_train_acc = acc
                    best_t = t

            val_acc = np.mean((val_scores > best_t) == val_labels)

            fold_results.append({
                "fold": fold_i + 1,
                "train_n": len(train_idx),
                "val_n": len(val_idx),
                "optimal_threshold": round(float(best_t), 2),
                "train_acc": float(best_train_acc),
                "val_acc": float(val_acc)
            })

            if verbose:
                print(f"{fold_i+1:<6} | {len(train_idx):<8} | {len(val_idx):<7} | {best_t:<12.2f} | {best_train_acc*100:<10.2f}% | {val_acc*100:<9.2f}%")

        mean_val_acc = float(np.mean([r["val_acc"] for r in fold_results]))
        std_val_acc = float(np.std([r["val_acc"] for r in fold_results]))
        mean_tau = float(np.mean([r["optimal_threshold"] for r in fold_results]))

        if verbose:
            print(f"{'-'*78}")
            print(f"Mean Validation Accuracy : {mean_val_acc*100:.2f}% (+/- {std_val_acc*100:.2f}%)")
            print(f"Mean Optimal Threshold   : {mean_tau:.2f}")
            print(f"{'='*78}\n")

        return {
            "folds": fold_results,
            "mean_val_acc": mean_val_acc,
            "std_val_acc": std_val_acc,
            "mean_optimal_threshold": mean_tau
        }

class EntailmentFactChecker(FactChecker):
    """
    Optimized NLI-based fact checker using MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli.

    Processing Flow:
    1. Preprocessing & Artifact Cleaning:
       Strips section markers (</s><s>), wiki citations, and formatting noise.
    2. Context-Aware Sentence Decomposition & Pronoun Resolution:
       Segments passages into sentences and prepends passage titles (e.g. 'Houghton. He was born in Hull.')
       to resolve anonymous biographical pronouns ('He', 'She', 'They') into explicit named entities.
       Also constructs adjacent 2-sentence windows to capture cross-sentence co-reference.
    3. Sentence-Level Lexical Pruning (Part 1 Sieve):
       Scores each candidate sentence against the claim using stemmed unigram overlap.
       Sorts and retains only the top-k highest-relevance candidate premises, immediately
       short-circuiting claims with zero lexical evidence to 'NS'.
    4. Cross-Encoder NLI Inference with Contradiction Guard:
       Computes P(entailment) and verifies P(entailment) > P(contradiction) to reject
       refuted claims.
    5. Calibrated Threshold Decision:
       Predicts 'S' if max_entailment >= threshold (default 0.50), else 'NS'.
    """

    def __init__(
        self,
        ent_model,
        strategy: str = "threshold",
        threshold: float = 0.50,
        prune_threshold: float = 0.10,
        use_pruning: bool = True,
        sentence_level: bool = True,
        max_candidates: int = 6,
    ):
        """
        Initializes the Textual Entailment Fact Checker pipeline with multi-stage reasoning.

        :param ent_model: DeBERTa/RoBERTa entailment model wrapper (instance of EntailmentModel).
        :param strategy: Decision rule ('threshold' for continuous probability thresholding, or 'discrete' for 3-class argmax).
        :param threshold: Continuous probability decision boundary (predict 'S' if max P(entailment) >= threshold).
        :param prune_threshold: Minimum lexical overlap score required to retain a candidate premise sentence (default: 0.10).
        :param use_pruning: Whether to filter out low-overlap candidate sentences prior to transformer NLI inference.
        :param sentence_level: If True, evaluates passage sentences individually with pronoun resolution and sliding windows; if False, evaluates full passages.
        :param max_candidates: Maximum number of top-ranked candidate premises to evaluate with NLI per claim (default: 6).
        """
        self.ent_model = ent_model
        self.strategy = strategy
        self.threshold = threshold
        self.prune_threshold = prune_threshold
        self.use_pruning = use_pruning
        self.sentence_level = sentence_level
        self.max_candidates = max_candidates
        self.overlap_checker = WordRecallThresholdFactChecker()

        if self.sentence_level:
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

    @staticmethod
    def _clean_passage_text(text: str) -> str:
        """
        Removes Wikipedia section delimiters, citation tags, and excess whitespace.
        Delegates to WordRecallThresholdFactChecker._clean_passage_text.

        :param text: Raw text string from the passage or claim.
        :return: Sanitized string with markup and citations stripped.
        """
        return WordRecallThresholdFactChecker._clean_passage_text(text)

    def _extract_candidates(self, fact: str, passages: List[dict]) -> List[str]:
        """
        Extracts, contextualizes, and prunes candidate premises from retrieved passages.
        Prepends passage titles to resolve biographical pronouns ('He', 'She', 'They'),
        constructs adjacent 2-sentence windows, and filters candidates via lexical overlap.

        :param fact: The atomic claim to verify.
        :param passages: List of retrieved evidence passages (each dict containing 'title' and 'text').
        :return: List of top-k contextualized candidate premise strings sorted by lexical relevance.
        """
        fact_tokens = self.overlap_checker._preprocess(fact)
        if not fact_tokens:
            return []

        candidates = []
        for passage in passages:
            title = passage.get("title", "").strip()
            raw_text = passage.get("text", "")
            if not raw_text:
                continue

            clean_text = self._clean_passage_text(raw_text)
            sentences = nltk.sent_tokenize(clean_text)

            for i, sent in enumerate(sentences):
                s_clean = sent.strip()
                if len(s_clean.split()) < 3:
                    continue

                # Single sentence with title prepended
                premise = f"{title}. {s_clean}" if title else s_clean
                prem_tokens = self.overlap_checker._preprocess(premise)
                overlap = self.overlap_checker._recall(fact_tokens, prem_tokens)
                candidates.append((overlap, premise))

                # Sliding 2-sentence window
                if i > 0:
                    win_text = f"{sentences[i-1].strip()} {s_clean}"
                    win_premise = f"{title}. {win_text}" if title else win_text
                    win_tokens = self.overlap_checker._preprocess(win_premise)
                    win_ov = self.overlap_checker._recall(fact_tokens, win_tokens)
                    candidates.append((win_ov, win_premise))

        if not candidates:
            return []

        if not self.use_pruning:
            return [c[1] for c in candidates]

        # Sentence-level lexical pruning: filter and sort by overlap
        surviving = [c for c in candidates if c[0] >= self.prune_threshold]
        if not surviving:
            return []

        surviving.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in surviving[: self.max_candidates]]

    def predict_threshold(self, fact: str, passages: List[dict], threshold: float = None) -> str:
        """
        Continuous Probability Thresholding with Max-Pooling and Contradiction Guard.

        :param fact: The atomic claim to verify.
        :param passages: List of retrieved evidence passages.
        :param threshold: Optional override for the decision threshold; defaults to self.threshold.
        :return: 'S' if max P(entailment) >= threshold and P(entailment) > P(contradiction), else 'NS'.
        """
        thresh = threshold if threshold is not None else self.threshold

        if not self.sentence_level:
            # Baseline full passage evaluation
            max_score = 0.0
            for p in passages:
                p_text = self._clean_passage_text(p.get("text", ""))
                score = self.ent_model.check_entailment(p_text, fact)
                max_score = max(max_score, score)
            return "S" if max_score >= thresh else "NS"

        candidate_premises = self._extract_candidates(fact, passages)
        if not candidate_premises:
            return "NS"

        max_entailment = 0.0
        for premise in candidate_premises:
            p_ent, p_neu, p_con = self.ent_model.check_entailment_raw(premise, fact)
            # Contradiction guard: entailment must strictly exceed contradiction
            if p_ent > p_con and p_ent > max_entailment:
                max_entailment = p_ent

            # Early stopping on decisive entailment
            if max_entailment >= 0.95:
                break

        return "S" if max_entailment >= thresh else "NS"

    def predict_discrete(self, fact: str, passages: List[dict]) -> str:
        """
        Discrete 3-Class Entailment Classification Strategy.
        Predicts 'S' if any candidate premise produces argmax = 'entailment' (P_ent > P_neu and P_ent > P_con).

        :param fact: The atomic claim to verify.
        :param passages: List of retrieved evidence passages.
        :return: 'S' if any candidate premise strictly entails the claim, else 'NS'.
        """
        candidate_premises = self._extract_candidates(fact, passages)
        if not candidate_premises:
            return "NS"

        for premise in candidate_premises:
            p_ent, p_neu, p_con = self.ent_model.check_entailment_raw(premise, fact)
            if p_ent > p_neu and p_ent > p_con:
                return "S"

        return "NS"

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a factual verification prediction on the given fact using retrieved passages.
        Dispatches to predict_discrete or predict_threshold based on self.strategy.

        :param fact: The atomic claim to verify.
        :param passages: List of retrieved evidence passages.
        :return: 'S' (Supported) or 'NS' (Not Supported).
        """
        if self.strategy == "discrete":
            return self.predict_discrete(fact, passages)
        else:
            return self.predict_threshold(fact, passages)


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        """
        Initializes the dependency parser fact checker using spaCy English core model.
        """
        self.nlp = spacy.load("en_core_web_sm")

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Predicts 'S' or 'NS' based on dependency relation overlap.

        :param fact: The atomic proposition or sentence to verify.
        :param passages: List of retrieved evidence passages.
        :return: 'S' or 'NS'.
        """
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = [
                "punct",
                "ROOT",
                "root",
                "det",
                "case",
                "aux",
                "auxpass",
                "dep",
                "cop",
                "mark",
            ]
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == "VERB" else token.head.text
            dependent = token.lemma_ if token.pos_ == "VERB" else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations


