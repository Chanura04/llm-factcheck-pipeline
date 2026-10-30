# factcheck.py
import torch
from typing import List
import numpy as np
import spacy
import gc
import nltk
import sys
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

try:
    stopwords.words('english')
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    wordnet.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def _patch_main_for_comparison():
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
        if isinstance(fact_checker, WordRecallThresholdFactChecker) and not getattr(fact_checker, 'use_lemmatization',
                                                                                    False):
            print("\n" + "=" * 72)
            print("  WORD_OVERLAP  —  PREPROCESSING COMPARISON")
            print("  Stemming (PorterStemmer)  vs  Lemmatization (WordNetLemmatizer)")
            print("=" * 72)

            print("\n>>> [1 / 2]  STEMMING  —  running...\n")
            original_predict(examples, WordRecallThresholdFactChecker(use_lemmatization=False))
            stem_acc = last_accuracy["val"]

            print("\n>>> [2 / 2]  LEMMATIZATION  —  running...\n")
            original_predict(examples, WordRecallThresholdFactChecker(use_lemmatization=True))
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
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    THRESHOLD = 0.6

    def __init__(self, use_lemmatization: bool = False):
        # Trigger the monkey-patch when this class is instantiated
        _patch_main_for_comparison()

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_lemmatization = use_lemmatization

    def _preprocess(self, text: str) -> set:
        """Convert raw test into clean set of tokens for overlap comparision"""

        #  lowercase + remove punctuation
        text = text.lower()

        for punc in [",", "!", ".", "?", ":", ";", "()", "[]", "{}", "'", "\"", "@", "#", "$", "%", "^", "&", "*", "~"]:
            text = text.replace(punc, "")

        # Tokenize
        tokens = text.split()

        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]

        # Remove prefixes , suffixes and Convert words to their root form (Stemming)
        tokens = [self.stemmer.stem(token) for token in tokens]

        return set(tokens)

    def _preprocess_with_lemm(self, text: str) -> set:
        """Identical to _preprocess() — only the stemming line is replaced with lemmatization."""

        #  lowercase + remove punctuation
        text = text.lower()

        for punc in [",", "!", ".", "?", ":", ";", "()", "[]", "{}", "'", "\"", "@", "#", "$", "%", "^", "&", "*", "~"]:
            text = text.replace(punc, "")

        # Tokenize
        tokens = text.split()

        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]

        # ONLY THIS LINE CHANGED: Lemmatization with pos='v' to handle verbs
        tokens = [self.lemmatizer.lemmatize(token, pos='v') for token in tokens]

        return set(tokens)

    def _recall(self, fact_tokens: set, passage_tokens: set) -> float:
        """
        Calculates the recall score, which measures the proportion of tokens from the fact that are present in the passage.
        This helps determine how much of the original fact is covered by the supporting text.
        Recall = |fact_tokens ∩ passage_tokens| / |fact_tokens|
        """
        if len(fact_tokens) == 0:
            return 0.0
        intersection = fact_tokens & passage_tokens
        return len(intersection) / len(fact_tokens)

    def _jaccard(self, fact_tokens: set, passage_tokens: set) -> float:
        """
        Calculates the Jaccard similarity score, which measures the similarity between the fact tokens and passage tokens.
        This helps determine how much of the original fact is covered by the supporting text.
        Jaccard = |fact_tokens ∩ passage_tokens| / |fact_tokens ∪ passage_tokens|
        """
        if len(fact_tokens | passage_tokens) == 0:
            return 0.0
        return len(fact_tokens & passage_tokens) / len(fact_tokens | passage_tokens)

    def _f1(self, fact_tokens: set, passage_tokens: set) -> float:
        intersection = fact_tokens & passage_tokens
        if len(intersection) == 0:
            return 0.0
        precision = len(intersection) / len(passage_tokens)
        recall = len(intersection) / len(fact_tokens)
        return 2 * precision * recall / (precision + recall)

    def predict(self, fact: str, passages: List[dict]) -> str:
        preprocess_fn = (
            self._preprocess_with_lemm if self.use_lemmatization else self._preprocess
        )

        fact_tokens = preprocess_fn(fact)

        max_score = 0.0
        for passage in passages:
            passage_tokens = preprocess_fn(passage["text"])
            score = self._recall(fact_tokens, passage_tokens)
            max_score = max(max_score, score)

        return "S" if max_score > self.THRESHOLD else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
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
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations


class DatasetEDA(object):
    """
    Encapsulates Exploratory Data Analysis (EDA) on the raw dataset before any
    preprocessing (tokenization, lowercasing, stopword removal, stemming) is applied.
    """

    def __init__(self, dev_path: str = "data/dev_labeled_ChatGPT.jsonl",
                 passages_path: str = "data/passages_bm25_ChatGPT_humfacts.jsonl"):
        self.dev_path = dev_path
        self.passages_path = passages_path
        self.raw_biographies = []
        self.fact_to_passages = {}
        self.raw_facts = []

    def load_data(self):
        """Loads raw JSONL datasets without applying any preprocessing."""
        import json
        import os

        if not os.path.exists(self.dev_path) or not os.path.exists(self.passages_path):
            raise FileNotFoundError(f"Missing data files: {self.dev_path} or {self.passages_path}")

        with open(self.passages_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.fact_to_passages[record["sent"]] = record["passages"]

        with open(self.dev_path, "r", encoding="utf-8") as f:
            self.raw_biographies = [json.loads(line) for line in f]

        self.raw_facts = []
        for bio in self.raw_biographies:
            for ann in bio.get("annotations", []):
                for hf in ann.get("human-atomic-facts", []):
                    self.raw_facts.append(hf)

    def print_topic_overview(self):
        """Analyzes entities, topics, and rarity categories."""
        from collections import Counter
        categories = Counter()
        regions = Counter()

        for bio in self.raw_biographies:
            cat = bio.get("cat", [])
            if len(cat) > 0:
                categories[cat[0]] += 1
            if len(cat) > 1:
                regions[cat[1]] += 1

        print("\n" + "=" * 60)
        print("1. TOPIC & ENTITY ANALYSIS")
        print("=" * 60)
        print(f"Total Biographies/Topics : {len(self.raw_biographies)}")
        print(f"Total Atomic Facts       : {len(self.raw_facts)}")
        print(f"Entity Popularity (cat)  : {dict(categories)}")
        print(f"Geographic Distribution  : {dict(regions)}")

    def print_label_distribution(self):
        """Analyzes class balance across S, NS, and IR."""
        from collections import Counter
        labels = Counter(f["label"] for f in self.raw_facts)
        total = len(self.raw_facts)

        print("\n" + "=" * 60)
        print("2. CLASS BALANCE (TARGET LABELS)")
        print("=" * 60)
        for lbl, count in labels.most_common():
            print(f"  {lbl:<4} : {count:>4} ({count / total * 100:>5.2f}%)")

        s_count = labels.get("S", 0)
        ns_count = labels.get("NS", 0) + labels.get("IR", 0)
        print(f"\nBinary Classification Split (IR mapped to NS):")
        print(f"  Supported (S)     : {s_count} ({s_count / total * 100:.2f}%)")
        print(f"  Not Supported (NS): {ns_count} ({ns_count / total * 100:.2f}%)")

    def print_length_analysis(self):
        """Analyzes word and character lengths of claims and evidence passages."""
        import statistics

        fact_words = [len(f["text"].split()) for f in self.raw_facts]
        fact_chars = [len(f["text"]) for f in self.raw_facts]

        passages_counts = []
        passage_words = []
        for f in self.raw_facts:
            ps = self.fact_to_passages.get(f["text"], [])
            passages_counts.append(len(ps))
            for p in ps:
                passage_words.append(len(p.get("text", "").split()))

        print("\n" + "=" * 60)
        print("3. RAW SEQUENCE LENGTH METRICS")
        print("=" * 60)
        print(
            f"Atomic Fact Words      : Min={min(fact_words)}, Max={max(fact_words)}, Mean={statistics.mean(fact_words):.2f}, Median={statistics.median(fact_words)}")
        print(
            f"Atomic Fact Characters : Min={min(fact_chars)}, Max={max(fact_chars)}, Mean={statistics.mean(fact_chars):.2f}, Median={statistics.median(fact_chars)}")
        print(
            f"Passages per Fact      : Min={min(passages_counts)}, Max={max(passages_counts)}, Mean={statistics.mean(passages_counts):.2f}")
        print(
            f"Words per Passage      : Min={min(passage_words)}, Max={max(passage_words)}, Mean={statistics.mean(passage_words):.2f}")

    def print_vocabulary_analysis(self, top_n: int = 10):
        """Examines the raw lexical space to justify subsequent preprocessing."""
        from collections import Counter
        vocab = Counter()
        for f in self.raw_facts:
            vocab.update(f["text"].split())

        print("\n" + "=" * 60)
        print(f"4. LEXICAL PROFILE (Top {top_n} Tokens Before Cleaning)")
        print("=" * 60)
        print(f"Total Unique Tokens (Raw Vocab): {len(vocab)}")
        for token, count in vocab.most_common(top_n):
            print(f"  {token:<15} : {count:>4} times")
        print("  -> Justification: Dominated by punctuation and stopwords ('is', 'was', 'in'),")
        print("     proving why lowercasing, stopword filtering, and stemming are necessary.")
        print("=" * 60 + "\n")

    def generate_plots(self, output_dir: str = "eda_plots"):
        """
        Generates and saves visual EDA plots (both individual figures and a master dashboard)
        to the specified directory for coursework reports.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from collections import Counter
            import statistics
            import os
        except ImportError:
            print("[EDA WARNING] matplotlib is not installed. To generate plots, run: uv pip install matplotlib")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[5. PLOT GENERATION] Generating EDA figures into '{output_dir}/'...")

        label_counts = Counter(f["label"] for f in self.raw_facts)
        fact_lengths = [len(f["text"].split()) for f in self.raw_facts]
        passage_lengths = [len(p["text"].split()) for f in self.raw_facts for p in
                           self.fact_to_passages.get(f["text"], [])]

        cat_counts = Counter()
        for b in self.raw_biographies:
            cat = b.get("cat", [])
            if cat:
                cat_counts[cat[0]] += 1
        rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]
        rarity_values = [cat_counts[r] for r in rarity_order]

        vocab = Counter(w for f in self.raw_facts for w in f["text"].split())
        top_tokens = vocab.most_common(10)

        recalls_s, recalls_ns = [], []
        for f in self.raw_facts:
            f_tokens = set(f["text"].split())
            ps = self.fact_to_passages.get(f["text"], [])
            best_r = max((len(f_tokens & set(p["text"].split())) / len(f_tokens) for p in ps),
                         default=0.0) if f_tokens else 0.0
            if f["label"] == "S":
                recalls_s.append(best_r)
            else:
                recalls_ns.append(best_r)

        # -------------------------------------------------------------
        # Master Figure: 2x3 Grid Dashboard
        # -------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Exploratory Data Analysis (EDA) - Raw Dataset Before Preprocessing", fontsize=16,
                     fontweight="bold")

        # 1. Target Label Distribution
        ax1 = axes[0, 0]
        ax1.pie(
            [label_counts["S"], label_counts["NS"], label_counts["IR"]],
            labels=["Supported (S)", "Not Supported (NS)", "Irrelevant (IR)"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2ecc71", "#e74c3c", "#f39c12"],
            explode=(0.04, 0.04, 0.08)
        )
        ax1.set_title("Target Label Distribution (Raw 3-Class)", fontsize=12, fontweight="bold")

        # 2. Entity Rarity in Wikipedia
        ax2 = axes[0, 1]
        bars2 = ax2.bar(rarity_order, rarity_values, color="#3498db", edgecolor="black", alpha=0.85)
        ax2.bar_label(bars2, padding=3)
        ax2.set_title("Entity Rarity in Wikipedia (cat[0])", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Number of Biographies")

        # 3. Fact Word Length Distribution
        ax3 = axes[0, 2]
        ax3.hist(fact_lengths, bins=range(min(fact_lengths), max(fact_lengths) + 2), color="#9b59b6", edgecolor="black",
                 alpha=0.8, align="left")
        ax3.axvline(statistics.mean(fact_lengths), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {statistics.mean(fact_lengths):.2f}")
        ax3.set_title("Atomic Fact Word Length (Words/Fact)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Word Count")
        ax3.set_ylabel("Frequency")
        ax3.legend()

        # 4. Evidence Passage Length
        ax4 = axes[1, 0]
        ax4.hist(passage_lengths, bins=25, color="#1abc9c", edgecolor="black", alpha=0.8)
        ax4.axvline(statistics.mean(passage_lengths), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {statistics.mean(passage_lengths):.1f}")
        ax4.set_title("Evidence Passage Length (Words/Passage)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Word Count")
        ax4.set_ylabel("Frequency")
        ax4.legend()

        # 5. Top Raw Fact Tokens (Uncleaned)
        ax5 = axes[1, 1]
        tokens, freqs = zip(*reversed(top_tokens))
        bars5 = ax5.barh(tokens, freqs, color="#e67e22", edgecolor="black", alpha=0.85)
        ax5.bar_label(bars5, padding=3)
        ax5.set_title("Top 10 Raw Fact Tokens (Uncleaned)", fontsize=12, fontweight="bold")
        ax5.set_xlabel("Occurrences (Dominated by Stopwords)")

        # 6. Raw Surface Token Overlap
        ax6 = axes[1, 2]
        bp = ax6.boxplot([recalls_s, recalls_ns], tick_labels=["Supported (S)", "Not Supported (NS/IR)"],
                         patch_artist=True)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax6.set_title("Raw Token Recall (Before Preprocessing)", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Max Jaccard/Recall Overlap")

        plt.tight_layout()
        dashboard_path = os.path.join(output_dir, "eda_dashboard.png")
        plt.savefig(dashboard_path, dpi=200)
        plt.close()
        print(f"  [OK] Master Dashboard saved to: {dashboard_path}")

        # -------------------------------------------------------------
        # Save Individual High-Resolution Figures for Coursework Report
        # -------------------------------------------------------------
        # Individual 1: Label Distribution
        plt.figure(figsize=(6, 5))
        plt.pie([label_counts["S"], label_counts["NS"], label_counts["IR"]],
                labels=["Supported (S)", "Not Supported (NS)", "Irrelevant (IR)"],
                autopct="%1.1f%%", startangle=140, colors=["#2ecc71", "#e74c3c", "#f39c12"], explode=(0.04, 0.04, 0.08))
        plt.title("Target Label Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "1_label_distribution.png"), dpi=200)
        plt.close()

        # Individual 2: Entity Rarity
        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(rarity_order, rarity_values, color="#3498db", edgecolor="black", alpha=0.85)
        plt.bar_label(bars, padding=3)
        plt.title("Entity Rarity Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Rarity in Wikipedia")
        plt.ylabel("Number of Biographies")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "2_entity_rarity.png"), dpi=200)
        plt.close()

        # Individual 3: Fact Lengths
        plt.figure(figsize=(7, 4.5))
        plt.hist(fact_lengths, bins=range(min(fact_lengths), max(fact_lengths) + 2), color="#9b59b6", edgecolor="black",
                 alpha=0.8, align="left")
        plt.axvline(statistics.mean(fact_lengths), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {statistics.mean(fact_lengths):.2f}")
        plt.title("Atomic Fact Length Distribution (Words)", fontsize=13, fontweight="bold")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_fact_lengths.png"), dpi=200)
        plt.close()

        # Individual 4: Passage Lengths
        plt.figure(figsize=(7, 4.5))
        plt.hist(passage_lengths, bins=25, color="#1abc9c", edgecolor="black", alpha=0.8)
        plt.axvline(statistics.mean(passage_lengths), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {statistics.mean(passage_lengths):.1f}")
        plt.title("Evidence Passage Length Distribution (Words)", fontsize=13, fontweight="bold")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "4_passage_lengths.png"), dpi=200)
        plt.close()

        # Individual 5: Top Tokens
        plt.figure(figsize=(7, 5))
        bars = plt.barh(tokens, freqs, color="#e67e22", edgecolor="black", alpha=0.85)
        plt.bar_label(bars, padding=3)
        plt.title("Top 10 Raw Tokens Before Preprocessing", fontsize=13, fontweight="bold")
        plt.xlabel("Occurrences")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "5_top_raw_tokens.png"), dpi=200)
        plt.close()

        # Individual 6: Raw Overlap
        plt.figure(figsize=(6, 5))
        bp = plt.boxplot([recalls_s, recalls_ns], tick_labels=["Supported (S)", "Not Supported (NS/IR)"],
                         patch_artist=True)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        plt.title("Raw Lexical Recall by Label", fontsize=13, fontweight="bold")
        plt.ylabel("Max Surface Recall Overlap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "6_raw_lexical_recall.png"), dpi=200)
        plt.close()

        print(f"  [OK] All 6 individual high-res figures saved to '{output_dir}/'")
        print("=" * 60 + "\n")

    def compare_similarity_metrics(self, output_dir: str = "eda_plots"):
        """
        Benchmarks different similarity & preprocessing methods (Word Recall,
        TF-IDF Cosine, ROUGE-1, ROUGE-2, ROUGE-L) on the dataset and prints
        a comparative evaluation table with optimal decision thresholds.
        """
        import os
        import numpy as np
        from collections import Counter
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        try:
            from rouge_score import rouge_scorer
        except ImportError:
            rouge_scorer = None

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            TfidfVectorizer = None

        stop_words = set(stopwords.words("english"))
        stemmer = PorterStemmer()

        def clean_tokens(text):
            text = text.lower()
            for punc in [",", "!", ".", "?", ":", ";", "()", "[]", "{}", "'", '"', "@", "#", "$", "%", "^", "&", "*",
                         "~"]:
                text = text.replace(punc, " ")
            return set(stemmer.stem(t) for t in text.split() if t not in stop_words)

        examples = []
        for f in self.raw_facts:
            text = f.get("text", "")
            if text in self.fact_to_passages:
                label_val = 1 if f.get("label") == "S" else 0
                examples.append((text, self.fact_to_passages[text], label_val))

        labels = [ex[2] for ex in examples]
        total_ex = len(labels)

        # 1. Word Recall & Jaccard
        recall_scores, jaccard_scores = [], []
        for fact, ps, _ in examples:
            ft = clean_tokens(fact)
            r_max, j_max = 0.0, 0.0
            for p in ps:
                pt = clean_tokens(p.get("text", ""))
                inter = len(ft & pt)
                r = inter / len(ft) if ft else 0.0
                j = inter / len(ft | pt) if (ft | pt) else 0.0
                r_max = max(r_max, r)
                j_max = max(j_max, j)
            recall_scores.append(r_max)
            jaccard_scores.append(j_max)

        metric_dict = {
            "Word Recall (Stem+Stopwords)": recall_scores,
            "Word Jaccard": jaccard_scores,
        }

        # 2. ROUGE Metrics
        if rouge_scorer is not None:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            r1_r, r2_r, rl_r = [], [], []
            r1_f, r2_f, rl_f = [], [], []
            for fact, ps, _ in examples:
                b_r1_r, b_r2_r, b_rl_r = 0.0, 0.0, 0.0
                b_r1_f, b_r2_f, b_rl_f = 0.0, 0.0, 0.0
                for p in ps:
                    p_text = p.get("text", "")
                    s = scorer.score(target=fact, prediction=p_text)
                    b_r1_r = max(b_r1_r, s["rouge1"].recall)
                    b_r2_r = max(b_r2_r, s["rouge2"].recall)
                    b_rl_r = max(b_rl_r, s["rougeL"].recall)
                    b_r1_f = max(b_r1_f, s["rouge1"].fmeasure)
                    b_r2_f = max(b_r2_f, s["rouge2"].fmeasure)
                    b_rl_f = max(b_rl_f, s["rougeL"].fmeasure)
                r1_r.append(b_r1_r)
                r2_r.append(b_r2_r)
                rl_r.append(b_rl_r)
                r1_f.append(b_r1_f)
                r2_f.append(b_r2_f)
                rl_f.append(b_rl_f)

            metric_dict["ROUGE-1 Recall"] = r1_r
            metric_dict["ROUGE-1 F1"] = r1_f
            metric_dict["ROUGE-2 Recall"] = r2_r
            metric_dict["ROUGE-2 F1"] = r2_f
            metric_dict["ROUGE-L Recall"] = rl_r
            metric_dict["ROUGE-L F1"] = rl_f

        # 3. TF-IDF Cosine & Weighted Recall
        if TfidfVectorizer is not None:
            corpus = list(set([p["text"] for ex in examples for p in ex[1]] + [ex[0] for ex in examples]))
            tfidf = TfidfVectorizer(stop_words="english", lowercase=True, norm="l2")
            tfidf.fit(corpus)
            tfidf_sub = TfidfVectorizer(stop_words="english", lowercase=True, sublinear_tf=True, norm="l2")
            tfidf_sub.fit(corpus)

            idf_map = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

            cos_scores = []
            cos_sub_scores = []
            weighted_recalls = []

            for fact, ps, _ in examples:
                f_vec = tfidf.transform([fact])
                p_vecs = tfidf.transform([p["text"] for p in ps])
                cos_scores.append(float(np.max(cosine_similarity(f_vec, p_vecs)[0])) if len(ps) > 0 else 0.0)

                f_sub_vec = tfidf_sub.transform([fact])
                p_sub_vecs = tfidf_sub.transform([p["text"] for p in ps])
                cos_sub_scores.append(
                    float(np.max(cosine_similarity(f_sub_vec, p_sub_vecs)[0])) if len(ps) > 0 else 0.0)

                # TF-IDF Weighted Recall
                f_tokens = [w for w in tfidf.build_tokenizer()(fact.lower()) if w in idf_map]
                f_idf_sum = sum(idf_map[w] for w in f_tokens)
                best_w_r = 0.0
                for p in ps:
                    p_toks = set(tfidf.build_tokenizer()(p["text"].lower()))
                    matched_idf = sum(idf_map[w] for w in f_tokens if w in p_toks)
                    w_r = matched_idf / f_idf_sum if f_idf_sum > 0 else 0.0
                    best_w_r = max(best_w_r, w_r)
                weighted_recalls.append(best_w_r)

            metric_dict["TF-IDF Cosine Similarity"] = cos_scores
            metric_dict["TF-IDF Sublinear Cosine"] = cos_sub_scores
            metric_dict["TF-IDF Weighted Recall"] = weighted_recalls

        # Sweep thresholds
        results = []
        for name, scores in metric_dict.items():
            best_acc, best_thresh = 0.0, 0.0
            best_f1_s, best_f1_ns = 0.0, 0.0
            for t in np.linspace(0.01, 0.99, 100):
                preds = [1 if s >= t else 0 for s in scores]
                tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
                fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
                tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
                fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
                acc = (tp + tn) / total_ex
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = t
                    prec_s = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec_s = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    best_f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0
                    prec_ns = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                    rec_ns = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    best_f1_ns = 2 * prec_ns * rec_ns / (prec_ns + rec_ns) if (prec_ns + rec_ns) > 0 else 0.0

            macro_f1 = (best_f1_s + best_f1_ns) / 2
            results.append({
                "name": name,
                "thresh": best_thresh,
                "acc": best_acc,
                "f1_s": best_f1_s,
                "f1_ns": best_f1_ns,
                "macro_f1": macro_f1
            })

        # Sort by Accuracy descending
        results.sort(key=lambda x: x["acc"], reverse=True)

        print("\n" + "=" * 80)
        print("SIMILARITY & PREPROCESSING BENCHMARK (DEV SET EVALUATION)")
        print("=" * 80)
        h_m, h_t, h_a, h_s, h_ns, h_mf = "Metric", "Thresh", "Acc (%)", "S-F1", "NS-F1", "Macro-F1"
        print(f"{h_m:<30} | {h_t:<7} | {h_a:<8} | {h_s:<7} | {h_ns:<7} | {h_mf:<8}")
        print("-" * 80)
        for r in results:
            m_name, th, ac = r["name"], r["thresh"], r["acc"] * 100
            fs, fns, mf = r["f1_s"], r["f1_ns"], r["macro_f1"]
            print(f"{m_name:<30} | {th:<7.2f} | {ac:<8.2f} | {fs:<7.4f} | {fns:<7.4f} | {mf:<8.4f}")
        print("=" * 80)
        print(
            f"-> Winner: '{results[0]['name']}' with {results[0]['acc'] * 100:.2f}% Accuracy at threshold {results[0]['thresh']:.2f}.\n")

        # Save plot if matplotlib available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(output_dir, exist_ok=True)
            plot_names = [r["name"] for r in results]
            plot_accs = [r["acc"] * 100 for r in results]
            colors = ["#2ecc71" if a >= 75 else "#3498db" if a >= 70 else "#e67e22" if a >= 65 else "#e74c3c" for a in
                      plot_accs]

            plt.figure(figsize=(12, 5.5))
            bars = plt.bar(plot_names, plot_accs, color=colors, edgecolor="black", alpha=0.85)
            plt.axhline(results[0]["acc"] * 100, color="gray", linestyle="--",
                        label=f"Top: {results[0]['name']} ({results[0]['acc'] * 100:.2f}%)")
            plt.bar_label(bars, fmt="%.1f%%", padding=3, fontweight="bold")
            plt.title("Comparison of Similarity & Preprocessing Metrics on Fact-Checking", fontsize=14,
                      fontweight="bold")
            plt.ylabel("Dev Set Accuracy (%)", fontsize=11)
            plt.ylim(50, 90)
            plt.xticks(rotation=25, ha="right", fontsize=9)
            plt.legend(loc="upper right")
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, "similarity_metrics_comparison.png")
            plt.savefig(comparison_path, dpi=200)
            plt.close()
            print(f"  [OK] Metric comparison plot saved to: {comparison_path}")
        except Exception as e:
            pass

    def run(self, generate_plots: bool = True, compare_metrics: bool = True):
        """Executes all EDA routines sequentially, including benchmark and plot generation."""
        print("\n>>> Running Full Dataset EDA (Before Preprocessing) <<<")
        self.load_data()
        self.print_topic_overview()
        self.print_label_distribution()
        self.print_length_analysis()
        self.print_vocabulary_analysis()
        if compare_metrics:
            self.compare_similarity_metrics()
        if generate_plots:
            self.generate_plots()


if __name__ == "__main__":
    eda = DatasetEDA()
    eda.run()