# factcheck.py
import torch
from typing import List
import numpy as np
import spacy
import gc
import nltk
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

        # ONLY THIS LINE CHANGED: Lemmatization (added pos='v' to handle verbs, making it competitive with stemming)
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
        # Dispatch to the correct preprocessor based on the flag
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
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from collections import Counter
            import statistics
            import os
        except ImportError:
            return

        os.makedirs(output_dir, exist_ok=True)
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

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Exploratory Data Analysis (EDA) - Raw Dataset Before Preprocessing", fontsize=16,
                     fontweight="bold")

        ax1 = axes[0, 0]
        ax1.pie([label_counts["S"], label_counts["NS"], label_counts["IR"]],
                labels=["Supported (S)", "Not Supported (NS)", "Irrelevant (IR)"],
                autopct="%1.1f%%", startangle=140, colors=["#2ecc71", "#e74c3c", "#f39c12"], explode=(0.04, 0.04, 0.08))
        ax1.set_title("Target Label Distribution (Raw 3-Class)", fontsize=12, fontweight="bold")

        ax2 = axes[0, 1]
        bars2 = ax2.bar(rarity_order, rarity_values, color="#3498db", edgecolor="black", alpha=0.85)
        ax2.bar_label(bars2, padding=3)
        ax2.set_title("Entity Rarity in Wikipedia (cat[0])", fontsize=12, fontweight="bold")

        ax3 = axes[0, 2]
        ax3.hist(fact_lengths, bins=range(min(fact_lengths), max(fact_lengths) + 2), color="#9b59b6", edgecolor="black",
                 alpha=0.8, align="left")
        ax3.axvline(statistics.mean(fact_lengths), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {statistics.mean(fact_lengths):.2f}")
        ax3.set_title("Atomic Fact Word Length (Words/Fact)", fontsize=12, fontweight="bold")
        ax3.legend()

        ax4 = axes[1, 0]
        ax4.hist(passage_lengths, bins=25, color="#1abc9c", edgecolor="black", alpha=0.8)
        ax4.axvline(statistics.mean(passage_lengths), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {statistics.mean(passage_lengths):.1f}")
        ax4.set_title("Evidence Passage Length (Words/Passage)", fontsize=12, fontweight="bold")
        ax4.legend()

        ax5 = axes[1, 1]
        tokens, freqs = zip(*reversed(top_tokens))
        bars5 = ax5.barh(tokens, freqs, color="#e67e22", edgecolor="black", alpha=0.85)
        ax5.bar_label(bars5, padding=3)
        ax5.set_title("Top 10 Raw Fact Tokens (Uncleaned)", fontsize=12, fontweight="bold")

        ax6 = axes[1, 2]
        bp = ax6.boxplot([recalls_s, recalls_ns], tick_labels=["Supported (S)", "Not Supported (NS/IR)"],
                         patch_artist=True)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax6.set_title("Raw Token Recall (Before Preprocessing)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_dashboard.png"), dpi=200)
        plt.close()

    def compare_similarity_metrics(self, output_dir: str = "eda_plots"):
        import os
        import numpy as np
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

        metric_dict = {"Word Recall (Stem+Stopwords)": recall_scores, "Word Jaccard": jaccard_scores}

        results = []
        for name, scores in metric_dict.items():
            best_acc, best_thresh = 0.0, 0.0
            for t in np.linspace(0.01, 0.99, 100):
                preds = [1 if s >= t else 0 for s in scores]
                acc = sum(p == l for p, l in zip(preds, labels)) / total_ex
                if acc > best_acc:
                    best_acc, best_thresh = acc, t
            results.append({"name": name, "thresh": best_thresh, "acc": best_acc})

        print("\n--- Similarity Metrics ---")
        for r in results:
            print(f"{r['name']}: {r['acc'] * 100:.2f}% Acc at threshold {r['thresh']:.2f}")

    def run(self):
        self.load_data()
        self.print_topic_overview()
        self.print_label_distribution()
        self.print_length_analysis()
        self.print_vocabulary_analysis()
        self.compare_similarity_metrics()
        self.generate_plots()


if __name__ == "__main__":
    eda = DatasetEDA()
    eda.run()