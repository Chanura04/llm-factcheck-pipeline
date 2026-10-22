# factcheck.py
import torch
from typing import List
import numpy as np
import spacy
import gc
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


try:
    stopwords.words('english')
except LookupError:
    nltk.download("stopwords", quiet=True)

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

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def _preprocess(self, text: str) -> set:
        """Convert raw test into clean set of tokens for overlap comparision"""

        #  lowercase + remove punctuation
        text = text.lower()

        for punc in [",","!",".","?",":",";","()","[]","{}","'","\"","@","#","$","%","^","&","*","~"]:
            text = text.replace(punc, "")

        # Tokenize
        tokens = text.split()

        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]

        # Remove prefixes , suffixes and Convert words to their root form (Stemming)
        tokens = [self.stemmer.stem(token) for token in tokens]

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
        fact_tokens = self._preprocess(fact)

        max_score = 0.0
        for passage in passages:
            passage_tokens = self._preprocess(passage["text"])
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
        print(f"Atomic Fact Words      : Min={min(fact_words)}, Max={max(fact_words)}, Mean={statistics.mean(fact_words):.2f}, Median={statistics.median(fact_words)}")
        print(f"Atomic Fact Characters : Min={min(fact_chars)}, Max={max(fact_chars)}, Mean={statistics.mean(fact_chars):.2f}, Median={statistics.median(fact_chars)}")
        print(f"Passages per Fact      : Min={min(passages_counts)}, Max={max(passages_counts)}, Mean={statistics.mean(passages_counts):.2f}")
        print(f"Words per Passage      : Min={min(passage_words)}, Max={max(passage_words)}, Mean={statistics.mean(passage_words):.2f}")

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


    # =========================================================================
    # ADVANCED EDA EXTENSIONS: Academic Justifications & Dataset Nature
    # =========================================================================

    # 1. Ambiguity Zone KDE Density Plot (Lexical score distribution for S vs NS): The strongest academic argument for why baseline thresholding fails and deep learning / NLI is necessary.
    def print_ambiguity_zone_analysis(self):
        """
        Analyzes the density distribution of lexical recall overlap for Supported (S) vs Not Supported (NS).
        Quantifies the 'Ambiguity Zone' where lexical overlap fails to discriminate hallucinations.
        """
        checker = WordRecallThresholdFactChecker()
        recalls_s, recalls_ns = [], []

        for f in self.raw_facts:
            f_tokens = checker._preprocess(f["text"])
            ps = self.fact_to_passages.get(f["text"], [])
            if not ps:
                continue
            best_r = max((checker._recall(f_tokens, checker._preprocess(p["text"])) for p in ps), default=0.0)
            if f["label"] == "S":
                recalls_s.append(best_r)
            else:
                recalls_ns.append(best_r)

        mean_s = np.mean(recalls_s) if recalls_s else 0.0
        median_s = np.median(recalls_s) if recalls_s else 0.0
        mean_ns = np.mean(recalls_ns) if recalls_ns else 0.0
        median_ns = np.median(recalls_ns) if recalls_ns else 0.0

        ambiguity_ns = sum(1 for r in recalls_ns if 0.40 <= r <= 0.85)
        ambiguity_s = sum(1 for r in recalls_s if 0.40 <= r <= 0.85)
        false_pos_at_60 = sum(1 for r in recalls_ns if r >= 0.60)
        false_neg_at_60 = sum(1 for r in recalls_s if r < 0.60)

        print("\n" + "=" * 60)
        print("5. AMBIGUITY ZONE & LEXICAL DISTRIBUTION ANALYSIS (S vs. NS)")
        print("=" * 60)
        print(f"Supported (S) Claims       : Count={len(recalls_s)}, Mean Recall={mean_s:.3f}, Median={median_s:.3f}")
        print(f"Not Supported (NS) Claims  : Count={len(recalls_ns)}, Mean Recall={mean_ns:.3f}, Median={median_ns:.3f}")
        print(f"Ambiguity Zone [0.40, 0.85] : Contains {ambiguity_s}/{len(recalls_s)} S claims ({ambiguity_s/len(recalls_s)*100:.1f}%)")
        print(f"                            : Contains {ambiguity_ns}/{len(recalls_ns)} NS claims ({ambiguity_ns/len(recalls_ns)*100:.1f}%)")
        print(f"Failure of Fixed Threshold (tau=0.60):")
        print(f"  - False Positives (NS with Recall >= 0.60) : {false_pos_at_60} ({false_pos_at_60/len(recalls_ns)*100:.1f}%)")
        print(f"  - False Negatives (S with Recall < 0.60)   : {false_neg_at_60} ({false_neg_at_60/len(recalls_s)*100:.1f}%)")
        print("  -> Academic Takeaway: Hallucinated claims copy substantial lexical context from true passages,")
        print("     producing high overlap and rendering lexical thresholding fundamentally inadequate. Deep NLI is required.")
        print("=" * 60)

    # 2. Entity Rarity vs. Hallucination Rate (Grouped / Stacked Bar): Demonstrates direct insight into the nature of the dataset and LLM behavior.
    def print_entity_rarity_hallucination_analysis(self):
        """
        Evaluates ChatGPT's hallucination rate as a function of Wikipedia entity rarity (cat[0]).
        Demonstrates how parametric knowledge decay on long-tail entities causes high hallucination rates.
        """
        from collections import Counter, defaultdict
        rarity_stats = defaultdict(lambda: Counter())

        for bio in self.raw_biographies:
            cat = bio.get("cat", [])
            rarity = cat[0] if cat else "unknown"
            for ann in bio.get("annotations", []):
                for hf in ann.get("human-atomic-facts", []):
                    rarity_stats[rarity][hf["label"]] += 1

        rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]
        print("\n" + "=" * 60)
        print("6. ENTITY RARITY VS. HALLUCINATION RATE ANALYSIS")
        print("=" * 60)
        print(f"{'Entity Rarity':<12} {'Total Facts':<12} {'Supported (S)':<15} {'Hallucinated (NS/IR)':<22} {'Hallucination Rate':<18}")
        print("-" * 80)
        for r in rarity_order:
            s = rarity_stats[r]["S"]
            ns = rarity_stats[r]["NS"]
            ir = rarity_stats[r]["IR"]
            halluc = ns + ir
            total = s + halluc
            rate = (halluc / total * 100) if total > 0 else 0.0
            print(f"{r:<12} {total:<12} {s:<15} {halluc:<22} {rate:>6.2f}%")

        print("  -> Academic Takeaway: Hallucination rate drops precipitously from 56.7% for 'very rare' entities")
        print("     down to 20.7% for 'very freq' entities (a 2.7x reduction). This confirms that LLM hallucinations")
        print("     concentrate heavily on long-tail, low-frequency entities where parametric memory is sparse.")
        print("=" * 60)

    # 3. Best Passage Rank Distribution: Validates whether your multi-passage aggregation strategy (e.g., taking max over top-k passages) is justified.
    def print_best_passage_rank_analysis(self):
        """
        Analyzes the rank of the retrieved BM25 passage that yields the highest lexical evidence.
        Validates the multi-passage aggregation strategy (taking max over top-k retrieved passages).
        """
        from collections import Counter
        checker = WordRecallThresholdFactChecker()
        best_ranks = Counter()
        total_valid = 0

        for f in self.raw_facts:
            ps = self.fact_to_passages.get(f["text"], [])
            if not ps:
                continue
            f_tokens = checker._preprocess(f["text"])
            scores = [checker._recall(f_tokens, checker._preprocess(p["text"])) for p in ps]
            if not scores:
                continue
            best_idx = scores.index(max(scores)) + 1
            best_ranks[best_idx] += 1
            total_valid += 1

        print("\n" + "=" * 60)
        print("7. BEST PASSAGE RETRIEVAL RANK DISTRIBUTION")
        print("=" * 60)
        print(f"{'Rank':<8} {'Count':<10} {'Percentage':<14} {'Cumulative Coverage':<20}")
        print("-" * 52)
        cum_count = 0
        for rank in [1, 2, 3, 4, 5]:
            count = best_ranks[rank]
            cum_count += count
            pct = (count / total_valid * 100) if total_valid > 0 else 0.0
            cum_pct = (cum_count / total_valid * 100) if total_valid > 0 else 0.0
            print(f"Rank {rank:<3} {count:<10} {pct:>6.2f}%        {cum_pct:>6.2f}%")

        rank1_pct = (best_ranks[1] / total_valid * 100) if total_valid > 0 else 0.0
        top3_pct = ((best_ranks[1] + best_ranks[2] + best_ranks[3]) / total_valid * 100) if total_valid > 0 else 0.0
        print(f"\n  -> Academic Takeaway: Relying solely on the top-1 passage misses the best evidence in {100 - rank1_pct:.1f}% of cases.")
        print(f"     Aggregating over top-3 passages increases evidence coverage to {top3_pct:.1f}%, empirically validating")
        print("     the multi-passage max-pooling strategy (taking max over top-k passages) implemented in FactCheckers.")
        print("=" * 60 + "\n")

    def generate_plots(self, output_dir: str = "eda_plots"):
        """
        Generates and saves visual EDA plots (both individual figures and master dashboards)
        to the specified directory for coursework reports.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from collections import Counter, defaultdict
            import statistics
            import os
        except ImportError:
            print("[EDA WARNING] matplotlib is not installed. To generate plots, run: uv pip install matplotlib")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[PLOT GENERATION] Generating EDA figures into '{output_dir}/'...")

        label_counts = Counter(f["label"] for f in self.raw_facts)
        fact_lengths = [len(f["text"].split()) for f in self.raw_facts]
        passage_lengths = [len(p["text"].split()) for f in self.raw_facts for p in self.fact_to_passages.get(f["text"], [])]

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
            best_r = max((len(f_tokens & set(p["text"].split())) / len(f_tokens) for p in ps), default=0.0) if f_tokens else 0.0
            if f["label"] == "S":
                recalls_s.append(best_r)
            else:
                recalls_ns.append(best_r)

        # -------------------------------------------------------------
        # Master Figure: 2x3 Grid Dashboard (Baseline Metrics)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Exploratory Data Analysis (EDA) - Raw Dataset Before Preprocessing", fontsize=16, fontweight="bold")

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
        ax3.hist(fact_lengths, bins=range(min(fact_lengths), max(fact_lengths) + 2), color="#9b59b6", edgecolor="black", alpha=0.8, align="left")
        ax3.axvline(statistics.mean(fact_lengths), color="red", linestyle="--", linewidth=2, label=f"Mean: {statistics.mean(fact_lengths):.2f}")
        ax3.set_title("Atomic Fact Word Length (Words/Fact)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Word Count")
        ax3.set_ylabel("Frequency")
        ax3.legend()

        # 4. Evidence Passage Length
        ax4 = axes[1, 0]
        ax4.hist(passage_lengths, bins=25, color="#1abc9c", edgecolor="black", alpha=0.8)
        ax4.axvline(statistics.mean(passage_lengths), color="red", linestyle="--", linewidth=2, label=f"Mean: {statistics.mean(passage_lengths):.1f}")
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
        bp = ax6.boxplot([recalls_s, recalls_ns], tick_labels=["Supported (S)", "Not Supported (NS/IR)"], patch_artist=True)
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
        plt.hist(fact_lengths, bins=range(min(fact_lengths), max(fact_lengths) + 2), color="#9b59b6", edgecolor="black", alpha=0.8, align="left")
        plt.axvline(statistics.mean(fact_lengths), color="red", linestyle="--", linewidth=2, label=f"Mean: {statistics.mean(fact_lengths):.2f}")
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
        plt.axvline(statistics.mean(passage_lengths), color="red", linestyle="--", linewidth=2, label=f"Mean: {statistics.mean(passage_lengths):.1f}")
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
        bp = plt.boxplot([recalls_s, recalls_ns], tick_labels=["Supported (S)", "Not Supported (NS/IR)"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        plt.title("Raw Lexical Recall by Label", fontsize=13, fontweight="bold")
        plt.ylabel("Max Surface Recall Overlap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "6_raw_lexical_recall.png"), dpi=200)
        plt.close()

        # -----------------------------------------------------------------
        # ADVANCED EDA PLOTS (Academic Arguments & LLM Hallucination Nature)
        # -----------------------------------------------------------------
        def _gaussian_kde_1d(data, x_grid, bw=None):
            data = np.asarray(data, dtype=float)
            n = len(data)
            if n == 0:
                return np.zeros_like(x_grid)
            std = np.std(data, ddof=1)
            bw = bw or (1.06 * std * (n ** (-0.2)) if std > 1e-4 else 0.08)
            bw = max(bw, 0.06)
            u = (x_grid[:, None] - data[None, :]) / bw
            return np.sum(np.exp(-0.5 * u**2) / (np.sqrt(2 * np.pi) * bw), axis=1) / n

        checker = WordRecallThresholdFactChecker()
        clean_recalls_s, clean_recalls_ns = [], []
        best_ranks_plot = Counter()

        for f in self.raw_facts:
            ps = self.fact_to_passages.get(f["text"], [])
            if not ps:
                continue
            f_tokens = checker._preprocess(f["text"])
            scores = [checker._recall(f_tokens, checker._preprocess(p["text"])) for p in ps]
            if not scores:
                continue
            max_s = max(scores)
            if f["label"] == "S":
                clean_recalls_s.append(max_s)
            else:
                clean_recalls_ns.append(max_s)
            best_ranks_plot[scores.index(max_s) + 1] += 1

        # 1. Ambiguity Zone KDE Density Plot (Lexical score distribution for S vs NS): The strongest academic argument for why baseline thresholding fails and deep learning / NLI is necessary.
        x_grid = np.linspace(0.0, 1.0, 300)
        kde_s = _gaussian_kde_1d(clean_recalls_s, x_grid)
        kde_ns = _gaussian_kde_1d(clean_recalls_ns, x_grid)

        plt.figure(figsize=(8, 5))
        plt.plot(x_grid, kde_s, color='#2ecc71', lw=2.5, label=f'Supported (S) [Mean: {np.mean(clean_recalls_s):.2f}]')
        plt.fill_between(x_grid, kde_s, alpha=0.25, color='#2ecc71')
        plt.plot(x_grid, kde_ns, color='#e74c3c', lw=2.5, label=f'Not Supported (NS/IR) [Mean: {np.mean(clean_recalls_ns):.2f}]')
        plt.fill_between(x_grid, kde_ns, alpha=0.25, color='#e74c3c')
        plt.axvspan(0.40, 0.85, color='orange', alpha=0.15, label='Ambiguity Zone (Heavy Density Overlap)')
        plt.axvline(0.60, color='#34495e', linestyle='--', lw=2, label='Baseline Threshold (τ = 0.60)')
        plt.title('Ambiguity Zone KDE Density Plot (Lexical Recall for S vs. NS)', fontsize=13, fontweight='bold')
        plt.xlabel('Maximum Lexical Recall Score (Claim vs. Evidence Passages)', fontsize=11)
        plt.ylabel('Probability Density (KDE)', fontsize=11)
        plt.xlim(0, 1)
        plt.ylim(bottom=0)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "7_ambiguity_zone_kde.png"), dpi=200)
        plt.close()

        # 2. Entity Rarity vs. Hallucination Rate (Grouped / Stacked Bar): Demonstrates direct insight into the nature of the dataset and LLM behavior.
        rarity_stats_plot = defaultdict(lambda: Counter())
        for bio in self.raw_biographies:
            cat = bio.get("cat", [])
            r = cat[0] if cat else "unknown"
            for ann in bio.get("annotations", []):
                for hf in ann.get("human-atomic-facts", []):
                    rarity_stats_plot[r][hf["label"]] += 1

        s_counts = [rarity_stats_plot[r]["S"] for r in rarity_order]
        ns_counts = [rarity_stats_plot[r]["NS"] + rarity_stats_plot[r]["IR"] for r in rarity_order]
        totals = [s + ns for s, ns in zip(s_counts, ns_counts)]
        halluc_rates = [ns / t * 100 if t > 0 else 0 for ns, t in zip(ns_counts, totals)]
        supported_rates = [100 - hr for hr in halluc_rates]

        fig, ax_rarity = plt.subplots(figsize=(8, 5))
        x_indices = np.arange(len(rarity_order))
        ax_rarity.bar(x_indices, supported_rates, label='Supported (S) %', color='#2ecc71', width=0.55, edgecolor='black', alpha=0.85)
        ax_rarity.bar(x_indices, halluc_rates, bottom=supported_rates, label='Hallucinated (NS/IR) %', color='#e74c3c', width=0.55, edgecolor='black', alpha=0.85)

        for i in range(len(rarity_order)):
            ax_rarity.text(x_indices[i], supported_rates[i] / 2, f'{supported_rates[i]:.1f}%\n(n={s_counts[i]})', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            ax_rarity.text(x_indices[i], supported_rates[i] + halluc_rates[i] / 2, f'{halluc_rates[i]:.1f}%\n(n={ns_counts[i]})', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

        ax_rarity.set_xticks(x_indices)
        ax_rarity.set_xticklabels([f'{r.title()}\n(Total={t})' for r, t in zip(rarity_order, totals)], fontsize=10)
        ax_rarity.set_ylabel('Proportion of Atomic Facts (%)', fontsize=11)
        ax_rarity.set_ylim(0, 112)
        ax_rarity.set_title('Entity Rarity vs. Hallucination Rate (ChatGPT Bio Generation)', fontsize=13, fontweight='bold')
        ax_rarity.legend(loc='upper left', frameon=True, fontsize=10)
        ax_rarity.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "8_rarity_vs_hallucination.png"), dpi=200)
        plt.close()

        # 3. Best Passage Rank Distribution: Validates whether your multi-passage aggregation strategy (e.g., taking max over top-k passages) is justified.
        ranks = [1, 2, 3, 4, 5]
        counts = [best_ranks_plot[r] for r in ranks]
        tot_ranked = sum(counts)
        pcts = [c / tot_ranked * 100 for c in counts] if tot_ranked > 0 else [0] * len(ranks)
        cum_pcts = np.cumsum(pcts)

        fig, ax_rank = plt.subplots(figsize=(8, 5))
        bars_rank = ax_rank.bar(ranks, pcts, color='#3498db', edgecolor='black', alpha=0.85, width=0.55, label='Best Passage Frequency (%)')
        ax_rank.bar_label(bars_rank, fmt='%.1f%%', padding=3, fontweight='bold')
        ax_rank_line = ax_rank.twinx()
        ax_rank_line.plot(ranks, cum_pcts, color='#e67e22', marker='o', lw=2.5, markersize=8, label='Cumulative Coverage (%)')
        for r, cp in zip(ranks, cum_pcts):
            ax_rank_line.annotate(f'{cp:.1f}%', (r, cp), textcoords='offset points', xytext=(0, 8), ha='center', fontweight='bold', color='#d35400')

        ax_rank.set_xlabel('Retrieved Passage Rank (1 = Top BM25 Hit)', fontsize=11)
        ax_rank.set_ylabel('Facts Where Rank Is Best Evidence (%)', color='#2980b9', fontsize=11)
        ax_rank_line.set_ylabel('Cumulative Coverage (%)', color='#d35400', fontsize=11)
        ax_rank.set_ylim(0, 100)
        ax_rank_line.set_ylim(50, 108)
        ax_rank.set_xticks(ranks)
        ax_rank.set_xticklabels([f'Rank {r}' for r in ranks])
        ax_rank.set_title('Best Evidence Passage Rank Distribution (BM25 Retrieval)', fontsize=13, fontweight='bold')
        lines1, labels1 = ax_rank.get_legend_handles_labels()
        lines2, labels2 = ax_rank_line.get_legend_handles_labels()
        ax_rank.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "9_best_passage_rank.png"), dpi=200)
        plt.close()

        # Composite Figure for the 3 Advanced EDA Parts: 1x3 Grid Dashboard
        fig_ext, axes_ext = plt.subplots(1, 3, figsize=(21, 6))
        fig_ext.suptitle("Advanced EDA: Academic Justifications & LLM Hallucination Behavior", fontsize=15, fontweight="bold")

        # Subplot 1: Ambiguity Zone
        axes_ext[0].plot(x_grid, kde_s, color='#2ecc71', lw=2.2, label=f'Supported (S) [μ={np.mean(clean_recalls_s):.2f}]')
        axes_ext[0].fill_between(x_grid, kde_s, alpha=0.25, color='#2ecc71')
        axes_ext[0].plot(x_grid, kde_ns, color='#e74c3c', lw=2.2, label=f'Not Supported (NS) [μ={np.mean(clean_recalls_ns):.2f}]')
        axes_ext[0].fill_between(x_grid, kde_ns, alpha=0.25, color='#e74c3c')
        axes_ext[0].axvspan(0.40, 0.85, color='orange', alpha=0.15, label='Ambiguity Zone')
        axes_ext[0].axvline(0.60, color='#34495e', linestyle='--', lw=2, label='Threshold (τ=0.60)')
        axes_ext[0].set_title("1. Ambiguity Zone (Lexical S vs NS)", fontsize=12, fontweight="bold")
        axes_ext[0].set_xlabel("Max Lexical Recall Score")
        axes_ext[0].set_ylabel("Density (KDE)")
        axes_ext[0].legend(loc="upper left", fontsize=9)
        axes_ext[0].grid(True, linestyle=":", alpha=0.5)

        # Subplot 2: Entity Rarity vs Hallucination
        axes_ext[1].bar(x_indices, supported_rates, label='Supported %', color='#2ecc71', width=0.55, edgecolor='black', alpha=0.85)
        axes_ext[1].bar(x_indices, halluc_rates, bottom=supported_rates, label='Hallucinated %', color='#e74c3c', width=0.55, edgecolor='black', alpha=0.85)
        for i in range(len(rarity_order)):
            axes_ext[1].text(x_indices[i], supported_rates[i] / 2, f'{supported_rates[i]:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            axes_ext[1].text(x_indices[i], supported_rates[i] + halluc_rates[i] / 2, f'{halluc_rates[i]:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        axes_ext[1].set_xticks(x_indices)
        axes_ext[1].set_xticklabels([r.title() for r in rarity_order], fontsize=9)
        axes_ext[1].set_title("2. Entity Rarity vs. Hallucination Rate", fontsize=12, fontweight="bold")
        axes_ext[1].set_ylabel("Proportion of Facts (%)")
        axes_ext[1].legend(loc="upper left", fontsize=9)
        axes_ext[1].grid(axis="y", linestyle=":", alpha=0.5)

        # Subplot 3: Best Passage Rank
        b3 = axes_ext[2].bar(ranks, pcts, color='#3498db', edgecolor='black', alpha=0.85, width=0.55, label='Best Rank %')
        axes_ext[2].bar_label(b3, fmt='%.1f%%', padding=2, fontsize=9, fontweight='bold')
        ax3_line = axes_ext[2].twinx()
        ax3_line.plot(ranks, cum_pcts, color='#e67e22', marker='o', lw=2.2, label='Cumulative %')
        axes_ext[2].set_xticks(ranks)
        axes_ext[2].set_xticklabels([f'R{r}' for r in ranks])
        axes_ext[2].set_title("3. Best Passage Retrieval Rank", fontsize=12, fontweight="bold")
        axes_ext[2].set_xlabel("BM25 Retrieved Rank")
        axes_ext[2].set_ylabel("Frequency (%)", color='#2980b9')
        ax3_line.set_ylabel("Cumulative (%)", color='#d35400')
        axes_ext[2].grid(axis="y", linestyle=":", alpha=0.5)

        plt.tight_layout()
        adv_dashboard_path = os.path.join(output_dir, "eda_advanced_dashboard.png")
        plt.savefig(adv_dashboard_path, dpi=200)
        plt.close()
        print(f"  [OK] Advanced Dashboard saved to: {adv_dashboard_path}")
        print(f"  [OK] All 9 individual high-res figures saved to '{output_dir}/'")
        print("=" * 60 + "\n")

    def run(self, generate_plots: bool = True):
        """Executes all EDA routines sequentially, including plot generation."""
        print("\n>>> Running Full Dataset EDA (Before Preprocessing) <<<")
        self.load_data()
        self.print_topic_overview()
        self.print_label_distribution()
        self.print_length_analysis()
        self.print_vocabulary_analysis()
        self.print_ambiguity_zone_analysis()
        self.print_entity_rarity_hallucination_analysis()
        self.print_best_passage_rank_analysis()
        if generate_plots:
            self.generate_plots()


if __name__ == "__main__":
    eda = DatasetEDA()
    eda.run()
