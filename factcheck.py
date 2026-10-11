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
