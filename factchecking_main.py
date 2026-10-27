# factscore_retrieval_interface.py

import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from tqdm import tqdm
import argparse
from factcheck import *


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        help="choose from [random', 'always_entail', 'word_overlap', 'parsing', 'entailment]")
    parser.add_argument('--labels_path', type=str, default="data/dev_labeled_ChatGPT.jsonl", help="path to the labels")
    parser.add_argument('--passages_path', type=str, default="data/passages_bm25_ChatGPT_humfacts.jsonl",
                        help="path to the passages retrieved for the ChatGPT human-labeled facts")
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true',
                        help='skip printing output on the test set')
    args = parser.parse_args()
    return args


def read_passages(path: str):
    """
    Reads the retrieved passages and puts then in a dictionary mapping facts to passages.
    :param path: path to the cached passages
    :return: dict mapping facts (strings) to passages
    """
    fact_to_passage_dict = {}
    with open(path, 'r') as file:
        all_lines = file.readlines()
        for nextline in all_lines:
            dict = json.loads(nextline)
            name = dict["name"]
            for passage in dict["passages"]:
                if passage["title"] != name:
                    raise Exception("Couldn't find a match for: " + name + " " + passage["title"])
            fact_to_passage_dict[dict["sent"]] = dict["passages"]
    return fact_to_passage_dict


def read_fact_examples(labeled_facts_path: str, fact_to_passage_dict: Dict):
    """
    Reads the labeled fact examples and constructs FactExample objects associating labeled, human-annotated facts
    with their corresponding passages
    :param labeled_facts_path: path to the list of labeled
    :param fact_to_passage_dict: the dict mapping facts to passages (see load_passages)
    :return: a list of FactExample objects to use as our dataset
    """
    examples = []
    with open(labeled_facts_path, 'r') as file:
        all_lines = file.readlines()
        for nextline in all_lines:
            dict = json.loads(nextline)
            if dict["annotations"] is not None:
                for sent in dict["annotations"]:
                    if sent["human-atomic-facts"] is None:
                        # Should never be the case, but just in case
                        print("No facts! Skipping this one: " + repr(sent))
                    else:
                        for fact in sent["human-atomic-facts"]:
                            if fact["text"] not in fact_to_passage_dict:
                                # Should never be the case, but just in case
                                print("Missing fact: " + fact["text"])
                            else:
                                examples.append(
                                    FactExample(fact["text"], fact_to_passage_dict[fact["text"]], fact["label"]))
    return examples


def predict_two_classes(examples: List[FactExample], fact_checker, return_accuracy: bool = False):
    """
    Compares against ground truth which is just the labels S and NS (IR is mapped to NS).
    Makes predictions and prints evaluation statistics on this setting.
    """
    gold_label_indexer = ["S", "NS"]
    confusion_mat = [[0, 0], [0, 0]]
    ex_count = 0

    for i, example in enumerate(tqdm(examples)):
        converted_label = "NS" if example.label == 'IR' else example.label
        gold_label = gold_label_indexer.index(converted_label)

        raw_pred = fact_checker.predict(example.fact, example.passages)
        pred_label = gold_label_indexer.index(raw_pred)

        confusion_mat[gold_label][pred_label] += 1
        ex_count += 1

    print_eval_stats(confusion_mat, gold_label_indexer)

    if return_accuracy:
        correct_preds = confusion_mat[0][0] + confusion_mat[1][1]
        total_preds = sum(sum(row) for row in confusion_mat)
        return correct_preds / total_preds if total_preds > 0 else 0.0
    return None


def print_eval_stats(confusion_mat, gold_label_indexer):
    """
    Takes a confusion matrix and the label indexer and prints accuracy and per-class F1
    :param confusion_mat: The confusion matrix, indexed as [gold_label, pred_label]
    :param gold_label_indexer: The Indexer for the labels as a List, not an Indexer
    """
    for row in confusion_mat:
        print("\t".join([repr(item) for item in row]))
    correct_preds = sum([confusion_mat[i][i] for i in range(0, len(gold_label_indexer))])
    total_preds = sum(
        [confusion_mat[i][j] for i in range(0, len(gold_label_indexer)) for j in range(0, len(gold_label_indexer))])
    print("Accuracy: " + repr(correct_preds) + "/" + repr(total_preds) + " = " + repr(correct_preds / total_preds))
    for idx in range(0, len(gold_label_indexer)):
        num_correct = confusion_mat[idx][idx]
        num_gold = sum([confusion_mat[idx][i] for i in range(0, len(gold_label_indexer))])
        num_pred = sum([confusion_mat[i][idx] for i in range(0, len(gold_label_indexer))])
        rec = num_correct / num_gold
        if num_pred > 0:
            prec = num_correct / num_pred
            f1 = 2 * prec * rec / (prec + rec)
        else:
            prec = "undefined"
            f1 = "undefined"
        print("Prec for " + gold_label_indexer[idx] + ": " + repr(num_correct) + "/" + repr(num_pred) + " = " + repr(
            prec))
        print(
            "Rec for " + gold_label_indexer[idx] + ": " + repr(num_correct) + "/" + repr(num_gold) + " = " + repr(rec))
        print("F1 for " + gold_label_indexer[idx] + ": " + repr(f1))


def compare_preprocessing_methods(examples: List[FactExample]):
    """
    Runs word_overlap twice — once with Porter stemming, once with WordNet
    lemmatization — prints both confusion matrices, then CLEARLY states
    which preprocessing gives the higher accuracy.
    """
    print("\n" + "=" * 72)
    print("  WORD_OVERLAP  —  PREPROCESSING COMPARISON")
    print("  Stemming (PorterStemmer)  vs  Lemmatization (WordNetLemmatizer)")
    print("=" * 72)

    print("\n>>> [1 / 2]  STEMMING  —  running...\n")
    stem_checker = WordRecallThresholdFactChecker(use_lemmatization=False)
    stem_acc = predict_two_classes(examples, stem_checker, return_accuracy=True)

    print("\n>>> [2 / 2]  LEMMATIZATION  —  running...\n")
    lem_checker = WordRecallThresholdFactChecker(use_lemmatization=True)
    lem_acc = predict_two_classes(examples, lem_checker, return_accuracy=True)

    # ---------------- final verdict ----------------
    print("\n" + "=" * 72)
    print("  FINAL RESULT  —  WHICH PREPROCESSING IS MORE ACCURATE?")
    print("=" * 72)
    print(f"  Stemming        accuracy : {stem_acc * 100:6.2f}%   ({stem_acc:.4f})")
    print(f"  Lemmatization   accuracy : {lem_acc * 100:6.2f}%   ({lem_acc:.4f})")
    print("-" * 72)

    if lem_acc > stem_acc:
        print(f"  ==> LEMMATIZATION wins  (+{(lem_acc - stem_acc) * 100:.2f} pp)")
        print(f"  ==> HIGHEST ACCURACY: Lemmatization = {lem_acc * 100:.2f}%")
    elif stem_acc > lem_acc:
        print(f"  ==> STEMMING wins       (+{(stem_acc - lem_acc) * 100:.2f} pp)")
        print(f"  ==> HIGHEST ACCURACY: Stemming      = {stem_acc * 100:.2f}%")
    else:
        print(f"  ==> TIE — both methods hit the same accuracy = {stem_acc * 100:.2f}%")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    fact_to_passage_dict = read_passages(args.passages_path)

    examples = read_fact_examples(args.labels_path, fact_to_passage_dict)
    print("Read " + repr(len(examples)) + " examples")
    print("Fact and length of passages for each fact:")
    for example in examples:
        print(example.fact + ": " + repr([len(p["text"]) for p in example.passages]))

    assert args.mode in ['random', 'always_entail', 'word_overlap', 'parsing', 'entailment'], "invalid method"
    print(f"Method: {args.mode}")

    fact_checker = None
    if args.mode == "random":
        fact_checker = RandomGuessFactChecker()
    elif args.mode == "always_entail":
        fact_checker = AlwaysEntailedFactChecker()
    elif args.mode == "word_overlap":
        # Run BOTH preprocessors, print both, and clearly state the winner.
        compare_preprocessing_methods(examples)
        # fact_checker stays None → final single-run is skipped below.
    elif args.mode == "parsing":
        fact_checker = DependencyRecallThresholdFactChecker()
    elif args.mode == "entailment":
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
        roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if args.cuda:
            roberta_ent_model.to('cuda')
        ent_model = EntailmentModel(roberta_ent_model, ent_tokenizer, args.cuda)
        fact_checker = EntailmentFactChecker(ent_model)
    else:
        raise NotImplementedError

    if fact_checker is not None:
        predict_two_classes(examples, fact_checker)