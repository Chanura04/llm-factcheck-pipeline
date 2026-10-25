import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from factcheck import WordRecallThresholdFactChecker


def main():
    checker = WordRecallThresholdFactChecker(use_lemmatization=False)

    sentences = [
        "The dogs were running quickly.",
        "She studies hard and studied harder.",
        "The universities produced universal knowledge.",
        "Two geese and one goose were flying.",
    ]

    print("=" * 70)
    print("DAY 1 SANITY CHECK: _preprocess vs _preprocess_with_lemm")
    print("=" * 70)

    for s in sentences:
        stem_tokens = checker._preprocess(s)
        lem_tokens = checker._preprocess_with_lemm(s)

        print(f"\nSent: {s}")
        print(f"  stem (Porter)      : {sorted(stem_tokens)}")
        print(f"  lem  (WordNet)     : {sorted(lem_tokens)}")
        if stem_tokens == lem_tokens:
            print("  >>> IDENTICAL — suspicious. Confirm wordnet is installed.")
        else:
            print("  >>> DIFFERENT — good, lemmatization is active.")

    print("\n" + "=" * 70)
    print("Quick wordnet probe (must print 'dog', not 'dogs'):")
    from nltk.stem import WordNetLemmatizer
    print("  lemmatize('dogs') =", WordNetLemmatizer().lemmatize("dogs"))
    print("=" * 70)


if __name__ == "__main__":
    main()