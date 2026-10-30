# scratch/verify_code_diff.py
import inspect
import difflib
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from factcheck import WordRecallThresholdFactChecker


def main():
    s_source = inspect.getsource(WordRecallThresholdFactChecker._preprocess)
    l_source = inspect.getsource(WordRecallThresholdFactChecker._preprocess_with_lemm)

    s_lines = [line.strip() for line in s_source.splitlines() if line.strip()]
    l_lines = [line.strip() for line in l_source.splitlines() if line.strip()]

    diff = difflib.unified_diff(s_lines, l_lines, fromfile='_preprocess (Stemming)',
                                tofile='_preprocess_with_lemm (Lemmatization)', lineterm='')

    print("=" * 60)
    print("CODE DIFF BETWEEN STEMMING AND LEMMATIZATION PIPELINES")
    print("=" * 60)
    for line in diff:
        print(line)


if __name__ == "__main__":
    main()