"""Functions allowing to compare the lexicons of two different files."""

from collections import Counter
from typing import Tuple
from random import sample

def compare_lexicons(train_set: str, other_set: str) -> Tuple[int, int, float]:
    """Computes the proportions of words in new set not present in train set."""
    train_lexicon = build_vocab(train_set)
    new_lexicon = build_vocab(other_set)

    n_oov, n_unique_oov = count_oov(train_lexicon, new_lexicon)
    proportion_oov = n_oov / sum(new_lexicon.values())

    print(f"{100*proportion_oov:.2f} % of oov tokens in {other_set} "
          f"({n_oov} oov tokens, {n_unique_oov} unique oov tokens)")
    print()

    return n_oov, n_unique_oov, proportion_oov

def count_oov(train_lexicon: Counter, new_lexicon: Counter,
              n_printed_oov=20) -> Tuple[int, int]:
    """Counts the number of out of train lexicon vocabulary tokens in the new_lexicon."""
    oov_tokens = [token for token in new_lexicon.keys()
                  if token not in train_lexicon]
    n_unique_oov = len(oov_tokens)
    n_oov = sum(new_lexicon[token] for token in oov_tokens)
    if n_printed_oov > 0:
        print(sample(oov_tokens, n_printed_oov))
    return n_oov, n_unique_oov

def build_vocab(file: str) -> Counter:
    """Builds a lexicon from the given file.

    Uses a whitespace tokenization. It is therefore better to compare
    already tokenized text, to at at least avoid counting oov that are
    a known word followed by punctuation.
    """
    with open(file, "r") as f:
        lexicon = Counter(f.read().split())
    return lexicon


if __name__ == "__main__":
    import sys
    compare_lexicons(sys.argv[1], sys.argv[2])
