"""Functions allowing to compare the lexicons of two different files."""

from typing import Tuple
import os
from os.path import join
import itertools
from collections import Counter
from random import sample

def oov_grid(src: str = "en", tgt: str = "fr"):
    """Prints the OOV table."""
    train_sets = [
        ("Euro", f"Europarl.{src}-{tgt}"),
        ("4square", "Foursquare"),
        ("MTNT", join("MTNT_reshuffled", f"{src}-{tgt}.1.0")),
        ("News", f"News-Commentary.{src}-{tgt}"),
        ("OSub", f"OpenSubtitles.{src}-{tgt}"),
    ]
    dev_only_sets = [
        ("CrapB", "Crapbank"),
    ]
    dev_sets = list(itertools.chain(train_sets, dev_only_sets))

    dev_lexicons = [build_vocab(join(os.environ["DATA"], dev_set,
                                     "preprocessed", f"dev.{src}"))
                    for _, dev_set in dev_sets]

    print("Tr\\Dev \t", "\t & \t ".join(dev_name for dev_name, _ in dev_sets))
    for train_name, train_set in train_sets:
        train_lexicon = build_vocab(join(os.environ["DATA"], train_set,
                                         "preprocessed", f"train.{src}"))
        print(train_name, end="\t")
        for dev_lexicon in dev_lexicons:
            n_oov, _ = count_oov(train_lexicon, dev_lexicon, 0)
            proportion_oov = n_oov / sum(dev_lexicon.values())
            print(f"& \t {100*proportion_oov:.2f}", end="")
        print("\\\\")


def compare_lexicons(train_set: str, other_set: str,
                     n_printed_oov=20, verbose=True) -> Tuple[int, int, float]:
    """Computes the proportions of words in new set not present in train set."""
    train_lexicon = build_vocab(train_set)
    new_lexicon = build_vocab(other_set)

    n_oov, n_unique_oov = count_oov(train_lexicon, new_lexicon, n_printed_oov)
    proportion_oov = n_oov / sum(new_lexicon.values())

    if verbose:
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
    if len(sys.argv) == 3:
        compare_lexicons(sys.argv[1], sys.argv[2])
    else:
        oov_grid()
