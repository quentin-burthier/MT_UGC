"""Functions allowing to compare the lexicons of different datasets."""

from typing import Tuple, Dict
import os
from os.path import join
import itertools
from collections import Counter, defaultdict
from random import sample
import pandas as pd

def oov_grid(src="fr", tgt="en", src_side=True, out_format="") -> pd.DataFrame:
    """Prints the OOV table.

    Args:
        src (str): source language. Default: fr
        tgt (en): source language. Default: en
        src_side (bool): compares source side distribution. Default: True
        out_format (str): tex, csv, or default pandas print
    """

    side = src if src_side else tgt

    train_sets = [
        ("EuParl", f"Europarl.{src}-{tgt}"),
        ("EuParl small", f"Europarl_small.{src}-{tgt}"),
        ("News", f"News-Commentary.{src}-{tgt}"),
        ("OpenSub", f"OpenSubtitles.{src}-{tgt}"),
        ("OpenSub small", f"OpenSubtitles_small.{src}-{tgt}"),
    ]
    dev_sets = [
        ("EuParl", f"Europarl.{src}-{tgt}"),
        ("News", f"News-Commentary.{src}-{tgt}"),
        ("OpenSub", f"OpenSubtitles.{src}-{tgt}"),
        ("MTNT", join("MTNT_reshuffled", f"{src}-{tgt}.1.0")),
        ("PFSMB", "Crapbank"),
        ("4SQ", "Foursquare"),
    ]

    train_lexicons = {train_set: build_vocab(join(os.environ["DATA"], train_path,
                                             "preprocessed", f"train.{side}"))
                      for train_set, train_path in train_sets}

    dev_lexicons = {dev_set: build_vocab(join(os.environ["DATA"], dev_path,
                                            "preprocessed", f"dev.{side}"))
                    for dev_set, dev_path in dev_sets}

    df = oov_grid_dataframe(train_lexicons, dev_lexicons)

    dev_sets_names = [dev_set for dev_set, _ in dev_sets]

    set_order = ["EuParl", "EuParl small",
                 "News",
                 "OpenSub", "OpenSub small",
                 "MTNT", "PFSMB", "4SQ"]
    set_order_key = defaultdict(int, {dataset: i for i, dataset in enumerate(set_order)})
    def index_sort_key(index):
        return [set_order_key[idx] for idx in index]

    df = df.sort_index(key=index_sort_key)

    if out_format == "tex":
        print(df["oov"].unstack()[dev_sets_names].to_latex(float_format="%.1f"), end="\n\n")
        print(df["unique_oov"].unstack()[dev_sets_names].to_latex(float_format="%.1f"))
    if out_format == "csv":
        print(df["oov"].unstack()[dev_sets_names].to_csv(), end="\n\n")
        print(df["unique_oov"].unstack()[dev_sets_names].to_csv())
    else:
        print(df["oov"].unstack()[dev_sets_names], end="\n\n")
        print(df["unique_oov"].unstack()[dev_sets_names])

def oov_grid_dataframe(train_lexicons: Dict[str, Counter],
                       dev_lexicons: Dict[str, Counter]):
    """Builds the OOV dataframe."""
    indexes = pd.MultiIndex.from_product([train_lexicons.keys(),
                                          dev_lexicons.keys()])
    df = pd.DataFrame(index=indexes)
    for train_name, train_lexicon in train_lexicons.items():
        for dev_name, dev_lexicon in dev_lexicons.items():
            _, p_oov, _, p_unique_oov = count_oov(train_lexicon, dev_lexicon, 0)
            df.loc[(train_name, dev_name), "oov"] = 100 * p_oov
            df.loc[(train_name, dev_name), "unique_oov"] = 100 * p_unique_oov

    return df


def compare_lexicons(train_set: str, other_set: str,
                     n_printed_oov=20):
    """Computes the proportions of words in new set not present in train set."""
    train_lexicon = build_vocab(train_set)
    new_lexicon = build_vocab(other_set)

    n_oov, p_oov, n_uniq_oov, p_uniq_oov = count_oov(train_lexicon, new_lexicon,
                                                       n_printed_oov)

    print(f"{n_oov} oov tokens, ({100*p_oov:.2f} %)")
    print(f"{n_uniq_oov} unique oov tokens, ({100*p_uniq_oov:.2f} %)")
    print()


def count_oov(train_lexicon: Counter, new_lexicon: Counter,
              n_printed_oov=20) -> Tuple[int, float, int, float]:
    """Counts the number of out of train lexicon vocabulary tokens in the new_lexicon.

    Returns:
        n_oov (int)
        proportion_oov (float): n_oov / n_tokens
        n_unique_oov (int)
        proportion_unique_oov (float): n_unique_oov / n_unique_tokens
    """
    oov_tokens = [token for token in new_lexicon.keys()
                  if token not in train_lexicon]
    n_oov = sum(new_lexicon[token] for token in oov_tokens)
    proportion_oov = n_oov / sum(new_lexicon.values())

    n_unique_oov = len(oov_tokens)
    proportion_unique_oov = n_unique_oov / len(new_lexicon)

    if n_printed_oov > 0:
        n_printed_oov = min(n_printed_oov, len(oov_tokens))
        print(" ".join(sample(oov_tokens, n_printed_oov)))

    return n_oov, proportion_oov, n_unique_oov, proportion_unique_oov


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
        compare_lexicons(sys.argv[1], sys.argv[2], 200)
    else:
        oov_grid(src_side=True, to_latex=True)
