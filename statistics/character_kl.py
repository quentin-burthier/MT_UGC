"""Functions allowing to compare the KL divergence of character ngrams distribution
of different corpora."""

from typing import Tuple, Dict
import os
from os.path import join
from collections import Counter, defaultdict
from random import sample

from itertools import chain
from more_itertools import windowed

from scipy.stats import entropy
import numpy as np
import pandas as pd

def KL_grid(src="fr", tgt="en", src_side=True, out_format="") -> pd.DataFrame:
    """Prints the Kullback-Leibler divergence of 3-grams characters distributions table.
    
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

    train_ngram_counts = {train_set: count_ngrams(join(os.environ["DATA"], train_path,
                                                       "preprocessed", f"train.{side}"))
                          for train_set, train_path in train_sets}

    dev_ngram_counts = {dev_set: count_ngrams(join(os.environ["DATA"], dev_path,
                                                   "preprocessed", f"dev.{side}"))
                        for dev_set, dev_path in dev_sets}

    df = KL_grid_dataframe(train_ngram_counts, dev_ngram_counts)

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
        print(df["KL"].unstack()[dev_sets_names].to_latex(float_format="%.2f"))
    elif out_format == "csv":
        print(df["KL"].unstack()[dev_sets_names].to_csv())
    else:
        print(df["KL"].unstack()[dev_sets_names])


def KL_grid_dataframe(train_ngram_counts: Dict[str, Counter],
                      dev_ngram_counts: Dict[str, Counter]) -> pd.DataFrame:
    """Builds the OOV dataframe."""
    indexes = pd.MultiIndex.from_product([train_ngram_counts.keys(), dev_ngram_counts.keys()])
    df = pd.DataFrame(index=indexes)
    for train_name, train_counts in train_ngram_counts.items():
        for dev_name, dev_counts in dev_ngram_counts.items():
            df.loc[(train_name, dev_name), "KL"] = counts_KL(train_counts, dev_counts)
    return df

def counts_KL(train_counts: Counter, dev_counts: Counter, smoothing_factor=1) -> float:
    """Computes the Kullback-Leibler divergence KL(train_freq || dev_freq)."""
    counts = np.array([(train_counts[ngram], dev_counts[ngram])
                       for ngram in train_counts.keys() | dev_counts.keys()])
    counts += smoothing_factor
    return entropy(counts[:, 1], counts[:, 0])


def count_ngrams(file: str, n=3) -> Counter:
    """Counts the number of appearance of each character n-gram in the file."""
    n_grams_count = Counter()
    with open(file, "r") as f:
        for line in f:
            n_grams_count.update(windowed(list(line), n))

    return n_grams_count


if __name__ == "__main__":
    KL_grid(src_side=True, out_format="csv")
