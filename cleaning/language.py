"""Script to filter sentences in the wrong languages."""

from langid.langid import model, LanguageIdentifier


def main(data_path: str, filtered_path: str,
         src_language: str, tgt_language: str):
    identifier = LanguageIdentifier.from_modelstring(model)

    with open(data_path, "r") as f:
        filtered_lines = [line for line in f.readlines()[:100]
                          if correct_language_pair(identifier, line,
                                                   src_language, tgt_language)]

    with open(filtered_path, "w") as f:
        print(len(filtered_lines))
        f.writelines(filtered_lines)


def correct_language_pair(identifier: LanguageIdentifier,
                          line: str, src_language: str, tgt_language: str):
    _, pair_src, pair_tgt = line.split("\t")
    pair_src_lang, _ = identifier.classify(pair_src)
    pair_tgt_lang, _ = identifier.classify(pair_tgt)
    return pair_src_lang == src_language and pair_tgt_lang == tgt_language


if __name__ == "__main__":
    SRC = "en"
    TGT = "fr" 
    DATA_PATH = f"data/MTNT/train/train.{SRC}-{TGT}.tsv"
    FILTERED_PATH = f"data_cleaned/{SRC}-{TGT}.tsv"
    main(DATA_PATH, FILTERED_PATH, SRC, TGT)
