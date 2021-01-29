import os
import itertools
import argparse


def main_from_cli():
    args = parse_cli()
    model_checkpoints_list(
        train_set=args.train_set, dev_set=args.dev_set,
        src=args.src, tgt=args.tgt,
        arch=args.arch, tokenlevel=args.tokenlevel, voc_sz=args.voc_sz,
        framework=args.framework
    )

def main_from_global():
    train_sets = [
        "News-Commentary",
        "Europarl",
        "Europarl_small",
        "OpenSubtitles",
        "OpenSubtitles_small"
    ]
    dev_sets = [
        "News-Commentary",
        "Europarl",
        "OpenSubtitles",
        "MTNT",
        "Crapbank",
        "Foursquare",
    ]
    tokenlevels_arch = [
        ("bpe", ""),
        ("char", ""),
        ("char", "conv")
    ]
    for train_set, dev_set, (tokenlevel, arch) in itertools.product(train_sets, dev_sets, tokenlevels_arch):
        args = parse_cli()
        model_checkpoints_list(
            train_set=train_set, dev_set=dev_set,
            src=args.src, tgt=args.tgt,
            arch=arch, tokenlevel=args.tokenlevel, voc_sz=args.voc_sz,
            framework=args.framework
        )

def model_checkpoints_list(train_set: str, dev_set: str,
                           src: str, tgt: str,
                           arch: str, tokenlevel: str, voc_sz: str,
                           framework: str):
    model_dir = os.path.join(
        os.environ["DATA"],
        f"models.{framework}",
        f"{arch}Trb.{train_set}"
        f".{tokenlevel if tokenlevel == 'char' else f'tokenlevel.{voc_sz}'}"
        f".{src}-{tgt}"
    )
    for epoch, train_iter in checkpoint_list(model_dir):
        print(" ".join([framework, src, tgt, train_set, dev_set, tokenlevel, epoch, train_iter, voc_sz, arch]).strip())



def checkpoint_list(model_dir: str) -> list:
    checkpoints = [
        file_name[:-3].split("_")[1:]
        for file_name in os.listdir(os.path.join(model_dir, "checkpoints"))
        if len(file_name.split("_")) == 3
    ]
    return checkpoints

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str)
    parser.add_argument("--src", type=str, default="fr")
    parser.add_argument("--tgt", type=str, default="en")
    parser.add_argument("--arch", type=str, default="")
    parser.add_argument("--tokenlevel", type=str, default="char")
    parser.add_argument("--voc_sz", type=str, default="32000")
    parser.add_argument("--framework", type=str, default="fairseq")
    return parser.parse_args()

if __name__ == "__main__":
    main_from_global()
