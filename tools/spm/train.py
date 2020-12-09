"""spm_train"""

import argparse
import sentencepiece as spm

def main():
    args = parse_cli()
    spm.SentencePieceTrainer.Train(
        input=args.input, model_prefix=args.model_prefix,
        vocab_size=args.vocab_size, model_type=args.model_type
    )

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--vocab_size", type=str)
    parser.add_argument("--character_coverage", type=float)
    parser.add_argument("--model_type", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main()
