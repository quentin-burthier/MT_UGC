"""spm_train"""

import argparse
import sentencepiece as spm

def main():
    args = parse_cli()
    if args.model_type == "char":
        spm.SentencePieceTrainer.Train(
            input=args.input, model_prefix=args.model_prefix,
            model_type=args.model_type,
            input_sentence_size=1_000_000, shuffle_input_sentence=True
        )
    else:
        spm.SentencePieceTrainer.Train(
            input=args.input, model_prefix=args.model_prefix,
            model_type=args.model_type, vocab_size=args.vocab_size,
            input_sentence_size=1_000_000, shuffle_input_sentence=True
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
