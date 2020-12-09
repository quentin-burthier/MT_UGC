"""spm_encode"""

import sys
import argparse
import sentencepiece as spm

def main():
    args = parse_cli()
    processor = spm.SentencePieceProcessor()
    processor.LoadFromFile(args.model)

    with open(args.input, "r") as f:
        input_lines = f.readlines()

    for line in input_lines:
        pieces = processor.Encode(input=line, out_type=args.output_format)
        sys.stdout.write(" ".join(pieces))
        sys.stdout.write("\n")

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_format", type=str)
    parser.add_argument("input")
    return parser.parse_args()


if __name__ == "__main__":
    main()
