"""spm_decode"""

import sys
import argparse
import sentencepiece as spm

def main():
    args = parse_cli()
    processor = spm.SentencePieceProcessor()
    processor.LoadFromFile(args.model)

    for line in sys.stdin:
        sys.stdout.write(processor.Decode(line))

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main()
