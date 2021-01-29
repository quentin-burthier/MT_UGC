import os
import json

def to_marian(spm_dict: str, marian_json: str):
    with open(spm_dict, "r") as spm_file:
        lines = spm_file.readlines()
    
    vocab_tokens = [line.split("\t")[0] for line in lines[3:]]

    marian_vocab = {
        f"{token}": index
        for index, token in enumerate(vocab_tokens, start=2)
    }
    marian_vocab["</s>"] = 0
    marian_vocab["<unk>"] = 1
    print(marian_vocab)

    with open(marian_json, "w") as f:
        json.dump(marian_vocab, fp=f)


if __name__ == "__main__":
    import sys
    to_marian(sys.argv[1], sys.argv[2])
