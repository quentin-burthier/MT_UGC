"""Split a file into sentences."""
 
import nltk.data


def sentencize(language: str, file: str, sentencized_file: str):
    """Sentencizes a file (based on nltk)."""
    sentencizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')

    with open(file, "r") as f:
        lines = f.readlines()

    sentences = [sentence
                 for line in lines
                 for sentence in sentencizer.tokenize(line)]

    with open(sentencized_file, "w") as f:
        f.write("\n".join(sentences))


if __name__ == "__main__":
    import sys
    sentencize(sys.argv[1], sys.argv[2], sys.argv[3])
