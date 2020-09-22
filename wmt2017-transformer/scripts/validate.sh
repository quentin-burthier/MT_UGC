#!/bin/bash

export LC_ALL=C.UTF-8

cat $1 \
    | sed 's/\@\@ //g' \
    | $TOOLS/moses-scripts/scripts/recaser/detruecase.perl 2>/dev/null \
    | $TOOLS/moses-scripts/scripts/tokenizer/detokenizer.perl -l de 2>/dev/null \
    | $TOOLS/moses-scripts/scripts/generic/multi-bleu-detok.perl data/valid.de \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
