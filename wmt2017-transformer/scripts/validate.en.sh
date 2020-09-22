#!/bin/bash

cat $1 \
    | sed 's/\@\@ //g' \
    | $TOOLS/moses-scripts/scripts/recaser/detruecase.perl 2>/dev/null \
    | $TOOLS/moses-scripts/scripts/tokenizer/detokenizer.perl -l en 2>/dev/null \
    | $TOOLS/moses-scripts/scripts/generic/multi-bleu-detok.perl data/valid.en \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
