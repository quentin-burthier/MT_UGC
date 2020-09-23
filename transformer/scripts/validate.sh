#!/bin/bash

# suffix of source language files
src=en
# suffix of target language files
tgt=fr
# MTNT and processed data paths
mtnt=$DATA/MTNT
dir=$mtnt/$src.$tgt

cat $1 \
    | sed 's/\@\@ //g' \
    | $TOOLS/moses-scripts/scripts/recaser/detruecase.perl 2>/dev/null \
    | $TOOLS/moses-scripts/scripts/tokenizer/detokenizer.perl -l $tgt 2>/dev/null \
    | $TOOLS/moses-scripts/scripts/generic/multi-bleu-detok.perl $DATA/valid.$tgt \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
