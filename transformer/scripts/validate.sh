#!/bin/bash

# suffix of source language files
src=$1
# suffix of target language files
tgt=$2
# MTNT and processed data paths
mtnt=$DATA/MTNT
dir=$mtnt/$src.$tgt

cat $3 \
    | sed 's/\@\@ //g' \
    | $TOOLS/moses-scripts/scripts/recaser/detruecase.perl 2>/dev/null \
    | $TOOLS/moses-scripts/scripts/tokenizer/detokenizer.perl -l $tgt 2>/dev/null \
    | sacrebleu $dir/splitted/valid.$tgt \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
