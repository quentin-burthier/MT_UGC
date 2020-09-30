#!/bin/bash

# suffix of source language files
src=$1
# suffix of target language files
tgt=$2
# MTNT and processed data paths
mtnt=$DATA/MTNT
dir=$mtnt/$src.$tgt

moses_scripts=$TOOLS/moses-scripts/scripts

cat $3 \
    | sed 's/\@\@ //g' \
    | $moses_scripts/recaser/detruecase.perl 2>/dev/null \
    | $moses_scripts/tokenizer/detokenizer.perl -l $tgt 2>/dev/null \
    | sacrebleu --score-only $dir/splitted/valid.$tgt
