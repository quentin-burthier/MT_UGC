#!/bin/bash

# suffix of source language files
src=$1
# suffix of target language files
tgt=$2

# Processed data path
dir=$3

cat $3 \
    | sed 's/\@\@ //g' \
    | $MOSES_SCRIPTS/recaser/detruecase.perl 2>/dev/null \
    | $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt 2>/dev/null \
    | sacrebleu --score-only $dir/splitted/valid.$tgt
