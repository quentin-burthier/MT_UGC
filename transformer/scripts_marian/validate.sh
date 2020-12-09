#!/bin/bash

# suffix of target language files
tgt=$1

# Processed data path
ref_val=$2

cat $3 \
    | python $TOOLS/spm/decode.py \
    | sed 's/\@\@ //g' \
    | $MOSES_SCRIPTS/recaser/detruecase.perl 2>/dev/null \
    | $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt 2>/dev/null \
    | sacrebleu --score-only $ref_val
