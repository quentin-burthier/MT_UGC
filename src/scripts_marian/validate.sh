#!/bin/bash

# suffix of target language files
tgt=$1

# Processed data path
ref_val=$2

spm_tgt_model=$3

cat $4 \
| python $TOOLS/spm/decode.py --model $spm_tgt_model \
| sed 's/\@\@ //g' \
| $MOSES_SCRIPTS/recaser/detruecase.perl 2>/dev/null \
| $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt 2>/dev/null \
| sacrebleu --score-only $ref_val
