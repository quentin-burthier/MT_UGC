#!/bin/bash -v

# suffix of target language files
SRC=en
TRG=de

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$TOOLS/moses-scripts

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$TOOLS/subword-nmt

# tokenize

prefix=news.2016

cat $DATA/$prefix/$prefix.$TRG \
    | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
    | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $DATA/$prefix/$prefix.tok.$TRG

$mosesdecoder/scripts/recaser/truecase.perl -model model/tc.$TRG < $DATA/$prefix/$prefix.tok.$TRG > $DATA/$prefix/$prefix.tc.$TRG

$subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < $DATA/$prefix/$prefix.tc.$TRG > $DATA/$prefix/$prefix.bpe.$TRG
