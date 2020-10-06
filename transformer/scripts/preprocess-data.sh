#!/bin/bash

# suffix of source language files
src=$1
# suffix of target language files
tgt=$2

# MTNT path
mtnt=$3

ratio=$4  # empty in using all dataset

# Processed data
dir=$mtnt/$src.$tgt$ratio

mkdir $dir

mkdir $dir/splitted
# Split tsv (adapted from the MTNT script)
cut -f2 $mtnt/train/train.$src-$tgt$ratio.tsv > $dir/splitted/train.$src
cut -f3 $mtnt/train/train.$src-$tgt$ratio.tsv > $dir/splitted/train.$tgt
for split in valid test
do
    cut -f2 $mtnt/$split/$split.$src-$tgt.tsv > $dir/splitted/$split.$src
    cut -f3 $mtnt/$split/$split.$src-$tgt.tsv > $dir/splitted/$split.$tgt
done

# tokenize
mkdir $dir/tokenized
for split in train valid test
do
    cat $dir/splitted/$split.$src \
        | $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $src \
        | $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $src > $dir/tokenized/$split.$src

    test -f $dir/splitted/$split.$tgt || continue

    cat $dir/splitted/$split.$tgt \
        | $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $tgt \
        | $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $tgt > $dir/tokenized/$split.$tgt
done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
mkdir $dir/unbalanced
mv $dir/tokenized/train.$src $dir/unbalanced/train.$src
mv $dir/tokenized/train.$tgt $dir/unbalanced/train.$tgt

$MOSES_SCRIPTS/training/clean-corpus-n.perl \
    $dir/unbalanced/train $src $tgt $dir/tokenized/train 1 100

# train truecaser
$MOSES_SCRIPTS/recaser/train-truecaser.perl \
    -corpus $dir/tokenized/train.$src -model model/tc.$src
$MOSES_SCRIPTS/recaser/train-truecaser.perl \
    -corpus $dir/tokenized/train.$tgt -model model/tc.$tgt

# apply truecaser (cleaned training corpus)
mkdir $dir/truecased
for split in train valid test
do
    $MOSES_SCRIPTS/recaser/truecase.perl \
        -model model/tc.$src < $dir/tokenized/$split.$src > $dir/truecased/$split.$src
    test -f $dir/tokenized/$split.$tgt || continue
    $MOSES_SCRIPTS/recaser/truecase.perl \
        -model model/tc.$tgt < $dir/tokenized/$split.$tgt > $dir/truecased/$split.$tgt
done
