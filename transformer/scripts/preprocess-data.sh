#!/bin/bash

# suffix of source language files
src=en
# suffix of target language files
tgt=fr

# MTNT and processed data paths
mtnt=$DATA/MTNT
dir=$mtnt/$src.$tgt

mkdir $dir

# number of merge operations
bpe_operations=32000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$TOOLS/moses-scripts

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$TOOLS/subword-nmt

mkdir $dir/splitted
# Split tsv (adapted from the MTNT script)
for split in train valid test
do
    cut -f2 $mtnt/$split/$split.$src-$tgt.tsv > $dir/splitted/$split.$src
    cut -f3 $mtnt/$split/$split.$src-$tgt.tsv > $dir/splitted/$split.$tgt
done

# tokenize
mkdir $dir/tokenized
for split in train valid test
do
    cat $dir/splitted/$split.$src \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $src \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $src > $dir/tokenized/$split.$src

    test -f $dir/splitted/$split.$tgt || continue

    cat $dir/splitted/$split.$tgt \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $tgt \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $tgt > $dir/tokenized/$split.$tgt
done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
mkdir $dir/unbalanced
mv $dir/tokenized/train.$src $dir/unbalanced/train.$src
mv $dir/tokenized/train.$tgt $dir/unbalanced/train.$tgt

$mosesdecoder/scripts/training/clean-corpus-n.perl \
    $dir/unbalanced/train $src $tgt $dir/tokenized/train 1 100

# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl \
    -corpus $dir/tokenized/train.$src -model model/tc.$src
$mosesdecoder/scripts/recaser/train-truecaser.perl \
    -corpus $dir/tokenized/train.$tgt -model model/tc.$tgt

# apply truecaser (cleaned training corpus)
mkdir $dir/truecased
for split in train valid test
do
    $mosesdecoder/scripts/recaser/truecase.perl \
        -model model/tc.$src < $dir/tokenized/$split.$src > $dir/truecased/$split.$src
    test -f $dir/tokenized/$split.$tgt || continue
    $mosesdecoder/scripts/recaser/truecase.perl \
        -model model/tc.$tgt < $dir/tokenized/$split.$tgt > $dir/truecased/$split.$tgt
done

# train BPE
cat $dir/truecased/train.$src $dir/truecased/train.$tgt | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$src$tgt.bpe

# apply BPE
mkdir $dir/bpe
for split in train valid test
do
    $subword_nmt/apply_bpe.py -c model/$src$tgt.bpe < $dir/truecased/$split.$src > $dir/bpe/$split.$src
    test -f $dir/truecased/$split.$tgt || continue
    $subword_nmt/apply_bpe.py -c model/$src$tgt.bpe < $dir/truecased/$split.$tgt > $dir/bpe/$split.$tgt
done
