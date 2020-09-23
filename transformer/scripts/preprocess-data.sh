#!/bin/bash -v

# suffix of source language files
SRC=en
# suffix of target language files
TRG=fr

# MTNT and processed data paths
MTNT=$DATA/MTNT
DIR=$MTNT/$SRC.$TGT

# number of merge operations
bpe_operations=32000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$TOOLS/moses-scripts

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$TOOLS/subword-nmt

# Split tsv (adapted from the MTNT script)
for split in train valid test
do
    cut -f2 $MTNT/$split/$split.$SRC-$TGT.tsv > $DIR/splitted/$split.$SRC
    cut -f3 $MTNT/$split/$split.$SRC-$TGT.tsv > $DIR/splitted/$split.$TGT
done

# tokenize
for split in train valid test
do
    cat $DIR/splitted/$split.$SRC \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > $DIR/tok/$split.$SRC

    test -f $DIR/splitted/$split.$TRG || continue

    cat $DIR/splitted/$split.$TRG \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $DIR/tok/$split.$TRG
done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
mv $DIR/tok/train.$SRC $DIR/tok.uncleaned/train.$SRC
mv $DIR/tok/train.$TRG $DIR/tok.uncleaned/train.$TRG
$mosesdecoder/scripts/training/clean-corpus-n.perl $DIR/train.tok.uncleaned/train $SRC $TRG $DIR/train/train.tok 1 100

# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $DIR/tok/train.$SRC -model model/tc.$SRC
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $DIR/tok/train.$TRG -model model/tc.$TRG

# apply truecaser (cleaned training corpus)
for split in train valid test
do
    $mosesdecoder/scripts/recaser/truecase.perl -model model/tc.$SRC < $DIR/tok/$split.$SRC > $DIR/truecased/$split.$SRC
    test -f $DIR/tok/$split.$TRG || continue
    $mosesdecoder/scripts/recaser/truecase.perl -model model/tc.$TRG < $DIR/tok/$split.$TRG > $DIR/truecased/$split.$TRG
done

# train BPE
cat $DIR/train/train.tc.$SRC $DIR/train/train.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe

# apply BPE
for split in train valid test
do
    $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < $DIR/truecased/$split.$SRC > $DIR/bpe/$split.$SRC
    test -f $DIR/truecased/$split.$TRG || continue
    $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < $DIR/truecased/$split.$TRG > $DIR/bpe/$split.$TRG
done
