#!/bin/bash -v

# suffix of source language files
SRC=en

# suffix of target language files
TRG=fr

# number of merge operations
bpe_operations=32000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$TOOLS/moses-scripts

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$TOOLS/subword-nmt

# Split tsv (adapted from the MTNT script)
DIR=$DATA/MTNT
for split in train valid test
do
  for lang in fr ja
  do
    # en -> $lang
    cut -f2 $DIR/$split/$split.en-$lang.tsv > $DIR/$split/$split.en-$lang.en
    cut -f3 $DIR/$split/$split.en-$lang.tsv > $DIR/$split/$split.en-$lang.$lang
    # $lang -> en
    cut -f2 $DIR/$split/$split.$lang-en.tsv > $DIR/$split/$split.$lang-en.$lang
    cut -f3 $DIR/$split/$split.$lang-en.tsv > $DIR/$split/$split.$lang-en.en
  done
done

# tokenize
for split in train valid test
do
    cat $DATA/$split/$split.$SRC-$TGT.$SRC \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > $DATA/$split/$split.tok.$SRC

    test -f $DATA/$split/$split.$TRG || continue

    cat $DATA/$split/$split.$TRG \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $DATA/$split/$split.tok.$TRG
done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
mv $DATA/train/train.tok.$SRC $DATA/train/train.tok.uncleaned.$SRC
mv $DATA/train/train.tok.$TRG $DATA/train/train.tok.uncleaned.$TRG
$mosesdecoder/scripts/training/clean-corpus-n.perl $DATA/train/train.tok.uncleaned $SRC $TRG $DATA/train/train.tok 1 100

# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $DATA/train/train.tok.$SRC -model model/tc.$SRC
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $DATA/train/train.tok.$TRG -model model/tc.$TRG

# apply truecaser (cleaned training corpus)
for split in train valid test
do
    $mosesdecoder/scripts/recaser/truecase.perl -model model/tc.$SRC < $DATA/$split/$split.tok.$SRC > $DATA/$split/$split.tc.$SRC
    test -f $DATA/$split/$split.tok.$TRG || continue
    $mosesdecoder/scripts/recaser/truecase.perl -model model/tc.$TRG < $DATA/$split/$split.tok.$TRG > $DATA/$split/$split.tc.$TRG
done

# train BPE
cat $DATA/train/train.tc.$SRC $DATA/train/train.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe

# apply BPE
for split in train valid test
do
    $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < $DATA/$split/$split.tc.$SRC > $DATA/$split/$split.bpe.$SRC
    test -f $DATA/$split/$split.tc.$TRG || continue
    $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < $DATA/$split/$split.tc.$TRG > $DATA/$split/$split.bpe.$TRG
done
