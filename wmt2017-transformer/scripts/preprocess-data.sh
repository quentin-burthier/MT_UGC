#!/bin/bash -v

# suffix of source language files
SRC=en

# suffix of target language files
TRG=de

# number of merge operations
bpe_operations=32000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$TOOLS/moses-scripts

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$TOOLS/subword-nmt

# tokenize
for prefix in corpus valid test2014 test2015 test2016 test2017
do
    cat $DATA/$prefix/$prefix.$SRC \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > $DATA/$prefix/$prefix.tok.$SRC

    test -f $DATA/$prefix/$prefix.$TRG || continue

    cat $DATA/$prefix/$prefix.$TRG \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $DATA/$prefix/$prefix.tok.$TRG
done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
mv $DATA/train/train.tok.$SRC $DATA/train/train.tok.uncleaned.$SRC
mv $DATA/train/train.tok.$TRG $DATA/train/train.tok.uncleaned.$TRG
$mosesdecoder/scripts/training/clean-corpus-n.perl $DATA/train/train.tok.uncleaned $SRC $TRG $DATA/train/train.tok 1 100

# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $DATA/train/train.tok.$SRC -model model/tc.$SRC
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $DATA/train/train.tok.$TRG -model model/tc.$TRG

# apply truecaser (cleaned training corpus)
for prefix in corpus valid test2014 test2015 test2016 test2017
do
    $mosesdecoder/scripts/recaser/truecase.perl -model model/tc.$SRC < $DATA/$prefix/$prefix.tok.$SRC > $DATA/$prefix/$prefix.tc.$SRC
    test -f $DATA/$prefix/$prefix.tok.$TRG || continue
    $mosesdecoder/scripts/recaser/truecase.perl -model model/tc.$TRG < $DATA/$prefix/$prefix.tok.$TRG > $DATA/$prefix/$prefix.tc.$TRG
done

# train BPE
cat $DATA/train/train.tc.$SRC $DATA/train/train.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe

# apply BPE
for prefix in corpus valid test2014 test2015 test2016 test2017
do
    $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < $DATA/$prefix/$prefix.tc.$SRC > $DATA/$prefix/$prefix.bpe.$SRC
    test -f $DATA/$prefix/$prefix.tc.$TRG || continue
    $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < $DATA/$prefix/$prefix.tc.$TRG > $DATA/$prefix/$prefix.bpe.$TRG
done
