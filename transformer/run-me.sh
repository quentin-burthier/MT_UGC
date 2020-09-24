#!/bin/bash

# suffix of source language files
src=en
# suffix of target language files
tgt=fr
# MTNT and processed data paths
mtnt=$DATA/MTNT
dir=$mtnt/$src.$tgt

marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder
marian_vocab=$MARIAN/marian-vocab
marian_scorer=$MARIAN/marian-scorer

# set chosen gpus
GPUS=0
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS

if [ ! -e $marian_train ]
then
    echo "marian is not installed in $MARIAN, you need to compile the toolkit first"
    exit 1
fi

if [ ! -e $TOOLS/moses-scripts ] || [ ! -e $TOOLS/subword-nmt ]
then
    echo "missing tools in $TOOLS, you need to download them first"
    exit 1
fi

if [ ! -e "$mtnt" ]
then
    ./scripts/download-files.sh
    echo "Downloaded data\n"
fi

mkdir -p model

# preprocess data
if [ ! -e "$dir" ]
then
    ./scripts/preprocess-data.sh
    echo "Preprocessed data\n"
fi

# create common vocabulary
if [ ! -e "model/vocab.$src$tgt.yml" ]
then
    cat $dir/bpe/train.$src $dir/bpe/train.$tgt | $marian_vocab --max-size 36000 > model/vocab.$src$tgt.yml
    echo "Created vocabulary\n"
fi

# train model
output_dir=$dir/output_$(date +"%d.%m.%Y_%T")
mkdir $output_dir
if [ ! -e "model/model.npz" ]
then
    $marian_train -c config.yml \
        --train-sets $dir/bpe/train.$src $dir/bpe/train.$tgt \
        --vocabs model/vocab.$src$tgt.yml model/vocab.$src$tgt.yml \
        --valid-script-args $src $tgt \
        --valid-sets $dir/bpe/valid.$src $dir/bpe/valid.$tgt \
        --valid-translation-output $output_dir/valid.$src \
        --devices $GPUS
fi

# find best model on dev set
ITER=`cat model/valid.log | grep translation | sort -rg -k12,12 -t' ' | cut -f8 -d' ' | head -n1`

# translate test sets
for split in valid test
do
    cat $dir/$bpe/$split.$src \
        | $marian_decoder -c model/model.npz.decoder.yml -m model/model.iter$ITER.npz -d $GPUS -b 12 -n -w 6000 \
        | sed 's/\@\@ //g' \
        | $TOOLS/moses-scripts/scripts/recaser/detruecase.perl \
        | $TOOLS/moses-scripts/scripts/tokenizer/detokenizer.perl -l $tgt \
        > $output_dir/$split.$tgt
done

# calculate bleu scores on test sets
cat $output_dir/test.$tgt | sacrebleu $dir/splitted/test.$tgt
# LC_ALL=C.UTF-8 sacrebleu -t wmt20/robust/set1 -l $src-$tgt < $output_dir/test.$tgt
