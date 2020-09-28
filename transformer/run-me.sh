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

if [ ! -e $TOOLS/moses-scripts ]
then
    echo "missing Moses tools in $TOOLS, you need to download them first"
    exit 1
fi

if [ ! -e "$mtnt" ]
then
    ./scripts/download-files.sh
    echo "Downloaded data"
    echo ""
fi

mkdir -p model

# preprocess data
if [ ! -e "$dir" ]
then
    ./scripts/preprocess-data.sh
    echo "Preprocessing done"
    echo ""
fi

echo "n_lines: $(wc -l $dir/splitted/train.$src | cut -d" " -f1)"

# train model
# output_dir=$dir/output_$(date +"%d.%m.%Y_%T")
output_dir=$dir/output
mkdir $output_dir
if [ ! -e "model/model.npz" ]
then
    $marian_train -c config.yml \
        --train-sets $dir/truecased/train.$src $dir/truecased/train.$tgt \
        --vocabs model/vocab.$src$tgt.spm model/vocab.$src$tgt.spm \
        --valid-sets $dir/truecased/valid.$src $dir/truecased/valid.$tgt \
        --valid-translation-output $output_dir/valid.$src \
        --valid-script-args $src $tgt \
        --devices $GPUS
fi

# find best model on dev set
ITER=`cat model/valid.log | grep translation | sort -rg -k12,12 -t' ' | cut -f8 -d' ' | head -n1`

# translate test sets
for split in valid test
do
    cat $dir/truecased/$split.$src \
        | $marian_decoder -c model/model.npz.decoder.yml -m model/model.iter$ITER.npz -d $GPUS -b 12 -n -w 6000 \
        | sed 's/\@\@ //g' \
        | $TOOLS/moses-scripts/scripts/recaser/detruecase.perl \
        | $TOOLS/moses-scripts/scripts/tokenizer/detokenizer.perl -l $tgt \
        > $output_dir/$split.$tgt
done

# calculate bleu scores on test sets
cat $output_dir/test.$tgt | sacrebleu $dir/splitted/test.$tgt
# LC_ALL=C.UTF-8 sacrebleu -t wmt20/robust/set1 -l $src-$tgt < $output_dir/test.$tgt
