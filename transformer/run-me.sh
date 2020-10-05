#!/bin/bash

# Ratio of training data
ratio=.0.9

# suffix of source language files
src=en
# suffix of target language files
tgt=fr
# MTNT and processed data paths
mtnt=$DATA/MTNT
dir=$mtnt/$src.$tgt$ratio

# set chosen gpus
GPUS=0
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS

marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder

if [ ! -e $TOOLS/moses-scripts ]
then
    echo "missing Moses tools in $TOOLS, you need to download them first"
    exit 1
fi
moses_scripts=$TOOLS/moses-scripts/scripts

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
    ./scripts/preprocess-data.sh $src $tgt $ratio
    echo "Preprocessing done"
    echo ""
fi

echo "Starting epoch 0"
n_lines='$(wc -l $dir/splitted/train.$src | cut -d" " -f1)'
echo "n_lines: $(wc -l $dir/splitted/train.$src | cut -d" " -f1)"
echo "ratio: $ratio"
python tools/compare_lexicon.py dir=$mtnt/$src.$tgt$ratio/splitted/{train,test}.$src

# train model
input_dir=$dir/truecased
# output_dir=$dir/output_$(date +"%d.%m.%Y_%T")
output_dir=$dir/output
mkdir $output_dir
mkdir "valid_output"
if [ ! -e "model/model.npz" ]
then
    $marian_train -c config.yml \
        --train-sets $input_dir/train.$src $input_dir/train.$tgt \
        --vocabs model/vocab.$src$tgt.spm model/vocab.$src$tgt.spm \
        --valid-sets $input_dir/valid.$src $input_dir/valid.$tgt \
        --valid-translation-output "valid_output/epoch.{E}.$tgt" \
        --valid-script-args $src $tgt \
        --devices $GPUS
fi

# translate test sets
for split in valid test
do
    cat $input_dir/$split.$src \
        | $marian_decoder \
            -c model/model.npz.decoder.yml \
            --quiet-translation \
            -m model/model.npz.best-translation.npz \
            -d $GPUS -b 12 -n -w 6000 \
        | sed 's/\@\@ //g' \
        | $moses_scripts/recaser/detruecase.perl \
        | $moses_scripts/tokenizer/detokenizer.perl -l $tgt \
        > $output_dir/$split.$tgt
done

# calculate bleu scores on test sets
echo "Starting epoch 0"
cat $output_dir/test.$tgt | sacrebleu $dir/splitted/test.$tgt
# LC_ALL=C.UTF-8 sacrebleu -t wmt20/robust/set1 -l $src-$tgt < $output_dir/test.$tgt
