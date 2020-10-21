#!/bin/bash
source $TOOLS/preprocess_fn.sh

tgt=$1
mono_dir=$2

if [ ! -e "$mono_dir/raw" ]
then
    mkdir -p $mono_dir/raw
    for language in french english
    do
        lang=${language:0:2}
        for split in train dev
        do
            python $TOOLS/sentencize.py $language {$DATA/MTNT/monolingual,$mono_dir/raw}/$split.$lang
        done
        cat $mono_dir/raw/train.$lang \
        | shuf -o $mono_dir/raw/train.$lang
    done
fi

mkdir -p $mono_dir/preprocessed
mkdir -p $mono_dir/truecaser

preprocess_monolingual $tgt $mono_dir/{raw,preprocessed,truecaser/model}
