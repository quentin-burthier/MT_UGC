#!/bin/bash

source $TOOLS/preprocess_fn.sh

src=en
tgt=fr
foursquare=$DATA/Foursquare_Parallel_Corpus
dir=$DATA/Foursquare

if [ ! -e "$dir/raw" ]
then
    mkdir -p $dir/raw
    for lang in $src $tgt
    do
        cp $foursquare/parallel/train-PE.$lang $dir/raw/train.$lang
        cp $foursquare/parallel/valid.$lang $dir/raw/val.$lang
        cp $foursquare/parallel/test.$lang $dir/raw/dev.$lang
    done
fi

if [ ! -e "$dir/preprocessed" ]
then
    mkdir $dir/preprocessed
    mkdir -p $dir/truecaser

    preprocess_corpus $src $tgt $dir/{raw,preprocessed,truecaser/model}
fi
