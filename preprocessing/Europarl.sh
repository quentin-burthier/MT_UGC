#!/bin/bash

source $TOOLS/preprocess_fn.sh

small=$3

# Files nomenclature: europarl-v7.$lang-en.{$lang,$en}
if [ $1 == en ]
then
    src=$2
    ln -s $DATA/Europarl$small.{$src-en,en-$src}
else
    src=$1
fi
tgt=en
dir=$DATA/Europarl$small.$src-en

if [ ! -e "$dir/raw" ]
then
    if [ ! -n "$3" ]
    then
        mkdir -p $dir

        paste $DATA/europarl-v7/europarl-v7.$src-$tgt.{$src,$tgt} \
        | shuf -o $dir/corpus.tsv

        split_corpus

        rm $dir/*.tsv
    else
        $TOOLS/subsample.sh $src $tgt $DATA/{News-Commentary,Europarl{,_small}}.$src-$tgt/raw
    fi
fi

if [ ! -e "$dir/preprocessed" ]
then
    mkdir $dir/preprocessed
    mkdir -p $dir/truecaser

    preprocess_corpus $src $tgt $dir/{raw,preprocessed,truecaser/model}
fi
