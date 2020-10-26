#!/bin/bash

source $TOOLS/preprocess_fn.sh

src=$1
tgt=$2
dir=$3

if [ ! -e "$dir/raw" ]
then
    mkdir -p $dir

    paste $DATA/europarl-v7/europarl-v7.$tgt-$src.{$src,$tgt} \
        > $DATA/europarl-v7/europarl-v7.$src-$tgt.tsv

    cat $DATA/{europarl-v7/europarl-v7,news-commentary/news-commentary-v15}.$src-$tgt.tsv \
        | shuf -o $dir/corpus.tsv

    split_corpus

    rm $dir/*.tsv
fi

if [ ! -e "$dir/preprocessed" ]
then
    mkdir $dir/preprocessed
    mkdir -p $dir/truecaser

    preprocess_corpus $src $tgt $dir/{raw,preprocessed,truecaser/model}
fi
