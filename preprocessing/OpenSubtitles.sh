#!/bin/bash

source $TOOLS/preprocess_fn.sh

if [ $1 \< $2 ]
then
    src=$1
    tgt=$2
else
    src=$2
    tgt=$1
    ln -s $DATA/OpenSubtitles.{$src-$tgt,$tgt-$src}
fi
dir=$DATA/OpenSubtitles.$src-$tgt

if [ ! -e "$dir/raw" ]
then
    mkdir -p $dir

    paste $DATA/OpenSubtitles.$src-$tgt/OpenSubtitles.$src-$tgt.{$src,$tgt} \
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
