#!/bin/bash

source $TOOLS/preprocess_fn.sh

small=$3

if [ $1 \< $2 ]
then
    src=$1
    tgt=$2
else
    src=$2
    tgt=$1
    ln -s $DATA/OpenSubtitles$small.{$src-$tgt,$tgt-$src}
fi
dir=$DATA/OpenSubtitles$small.$src-$tgt

if [ ! -e "$dir/raw" ]
then
    if [ ! -n "$3" ]
    then
        mkdir -p $dir

        paste $DATA/OpenSubtitles.$src-$tgt/OpenSubtitles.$src-$tgt.{$src,$tgt} \
        | shuf -o $dir/corpus.tsv

        split_corpus

        rm $dir/*.tsv
    else
        $TOOLS/subsample.sh $src $tgt $DATA/{News-Commentary,OpenSubtitles{,_small}}.$src-$tgt/raw
    fi
fi

if [ ! -e "$dir/preprocessed" ]
then
    mkdir $dir/preprocessed
    mkdir -p $dir/truecaser

    preprocess_corpus $src $tgt $dir/{raw,preprocessed,truecaser/model}
fi
