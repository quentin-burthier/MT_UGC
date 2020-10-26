#!/bin/bash
source $TOOLS/preprocess_fn.sh

src=$1
tgt=$2

# Files nomenclature: news-commentary-v15.$lang1-$lang2.tsv with $lang1 < $lang2
if [ $1 \< $2 ]
then
    src=$1
    tgt=$2
else
    src=$2
    tgt=$1
    ln -s $DATA/News-Commentary.{$src-$tgt,$tgt-$src}
fi
dir=$DATA/News-Commentary.$src-$tgt

if [ ! -e "$dir/raw" ]
then
    mkdir -p $dir

    shuf $DATA/news-commentary/news-commentary-v15.$src-$tgt.tsv -o $dir/corpus.tsv

    split_corpus

    rm $dir/*.tsv
fi

if [ ! -e "$dir/preprocessed" ]
then
    mkdir $dir/preprocessed
    mkdir -p $dir/truecaser

    preprocess_corpus $src $tgt $dir/{raw,preprocessed,truecaser/model}
fi
