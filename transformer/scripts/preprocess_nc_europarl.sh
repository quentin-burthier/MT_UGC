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

    head -4096 $dir/corpus.tsv > $dir/val.tsv
    sed -n '4097,8192p;8193q' $dir/corpus.tsv > $dir/dev.tsv
    sed -n '8193,$p' $dir/corpus.tsv > $dir/train.tsv

    mkdir $dir/raw
    for split in train val dev
    do
        cut -f1 $dir/$split.tsv > $dir/raw/$split.$src
        cut -f2 $dir/$split.tsv > $dir/raw/$split.$tgt
    done

    rm $dir/*.tsv
fi

mkdir $dir/preprocessed
mkdir model

preprocess_corpus $src $tgt $dir/raw $dir/preprocessed truecaser/tc
