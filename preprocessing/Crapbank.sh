#!/bin/bash

source $TOOLS/preprocess_fn.sh

truecaser=$1

src=fr
tgt=en
nodalida=$DATA/NODALIDA
dir=$DATA/Crapbank

if [ ! -e "$dir/raw" ]
then
    mkdir -p $dir/raw
    for lang in $src $tgt
    do
        cp $nodalida/test/Crapbank.test.$lang $dir/raw/dev.$lang
        cp $nodalida/blind_tests/Crapbank_bind.test.$lang $dir/raw/test.$lang
    done
fi

if [ ! -e "$dir/preprocessed" ]
then
    mkdir $dir/preprocessed
    for lang in $src $tgt
    do
        for split in dev test
        do 
            cat $dir/raw/$split.$lang \
            | normalize_tokenize $lang \
            | truecaser.transform $truecaser.$lang \
            > $dir/preprocessed/$split.$lang
        done
    done
fi
