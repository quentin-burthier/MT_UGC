#!/bin/bash

function preprocess_corpus() {
    local src=$1
    local tgt=$2
    local input_dir=$3
    local output_dir=$4
    local model_prefix=$5

    mkdir $output_dir/tokenized
    for lang in $src $tgt; do
        for split in val dev train; do
            cat $input_dir/$split.$lang \
            | normalize_tokenize $lang \
            > $output_dir/tokenized/$split.$lang
        done
    done

    clean_corpus $src $tgt $output_dir/tokenized/train

    for lang in $src $tgt; do
        truecaser.fit $output_dir/tokenized/train.$lang $model_prefix.$lang
        for split in val dev train; do
            cat $output_dir/tokenized/$split.$lang \
            | truecaser.transform $model_prefix.$lang \
            > $output_dir/$split.$lang
        done
    done

    # rm -r $output_dir/tokenized
}

function normalize_tokenize() {
    local lang=$1
    $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $lang \
    | $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $lang
}

function clean_corpus() {
    # clean empty and long sentences,
    # and sentences with high source-target ratio (training corpus only)
    local src=$1
    local tgt=$2
    local input_prefix=$3

    mv $input_prefix.$src .unbalanced.$src
    mv $input_prefix.$tgt .unbalanced.$tgt

    $MOSES_SCRIPTS/training/clean-corpus-n.perl \
        -ratio 2 .unbalanced $src $tgt $input_prefix 1 100

    rm .unbalanced.{$src,$tgt}
}

function truecaser.fit() {
    local corpus=$1
    local model=$2
    $MOSES_SCRIPTS/recaser/train-truecaser.perl \
        -corpus $corpus \
        -model $model
}

function truecaser.transform() {
    local model=$1
    
    $MOSES_SCRIPTS/recaser/truecase.perl -model $model
}
