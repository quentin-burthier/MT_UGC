#!/bin/zsh

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

    for lang in $src $tgt
    do
        mv $input_prefix.$lang .unbalanced.$lang
        split -a 1 --numeric-suffixes=1 -l 25000000 .unbalanced.$lang .chunk.$lang.
    done

    n_chunks=$( ls .chunk.$src* | wc -l )

    for chunk_id in $(seq 1 $n_chunks)
    do
        mv .chunk.$src.$chunk_id .chunk.$chunk_id.$src
        mv .chunk.$tgt.$chunk_id .chunk.$chunk_id.$tgt
        $MOSES_SCRIPTS/training/clean-corpus-n.perl \
            -ratio 1.8 .chunk.$chunk_id $src $tgt $input_prefix.$chunk_id 1 100
    done

    cat $input_prefix.*.$src > $input_prefix.$src
    cat $input_prefix.*.$tgt > $input_prefix.$tgt

    rm .chunk.*
    rm .unbalanced.{$src,$tgt}
    rm $input_prefix.*.{$src,$tgt}
}

function truecaser.fit() {
    local corpus=$1
    local model=$2
    head -n 25000000 $corpus > $corpus.sampled
    $MOSES_SCRIPTS/recaser/train-truecaser.perl \
        -corpus $corpus.sampled \
        -model $model
    rm $corpus.sampled
}

function truecaser.transform() {
    local model=$1
    $MOSES_SCRIPTS/recaser/truecase.perl -model $model
}
