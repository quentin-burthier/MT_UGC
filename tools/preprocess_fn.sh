#!/bin/zsh

function preprocess_corpus() {
    local src=$1
    local tgt=$2
    local input_dir=$3
    local output_dir=$4
    local model_prefix=$5

    filtered_dir=$output_dir/lang_filtered
    mkdir $filtered_dir
    lang_filter $src $tgt $input_dir $filtered_dir

    tokenized_dir=$output_dir/tokenized
    mkdir $tokenized_dir
    for lang in $src $tgt; do
        for split in val dev train; do
            cat $filtered_dir/$split.$lang \
            | normalize_tokenize $lang \
            > $tokenized_dir/$split.$lang
        done
    done

    clean_corpus $src $tgt $tokenized_dir/train

    for lang in $src $tgt; do
        truecaser.fit $tokenized_dir/train.$lang $model_prefix.$lang
        for split in val dev train; do
            cat $tokenized_dir/$split.$lang \
            | truecaser.transform $model_prefix.$lang \
            > $output_dir/$split.$lang
        done
    done

    # rm -r $tokenized_dir
    # rm -r $filtered_dir
}

function preprocess_monolingual() {
    local lang=$1
    local input_dir=$2
    local output_dir=$3
    local model_prefix=$4

    mkdir $output_dir/tokenized
    for split in train dev
    do
        cat $input_dir/$split.$lang \
        | normalize_tokenize $lang \
        > $output_dir/tokenized/$split.$lang
    done

    awk -F" " 'NF < 100' $output_dir/tokenized/train.$lang \
    > $output_dir/temp && mv $output_dir/temp $output_dir/tokenized/train.$lang

    truecaser.fit $output_dir/tokenized/train.$lang $model_prefix.$lang
    for split in dev train
    do
        cat $output_dir/tokenized/$split.$lang \
        | truecaser.transform $model_prefix.$lang \
        > $output_dir/$split.$lang
    done
}

function split_corpus() {
    head -4096 $dir/corpus.tsv > $dir/val.tsv
    sed -n '4097,8192p;8193q' $dir/corpus.tsv > $dir/dev.tsv
    sed -n '8193,$p' $dir/corpus.tsv > $dir/train.tsv

    mkdir $dir/raw
    for split in train val dev
    do
        cut -f1 $dir/$split.tsv > $dir/raw/$split.$src
        cut -f2 $dir/$split.tsv > $dir/raw/$split.$tgt
    done
}

function lang_filter() {
    local src=$1
    local tgt=$2
    local input_dir=$3
    local filtered_dir=$4

    for lang in $src $tgt; do
        $FASTTEXT/fasttext predict $FASTTEXT/models/lid.176.bin $input_dir/train.$lang > $filtered_dir/labels.train.$lang
        for split in val dev; do
            ln -s $input_dir/$split.$lang $filtered_dir/$split.$lang
        done
    done
    python $TOOLS/lid_filter.py {{$input_dir/,$filtered_dir/{labels.,}}train.,}{$src,$tgt}
}

function normalize_tokenize() {
    local lang=$1
    $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $lang \
    | $MOSES_SCRIPTS/tokenizer/tokenizer.perl -no-escape -a -l $lang
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
