#!/bin/bash

source $TOOLS/parse_cli.sh  # parse_cli, set_dataset_args

# Default CLI args
# framework=framework
# architecture=transformer
# src=en
# tgt=fr
# nwordssrc=32000
# nwordstgt=32000
# tokenlevel=bpe

parse_cli $@

marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder

set_dataset_args

input_dir=$dir/preprocessed
if [ ! -e "$input_dir/dev.$tgt" ]
then
    $HOME/robust_bench/preprocessing/$dataset.sh $preprocess_args
fi

# Back-translation: tgt -> src model is assumed to have been trained
if $back_translate && [ ! -e "$input_dir/train.bt.$src" ]
then
    if [ ! -e "$mono_dir/preprocessed/train.$tgt" ]
    then
        $HOME/robust_bench/preprocessing/${dataset}_monolingual.sh $tgt $mono_dir
    fi

    if [ ! -e "$bt_dir/train.$src" ]
    then
        ./scripts_$framework/backtranslate.sh
    fi

    # awk '{print "<syn>" $0 }' $bt_dir/train.$src > $bt_dir/train.$src
    # awk '{print "<nat>" $0 }' $input_dir.$src > $input_dir.$src
    cat $bt_dir/train.$src $input_dir/train.$src{,} \
    > $input_dir/train.bt.$src

    cat $mono_dir/preprocessed/train.$tgt $input_dir/train.$tgt{,} \
    > $input_dir/train.bt.$tgt
fi

mkdir -p $model_dir

source spm_train_encode.sh
spm_train
spm_encode_train_val

source ./scripts_$framework/train_generate.sh
train
translate_dev

# calculate bleu scores on dev set
echo ": Up. 0 :"
cat $output_dir/dev.$tgt | sacrebleu $dir/raw/dev.$tgt
