#!/bin/bash

source $TOOLS/parse_cli.sh  # parse_cli, set_dataset_args

# Default CLI args
framework=marian
architecture=transformer
src=en
tgt=fr
nwordssrc=32000
nwordstgt=32000
tokenlevel=sentencepiece

model_dir=model
back_translate=false

parse_cli $@

marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder

set_dataset_args

input_dir=$dir/preprocessed
if [ ! -e "$input_dir" ]
then
    $HOME/robust_bench/preprocessing/$dataset.sh $preprocess_args
fi

echo ": Up. 0 :"
echo "n_lines: $(wc -l $dir/raw/train.$src | cut -d" " -f1)"

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

bpe_dir=$dir/bpe.$nwordssrc.$nwordstgt
if [ ! -e "$bpe_dir/train.$src" ]
then
    if [ ! -e "$model_dir/spm.$src.model" ]
    then
        python $TOOLS/spm/train.py --input=$input_dir/train.$src \
            --model_prefix $model_dir/spm.$src \
            --vocab_size $nwordssrc \
            --character_coverage 1.0 \
            --model_type $tokenlevel

        python $TOOLS/spm/train.py --input=$input_dir/train.$tgt \
            --model_prefix=$model_dir/spm.$tgt \
            --vocab_size=$nwordstgt \
            --character_coverage=1.0 \
            --model_type=$tokenlevel
    fi

    mkdir -p $bpe_dir
    for split in train val
    do
        for lang in $src $tgt
        do
            python $TOOLS/spm/encode.py $input_dir/$split.$lang \
                --model=$model_dir/spm.$lang.model --output_format=piece \
            > $bpe_dir/$split.$lang 
        done
    done
fi

mkdir -p $output_dir

./scripts_$framework/train_generate.sh $gpus $src $tgt $nwordssrc $nwordstgt \
    $input_dir $bpe_dir $output_dir $model_dir $architecture $bt

# calculate bleu scores on dev set
echo ": Up. 0 :"
cat $output_dir/dev.$tgt | sacrebleu $dir/raw/dev.$tgt
