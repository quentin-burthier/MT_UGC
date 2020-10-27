#!/bin/bash

source $TOOLS/parse_cli.sh  # parse_cli, set_dataset_args

# Default CLI args
src=en
tgt=fr
voc_sz=32000

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
# python $TOOLS/compare_lexicons.py $dir/raw/{train,dev}.$src

# Back-translation: tgt -> src model is assumed to have been trained
if $back_translate && [ ! -e "$input_dir/train.bt.$src" ]
then
    if [ ! -e "$mono_dir/preprocessed/train.$tgt" ]
    then
        $HOME/robust_bench/preprocessing/${dataset}_monolingual.sh $tgt $mono_dir
    fi

    if [ ! -e "$bt_dir/train.$src" ]
    then
        mkdir -p $bt_dir
        $marian_decoder \
            -i $mono_dir/preprocessed/train.$tgt \
            -c $bt_model/model.npz.best-translation.npz.decoder.yml \
            -d $gpus -b 1 -n -w 10000 \
            --max-length 100 --max-length-crop \
            -o $bt_dir/train.$src
    fi

    # awk '{print "<syn>" $0 }' $bt_dir/train.$src > $bt_dir/train.$src
    # awk '{print "<nat>" $0 }' $input_dir.$src > $input_dir.$src
    cat $bt_dir/train.$src $input_dir/train.$src{,} \
    > $input_dir/train.bt.$src

    cat $mono_dir/preprocessed/train.$tgt $input_dir/train.$tgt{,} \
    > $input_dir/train.bt.$tgt
fi

# train model
if [ ! -e "$model_dir/model.npz" ]
then
    mkdir -p $model_dir
    mkdir -p $val_output_dir
    mkdir -p log
    $marian_train \
        -c config.yml \
        -m $model_dir/model.npz \
        --train-sets $input_dir/train.$bt{$src,$tgt} \
        --valid-sets $input_dir/val.{$src,$tgt} \
        --valid-translation-output "$val_output_dir/epoch.{E}.$tgt" \
        --valid-script-args $tgt $dir/raw/val.$tgt \
        --vocabs $model_dir/vocab.$src$tgt.spm{,} \
        --dim-vocabs $voc_sz $voc_sz \
        --devices $gpus
fi

# translate dev sets
mkdir -p $output_dir
for split in dev
do
    $marian_decoder \
        -i $input_dir/$split.$src \
        -c $model_dir/model.npz.decoder.yml \
        -d $gpus -b 12 -n -w 6000 \
        --quiet-translation --quiet \
    | sed 's/\@\@ //g' \
    | $MOSES_SCRIPTS/recaser/detruecase.perl \
    | $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt \
    > $output_dir/$split.$tgt
done

# calculate bleu scores on dev set
echo ": Up. 0 :"
cat $output_dir/dev.$tgt | sacrebleu $dir/raw/dev.$tgt
