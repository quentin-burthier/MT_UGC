#!/bin/bash

marian_vocab=$MARIAN/marian-vocab
marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder

gpus=$1
src=$2
tgt=$3
# nwordssrc=$4
# nwordstgt=$5
input_dir=$6
bpe_dir=$7
output_dir=$8
model_dir=$9
# architecture=${10}
bt=${11}

if [ ! -e "$model_dir/vocabs.yml" ]
then
    cat $bpe_dir/train.$bt{$src,$tgt}* \
    | $marian_vocab \
    > $model_dir/vocabs.yml
fi

# train model
if [ ! -e "$model_dir/model.npz" ]
then
    mkdir -p $val_output_dir
    mkdir -p log
    $marian_train \
        -c scripts_marian/config.yml \
        -m $model_dir/model.npz \
        --train-sets $bpe_dir/train.$bt{$src,$tgt} \
        --valid-sets $bpe_dir/val.{$src,$tgt} \
        --valid-translation-output "$val_output_dir/epoch.{E}.$tgt" \
        --valid-script-args $tgt $dir/raw/val.$tgt \
        --vocabs $model_dir/vocab{,}.yml \
        --devices $gpus
fi

# translate dev set
python $TOOLS/spm/encode.py $input_dir/dev.$src \
    --model=$model_dir/spm.$src.model \
    --output_format=piece \
| $marian_decoder \
    -c $model_dir/model.npz.decoder.yml \
    -d $gpus -b 12 -n -w 6000 \
    --quiet-translation --quiet \
| python $TOOLS/spm/decode.py --model $model_dir/spm.$tgt.model \
| sed 's/\@\@ //g' \
| $MOSES_SCRIPTS/recaser/detruecase.perl \
| $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt \
> $output_dir/$split.$tgt
