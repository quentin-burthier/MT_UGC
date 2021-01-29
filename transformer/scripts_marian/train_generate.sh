#!/bin/bash

# marian_vocab=$MARIAN/marian-vocab
marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder

function train() {
    # --sentencepiece-options "--model_type $tokenlevel" may need single quotes
    if [ $tokenlevel != "char" ]
    then
        dim_vocabs="--dim-vocabs $nwordssrc $nwordstgt"
    fi

    if [ ! -e "$model_dir/model.npz" ]
    then
        mkdir -p $val_output_dir
        mkdir -p log
        $marian_train \
            -c scripts_marian/config.yml \
            -m $model_dir/model.npz \
            --train-sets $input_dir/train.$bt{$src,$tgt} \
            --valid-sets $input_dir/val.{$src,$tgt} \
            --valid-translation-output "$val_output_dir/epoch.{E}.$tgt" \
            --valid-script-args $tgt $dir/raw/val.$tgt $spm_tgt_model \
            --vocabs $model_dir/vocab.$src-$tgt.spm{,} \
            --sentencepiece-options "--model_type $tokenlevel" $dim_vocabs \
            --devices $gpus
    fi
}

function translate_dev() {
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
}
