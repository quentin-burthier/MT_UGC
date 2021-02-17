#!/bin/bash

function train() {
    if [ ! -e "$model_dir/checkpoints/checkpoint_best.pt" ]
    then
        bin_dir=$bpe_dir/bin
        if [ ! -e "$bin_dir/dict.$tgt.txt" ]
        then
            binarise_preprocess       
        fi

        ln -s $bin_dir $model_dir/bin

        mkdir -p $model_dir/tensorboard
        mkdir -p $model_dir/checkpoints

        fairseq-train $bin_dir \
        --arch $architecture \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --lr 0.0003 \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 16000 \
        --patience 10 \
        --eval-bleu \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --save-interval-updates 10000 \
        --clip-norm 5 \
        --max-tokens 4000 \
        --user-dir $HOME/robust_bench/convtransformer \
        --save-dir $model_dir/checkpoints \
        --tensorboard-logdir $model_dir/tensorboard \
        --num-workers $(nproc) \
        --skip-invalid-size-inputs-valid-test
    fi
}


function binarise_preprocess() {
    if $joint_dictionary
    then
        cut -f1 $model_dir/spm.$src-$tgt.vocab \
        | tail -n +4 \
        | sed "s/$/ 100/g" \
        > $model_dir/fairseq.$src-$tgt.vocab

        dictargs="--joined-dictionary --srcdict $model_dir/fairseq.$src.vocab"
    else
        for lang in $src $tgt
        do
            cut -f1 $model_dir/spm.$lang.vocab \
            | tail -n +4 \
            | sed "s/$/ 100/g" \
            > $model_dir/fairseq.$lang.vocab
        done

        dictargs="--srcdict $model_dir/fairseq.$src.vocab --tgtdict $model_dir/fairseq.$tgt.vocab"
    fi

    fairseq-preprocess -s $src -t $tgt \
        --destdir $bin_dir \
        --trainpref $bpe_dir/train$bt \
        --validpref $bpe_dir/val \
        $dictargs \
        --workers $(nproc)
}


function translate_dev() {
    mkdir -p $output_dir

    python $TOOLS/spm/encode.py $input_dir/dev.$src \
        --model=$spm_src_model \
        --output_format=piece \
    | cut -d" " -f 1-1022 \
    | fairseq-interactive $model_dir/bin \
        -s $src -t $tgt \
        --path $model_dir/checkpoints/$checkpoint \
        --buffer-size 2000 \
        --batch-size 32 \
        --beam 6 \
        --user-dir $HOME/robust_bench/convtransformer \
    > $output_dir/dev.$tgt.temp

    grep ^H $output_dir/dev.$tgt.temp | cut -f3 \
    | python $TOOLS/spm/decode.py --model $spm_tgt_model \
    | sed 's/\@\@ //g' \
    | $MOSES_SCRIPTS/recaser/detruecase.perl \
    | $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt \
    > $output_dir/dev.$tgt

    rm $output_dir/dev.$tgt.temp
}
