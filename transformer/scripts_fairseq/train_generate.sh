#!/bin/bash

function train() {
    if [ ! -e "$model_dir/checkpoint_best.pt" ]
    then
        bin_dir=$bpe_dir/bin
        if [ ! -e "$bin_dir" ]
        then
            fairseq-preprocess -s $src -t $tgt \
            --destdir $bin_dir \
            --trainpref $bpe_dir/train$bt \
            --validpref $bpe_dir/val \
            --bpe sentencepiece \
            --joined-dictionary \
            --workers $(nproc)

            ln -s $bin_dir $model_dir/bin
        fi

        fairseq-train $bin_dir \
        --arch $architecture \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --lr 0.0003 \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 16000 \
        --patience 10 \
        --clip-norm 5 \
        --max-tokens 4000 \
        --user-dir $HOME/robust_bench/convtransformer \
        --save-dir $model_dir \
        --no-epoch-checkpoints \
        --no-last-checkpoints \
        --num-workers $(nproc) \
        --skip-invalid-size-inputs-valid-test
    fi
}

function translate_dev() {
    mkdir -p $output_dir

    python $TOOLS/spm/encode.py $input_dir/dev.$src \
        --model=$spm_src_model \
        --output_format=piece \
    | cut -d" " -f 1-1022 \
    | fairseq-interactive $model_dir/bin \
        -s $src -t $tgt \
        --path $model_dir/checkpoint_best.pt \
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
