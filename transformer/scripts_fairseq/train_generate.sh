#!/bin/bash

CUDA_VISIBLE_DEVICES=$1
src=$2
tgt=$3
nwordssrc=$4
nwordstgt=$5
input_dir=$6
bpe_dir=$7
output_dir=$8
model_dir=$9
architecture=${10}
bt=${11}

bin_dir=$bpe_dir/bin
if [ ! -e "$bin_dir" ]
then
    fairseq-preprocess -s $src -t $tgt \
    --destdir $bin_dir \
    --trainpref $bpe_dir/train$bt \
    --validpref $bpe_dir/val \
    --bpe sentencepiece \
    --joined-dictionary \
    --dataset-impl raw \
    --workers $(nproc)
fi

if [ ! -e "$model_dir/checkpoint_best.pt" ]
then
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
    --max-tokens 6000 \
    --dataset-impl raw \
    --user-dir $HOME/robust_bench/convtransformer \
    --save-dir $model_dir \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --num-workers $(nproc) \
    --skip-invalid-size-inputs-valid-test
fi

python $TOOLS/spm/encode.py $input_dir/dev.$src \
    --model=$model_dir/spm.$src.model \
    --output_format=piece \
| fairseq-interactive $bin_dir \
    -s $src -t $tgt \
    --path $model_dir/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --batch-size 128 \
    --beam 6 \
    --dataset-impl raw \
> $output_dir/dev.$tgt.temp

grep ^H $output_dir/dev.$tgt.temp | cut -f3 \
| python $TOOLS/spm/decode.py --model $model_dir/spm.$tgt.model \
| sed 's/\@\@ //g' \
| $MOSES_SCRIPTS/recaser/detruecase.perl \
| $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt \
> $output_dir/dev.$tgt

rm $output_dir/dev.$tgt.temp
