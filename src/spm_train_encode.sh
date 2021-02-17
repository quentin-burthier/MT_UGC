#!/bin/bash

function spm_train() {
    if $joint_dictionary
    then
        if [ ! -e "$model_dir/spm.$src-$tgt.model" ]
        then
            cat $input_dir/train.{$src,$tgt} \
            | shuf \
            > $input_dir/train.$src-$tgt

            python $TOOLS/spm/train.py \
                --input $input_dir/train.$src-$tgt \
                --model_prefix $model_dir/spm.$src-$tgt \
                --vocab_size $nwords \
                --character_coverage 1.0 \
                --model_type $segmentation

            # rm $input_dir/train.$src-$tgt
        fi
        spm_src_model=$model_dir/spm.$src-$tgt.model
        spm_tgt_model=$model_dir/spm.$src-$tgt.model
    else
        if [ ! -e "$model_dir/spm.$src.model" ]
        then
            python $TOOLS/spm/train.py \
                --input $input_dir/train.$src \
                --model_prefix $model_dir/spm.$src \
                --vocab_size $nwordssrc \
                --character_coverage 1.0 \
                --model_type $src_segmentation
        fi
        if [ ! -e "$model_dir/spm.$tgt.model" ]
        then
            python $TOOLS/spm/train.py \
                --input $input_dir/train.$tgt \
                --model_prefix=$model_dir/spm.$tgt \
                --vocab_size=$nwordstgt \
                --character_coverage=1.0 \
                --model_type=$tgt_segmentation
        fi
        spm_src_model=$model_dir/spm.$src.model
        spm_tgt_model=$model_dir/spm.$tgt.model
    fi
}

function spm_encode_train_val() {
    if [ ! -e "$bpe_dir/train.$tgt" ]
    then
        mkdir -p $bpe_dir
        for split in train val
        do
            python $TOOLS/spm/encode.py $input_dir/$split.$src \
                --model=$spm_src_model \
                --output_format=piece \
            > $bpe_dir/$split.$src

            python $TOOLS/spm/encode.py $input_dir/$split.$tgt \
                --model=$spm_tgt_model \
                --output_format=piece \
            > $bpe_dir/$split.$tgt 
        done
    fi
}
