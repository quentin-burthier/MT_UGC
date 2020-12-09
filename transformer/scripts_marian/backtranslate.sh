#!/bin/bash

marian_decoder=$MARIAN/marian-decoder

mkdir -p $bt_dir
$marian_decoder \
    -i $mono_dir/preprocessed/train.$tgt \
    -c $bt_model/model.npz.best-translation.npz.decoder.yml \
    -d $gpus -b 1 -n -w 10000 \
    --max-length 100 --max-length-crop \
    -o $bt_dir/train.$src
