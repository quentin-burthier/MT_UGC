#!/bin/bash
# ./subsample.sh en fr $DATA/{MTNT_reshuffled/en-fr.1.0,europarl_nc.en-fr,nce.en-fr_small}/raw

src=$1
tgt=$2
ref_dir=$3
sampled_dir=$4
output_dir=$5

mkdir -p $output_dir

for split in train val dev
do  
    n_sampled_lines=$(wc -l $ref_dir/$split.$src | cut -d" " -f1)
    for lang in $src $tgt
    do
        head -n $n_sampled_lines $sampled_dir/$split.$lang \
        > $output_dir/$split.$lang
    done
done
