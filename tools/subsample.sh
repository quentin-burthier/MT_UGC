#!/bin/bash
# ./subsample.sh en fr $DATA/{News-Commentary,OpenSubtitles{,_small}}.fr-en/raw

src=$1
tgt=$2
ref_dir=$3
sampled_dir=$4
output_dir=$5

mkdir -p $output_dir

n_sampled_lines=$(wc -l $ref_dir/train.$src | cut -d" " -f1)
for lang in $src $tgt
do
    head -n $n_sampled_lines $sampled_dir/train.$lang \
    > $output_dir/train.$lang

    for split in val dev
    do
        cp {$sampled_dir,$output_dir}/$split.$lang
    done
done
