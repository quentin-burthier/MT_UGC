#!/bin/bash
source $TOOLS/preprocess_fn.sh

src=$1
tgt=$2
dir=$3
mtnt=$4
ratio=$5

if [ ! -e "$DATA/MTNT" ]
then
    # MTNT data
    wget -nc -O $DATA/MTNT.1.1.tar.gz \
        https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz

    # extract data
    tar -C $DATA -xf $DATA/MTNT.1.1.tar.gz

    # clean
    rm -r $DATA/MTNT.1.1.tar.gz
fi

if [ ! -e "$mtnt" ] # reshuffle case
then
    mkdir $mtnt
    mkdir $mtnt/{train,valid}
    cp -r $DATA/MTNT/test $mtnt
    for lang in "fr" "ja"
    do
        for lang_pair in "en-$lang" "$lang-en"
        do
            cat $DATA/MTNT/{train/train,valid/valid}.$lang_pair.tsv \
                | shuf -o $mtnt/corpus.$lang_pair.tsv
            head -n $( wc -l <$DATA/MTNT/train/train.$lang_pair.tsv ) $mtnt/corpus.$lang_pair.tsv \
                > $mtnt/train/train.$lang_pair.tsv
            tail -n $( wc -l <$DATA/MTNT/valid/valid.$lang_pair.tsv ) $mtnt/corpus.$lang_pair.tsv \
                > $mtnt/valid/valid.$lang_pair.tsv
            rm $mtnt/corpus.$lang_pair.tsv
        done
    done
fi

if [ ! -e "$dir" ]
then
    mkdir $dir
    awk -v src=$src -v tgt=$tgt -v r=$ratio -v dir=$dir \
        'BEGIN  {srand()}
        !/^$/  { if (rand() <= r || FNR==1) print > dir"/train.tsv"}' $mtnt/train/train.$src-$tgt.tsv
fi

if [ ! -e "$dir/raw" ]
then
    mkdir -p $dir/raw
    # train
    cut -f2 $dir/train.tsv > $dir/raw/train.$src
    cut -f3 $dir/train.tsv > $dir/raw/train.$tgt
    rm $dir/train.tsv
    # validation set
    cut -f2 $mtnt/valid/valid.$src-$tgt.tsv > $dir/raw/val.$src
    cut -f3 $mtnt/valid/valid.$src-$tgt.tsv > $dir/raw/val.$tgt
    # dev set
    cut -f2 $mtnt/test/test.$src-$tgt.tsv > $dir/raw/dev.$src
    cut -f3 $mtnt/test/test.$src-$tgt.tsv > $dir/raw/dev.$tgt
fi

mkdir -p $dir/preprocessed
mkdir -p truecaser_mtnt

preprocess_corpus $src $tgt $dir/raw $dir/preprocessed truecaser_mtnt/tc
