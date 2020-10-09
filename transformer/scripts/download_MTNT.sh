#!/bin/bash

# MTNT data
wget -nc -O $DATA/MTNT.1.1.tar.gz \
    https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz

# extract data
tar -C $DATA -xf $DATA/MTNT.1.1.tar.gz

# clean
rm -r $DATA/MTNT.1.1.tar.gz

# reshuffle
mkdir $DATA/MTNT_reshuffled
cp -r $DATA/MTNT/test $DATA/MTNT_reshuffled/
for lang_pair in "en-ja" "ja-en"
do
    cat $DATA/MTNT/train/train.$lang_pair.tsv \
        $DATA/MTNT/valid/valid.$lang_pair.tsv \
    | shuf \
    | split -a1 -d -l $( wc -l <$DATA/MTNT/train/train.$lang_pair.tsv ) - output
    mv output0 $DATA/MTNT_reshuffled/train/train.$lang_pair.tsv
    mv output1 $DATA/MTNT_reshuffled/valid/valid.$lang_pair.tsv
done
