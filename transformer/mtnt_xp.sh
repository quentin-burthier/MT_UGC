#!/bin/bash

source $TOOLS/parse_cli.sh

# Default CLI args
src=en
tgt=fr
ratio=""
voc_sz=32000

parse_cli $@

# MTNT and processed data paths
mtnt=$DATA/MTNT_reshuffled
dir=$mtnt/$src.$tgt$ratio

marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder

if [ ! -e "$mtnt" ]
then
    ./scripts/download-files.sh
    echo "Downloaded data"
    echo ""
fi

mkdir -p model

# preprocess data
if [ ! -e "$dir" ]
then
    ./scripts/preprocess-data.sh $src $tgt $mtnt $ratio
    echo "Preprocessing done"
    echo ""
fi

echo "Starting epoch 0"
n_lines='$(wc -l $dir/splitted/train.$src | cut -d" " -f1)'
echo "n_lines: $(wc -l $dir/splitted/train.$src | cut -d" " -f1)"

python $TOOLS/compare_lexicon.py $dir/splitted/{train,test}.$src

# train model
input_dir=$dir/truecased
# output_dir=$dir/output
output_dir=$dir/output_$voc_sz
# output_dir=$dir/output_$(date +"%d.%m.%Y_%T")
mkdir $output_dir
mkdir "valid_output"
if [ ! -e "model/model.npz" ]
then
    $marian_train -c config.yml \
        --train-sets $input_dir/train.$src $input_dir/train.$tgt \
        --vocabs model/vocab.$src$tgt.spm model/vocab.$src$tgt.spm \
        --dim-vocabs $voc_sz $voc_sz \
        --valid-sets $input_dir/valid.$src $input_dir/valid.$tgt \
        --valid-translation-output "valid_output/epoch.{E}.$tgt" \
        --valid-script-args $src $tgt $dir \
        --devices $gpus
fi

# translate test sets
for split in valid test
do
    cat $input_dir/$split.$src \
        | $marian_decoder \
            -c model/model.npz.decoder.yml \
            --quiet-translation \
            -m model/model.npz.best-translation.npz \
            -d $gpus -b 12 -n -w 6000 \
        | sed 's/\@\@ //g' \
        | $MOSES_SCRIPTS/recaser/detruecase.perl \
        | $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt \
        > $output_dir/$split.$tgt
done

# calculate bleu scores on test sets
echo "Starting epoch 0"
cat $output_dir/test.$tgt | sacrebleu $dir/splitted/test.$tgt
# LC_ALL=C.UTF-8 sacrebleu -t wmt20/robust/set1 -l $src-$tgt < $output_dir/test.$tgt
