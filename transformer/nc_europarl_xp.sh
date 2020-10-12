#!/bin/bash

source $TOOLS/parse_cli.sh

# Default CLI args
src=en
tgt=fr
voc_sz=32000

parse_cli $@

marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder

dir=$DATA/europarl_nc.$src-$tgt
input_dir=$dir/preprocessed
if [ ! -e "$input_dir" ]
then
    ./scripts/preprocess_nc_europarl.sh $tgt
    echo "Preprocessing done"
    echo ""
fi

echo ": Up. 0 :"
echo "n_lines: $(wc -l $dir/raw/train.$src | cut -d" " -f1)"
python $TOOLS/compare_lexicons.py $dir/raw/{train,dev}.$src

# train model
# output_dir=$dir/output
# output_dir=$dir/output_$voc_sz
train_date=$(date +"%d.%m.%Y_%T")
output_dir=$dir/output_$train_date
valid_output_dir=$dir/valid_output_$train_date
model_dir=model
mkdir -p $output_dir
mkdir -p $valid_output_dir
mkdir -p $model_dir
if [ ! -e "$model_dir/model.npz" ]
then
    $marian_train -c config.yml \
        --train-sets $input_dir/train.{$src,$tgt} \
        --valid-sets $input_dir/val.{$src,$tgt} \
        --valid-translation-output "$valid_output_dir/epoch.{E}.$tgt" \
        --valid-script-args $tgt $dir/raw/val.$tgt \
        --vocabs $model_dir/vocab.$src$tgt.spm $model_dir/vocab.$src$tgt.spm \
        --dim-vocabs $voc_sz $voc_sz \
        --devices $gpus
fi

# translate dev sets
for split in val dev
do
    cat $input_dir/$split.$src \
        | $marian_decoder \
            -c $model_dir/model.npz.decoder.yml \
            -m $model_dir/model.npz.best-translation.npz \
            -d $gpus -b 12 -n -w 6000 \
            --quiet-translation \
        | sed 's/\@\@ //g' \
        | $MOSES_SCRIPTS/recaser/detruecase.perl \
        | $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt \
        > $output_dir/$split.$tgt
done

# calculate bleu scores on dev set
echo ": Up. 0 :"
cat $output_dir/dev.$tgt | sacrebleu $dir/raw/dev.$tgt
