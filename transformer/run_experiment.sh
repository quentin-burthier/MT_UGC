#!/bin/bash

source $TOOLS/parse_cli.sh

# Default CLI args
src=en
tgt=fr
voc_sz=32000

model_dir=model

parse_cli $@

marian_train=$MARIAN/marian
marian_decoder=$MARIAN/marian-decoder

case $dataset in
    MTNT)
        if $shuffle; then
            mtnt=$DATA/MTNT_reshuffled
        else
            mtnt=$DATA/MTNT
        fi
        dir=$mtnt/$src-$tgt.$ratio
        preprocess_args="$src $tgt $dir $mtnt $ratio"
    ;;
    europarl_nc)
        dir=$DATA/europarl_nc.$src-$tgt
        dataset_args=$tgt
        preprocess_args="$src $tgt $dir"
    ;;
    nce_small)
        dir=$DATA/nce_small.$src-$tgt
        dataset_args=$tgt
        dataset=europarl_nc
        preprocess_args="$src $tgt $dir"
    ;;
    OpenSubtitles)
        dir=$DATA/OpenSubtitles.$src-$tgt
        dataset_args=$tgt
        dataset=europarl_nc
        preprocess_args="$src $tgt $dir"
    ;;
esac
formated_date=$(date +"%d.%m.%Y_%T")
if [ ! "$output_dir" ]; then output_dir=$dir/output_$formated_date; fi
if [ ! "$val_output_dir" ]; then val_output_dir=$dir/val_output_$formated_date; fi

input_dir=$dir/preprocessed
if [ ! -e "$input_dir" ]
then
    ./scripts/preprocess_$dataset.sh $preprocess_args
    echo "Preprocessing done"
    echo ""
fi

echo ": Up. 0 :"
echo "n_lines: $(wc -l $dir/raw/train.$src | cut -d" " -f1)"
# python $TOOLS/compare_lexicons.py $dir/raw/{train,dev}.$src

# train model
if [ ! -e "$model_dir/model.npz" ]
then
    mkdir -p $model_dir
    mkdir -p $val_output_dir
    mkdir -p log
    $marian_train \
        -c config.yml \
        -m $model_dir/model.npz \
        --train-sets $input_dir/train.{$src,$tgt} \
        --valid-sets $input_dir/val.{$src,$tgt} \
        --valid-translation-output "$val_output_dir/epoch.{E}.$tgt" \
        --valid-script-args $tgt $dir/raw/val.$tgt \
        --vocabs $model_dir/vocab.$src$tgt.spm $model_dir/vocab.$src$tgt.spm \
        --dim-vocabs $voc_sz $voc_sz \
        --devices $gpus
fi

# translate dev sets
mkdir -p $output_dir
for split in val dev
do
    cat $input_dir/$split.$src \
        | $marian_decoder \
            -c $model_dir/model.npz.decoder.yml \
            -m $model_dir/model.npz.best-translation.npz \
            -d $gpus -b 12 -n -w 6000 \
            --quiet-translation --quiet \
        | sed 's/\@\@ //g' \
        | $MOSES_SCRIPTS/recaser/detruecase.perl \
        | $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $tgt \
        > $output_dir/$split.$tgt
done

# calculate bleu scores on dev set
echo ": Up. 0 :"
cat $output_dir/dev.$tgt | sacrebleu $dir/raw/dev.$tgt

lang=en
train=europarl_nc.en-fr 
# dev=$train
dev=MTNT_reshuffled/en-fr.1.0
python $TOOLS/compare_lexicons.py $DATA/$train/preprocessed/train.$lang $DATA/$dev/preprocessed/dev.$lang
