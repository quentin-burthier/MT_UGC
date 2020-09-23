#!/bin/bash -v

# suffix of source language files
SRC=en
# suffix of target language files
TRG=fr

# MTNT and processed data paths
MTNT=$DATA/MTNT
DIR=$MTNT/$SRC.$TGT

MARIAN_TRAIN=$MARIAN/marian
MARIAN_DECODER=$MARIAN/marian-decoder
MARIAN_VOCAB=$MARIAN/marian-vocab
MARIAN_SCORER=$MARIAN/marian-scorer

# set chosen gpus
GPUS=0
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS

if [ ! -e $MARIAN_TRAIN ]
then
    echo "marian is not installed in $MARIAN, you need to compile the toolkit first"
    exit 1
fi

if [ ! -e $TOOLS/moses-scripts ] || [ ! -e $TOOLS/subword-nmt ] || [ ! -e $TOOLS/sacreBLEU ]
then
    echo "missing tools in $TOOLS, you need to download them first"
    exit 1
fi

if [ ! -e "$MTNT" ]
then
    ./scripts/download-files.sh
fi

mkdir -p model

# preprocess data
if [ ! -e "$DIR" ]
then
    ./scripts/preprocess-data.sh
fi

# create common vocabulary
if [ ! -e "model/vocab.$SRC$TGT.yml" ]
then
    cat $DIR/bpe/train.$SRC $DIR/bpe/train.$TGT | $MARIAN_VOCAB --max-size 36000 > model/vocab.$SRC$TGT.yml
fi

# train model
if [ ! -e "model/model.npz" ]
then
    $MARIAN_TRAIN \
        --model model/model.npz --type transformer \
        --train-sets $DIR/bpe/train.$SRC $DIR/bpe/train.$TGT \
        --max-length 100 \
        --vocabs model/vocab.$SRC$TGT.yml model/vocab.$SRC$TGT.yml \
        --mini-batch-fit -w 10000 --maxi-batch 1000 \
        --early-stopping 10 --cost-type=ce-mean-words \
        --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-sets $DIR/bpe/valid.$SRC $DIR/bpe/valid.$TGT \
        --valid-script-path "bash ./scripts/validate.sh" \
        --valid-translation-output $DIR/bpe/valid.$SRC.output --quiet-translation \
        --valid-mini-batch 64 \
        --beam-size 6 --normalize 0.6 \
        --log model/train.log --valid-log model/valid.log \
        --enc-depth 6 --dec-depth 6 \
        --transformer-heads 8 \
        --transformer-postprocess-emb d \
        --transformer-postprocess dan \
        --transformer-dropout 0.1 --label-smoothing 0.1 \
        --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
        --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
        --tied-embeddings-all \
        --devices $GPUS --sync-sgd --seed 1111 \
        --exponential-smoothing
fi

# find best model on dev set
ITER=`cat model/valid.log | grep translation | sort -rg -k12,12 -t' ' | cut -f8 -d' ' | head -n1`

# translate test sets
for split in valid test
do
    cat $DIR/$bpe/$split.$SRC \
        | $MARIAN_DECODER -c model/model.npz.decoder.yml -m model/model.iter$ITER.npz -d $GPUS -b 12 -n -w 6000 \
        | sed 's/\@\@ //g' \
        | $TOOLS/moses-scripts/scripts/recaser/detruecase.perl \
        | $TOOLS/moses-scripts/scripts/tokenizer/detokenizer.perl -l $TGT \
        > $DIR/output/$split.$TGT.output
done

# calculate bleu scores on test sets
LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt14 -l $SRC-$TGT < $DIR/output/test.$TGT
