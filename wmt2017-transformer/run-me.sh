#!/bin/bash -v

SRC=en
TGT=de

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


WORKSPACE=9500
N=4
EPOCHS=8
B=12

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

mkdir -p model

# preprocess data
if [ ! -e "$DATA/train/train.bpe.$SRC" ]
then
    LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt16 -l $SRC.$TGT --echo src > $DATA/valid.$SRC
    LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt16 -l $SRC.$TGT --echo ref > $DATA/valid.$TGT

    LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt14 -l $SRC.$TGT --echo src > $DATA/test2014.$SRC
    LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt15 -l $SRC.$TGT --echo src > $DATA/test2015.$SRC
    LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt16 -l $SRC.$TGT --echo src > $DATA/test2016.$SRC
    LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt17 -l $SRC.$TGT --echo src > $DATA/test2017.$SRC

    ./scripts/preprocess-data.sh
fi

if [ ! -e "$DATA/monolingual/dev.$TGT" ]
then
    ./scripts/download-files-mono.sh
fi

if [ ! -e "$DATA/monolingual/dev.$TGT" ]
then
    ./scripts/preprocess-data-mono.sh
fi

# create common vocabulary
if [ ! -e "model/vocab.$SRC$TGT.yml" ]
then
    cat $DATA/train/train.bpe.$SRC $DATA/train/train.bpe.$TGT | $MARIAN_VOCAB --max-size 36000 > model/vocab.$SRC$TGT.yml
fi

# train model
mkdir -p model.back
if [ ! -e "model.back/model.npz.best-translation.npz" ]
then
    $MARIAN_TRAIN \
        --model model.back/model.npz --type s2s \
        --train-sets $DATA/train/train.bpe.$TGT $DATA/train/train.bpe.$SRC \
        --max-length 100 \
        --vocabs model/vocab.$SRC$TGT.yml model/vocab.$SRC$TGT.yml \
        --mini-batch-fit -w 3500 --maxi-batch 1000 \
        --valid-freq 10000 --save-freq 10000 --disp-freq 1000 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-script-path "bash ./scripts/validate.$SRC.sh" \
        --valid-translation-output $DATA/valid.bpe.$TGT.output --quiet-translation \
        --valid-sets $DATA/valid.bpe.$TGT $DATA/valid.bpe.$SRC \
        --valid-mini-batch 64 --beam-size 12 --normalize=1 \
        --overwrite --keep-best \
        --early-stopping 5 --after-epochs 10 --cost-type=ce-mean-words \
        --log model.back/train.log --valid-log model.back/valid.log \
        --tied-embeddings-all --layer-normalization \
        --devices $GPUS --seed 1111 \
        --exponential-smoothing
fi

if [ ! -e "$DATA/monolingual/dev.bpe.$SRC" ]
then
    $MARIAN_DECODER \
      -c model.back/model.npz.best-translation.npz.$TGTcoder.yml \
      -i $DATA/monolingual/dev.bpe.$TGT \
      -b 6 --normalize=1 -w 2500 -d $GPUS \
      --mini-batch 64 --maxi-batch 100 --maxi-batch-sort src \
      --max-length 200 --max-length-crop \
      > $DATA/monolingual/dev.bpe.$SRC
fi

if [ ! -e "$DATA/all.bpe.$SRC" ]
then
    cat $DATA/train/train.bpe.$SRC $DATA/train/train.bpe.$SRC $DATA/monolingual/dev.bpe.$SRC > $DATA/all.bpe.$SRC
    cat $DATA/train/train.bpe.$TGT $DATA/train/train.bpe.$TGT $DATA/monolingual/dev.bpe.$TGT > $DATA/all.bpe.$TGT
fi

for i in $(seq 1 $N)
do
  mkdir -p model/ens$i
  # train model
    $MARIAN_TRAIN \
        --model model/ens$i/model.npz --type transformer \
        --train-sets $DATA/all.bpe.$SRC $DATA/all.bpe.$TGT \
        --max-length 100 \
        --vocabs model/vocab.$SRC$TGT.yml model/vocab.$SRC$TGT.yml \
        --mini-batch-fit -w $WORKSPACE --mini-batch 1000 --maxi-batch 1000 \
        --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-sets $DATA/valid.bpe.$SRC $DATA/valid.bpe.$TGT \
        --valid-script-path "bash ./scripts/validate.sh" \
        --valid-translation-output $DATA/valid.bpe.$SRC.output --quiet-translation \
        --beam-size 12 --normalize=1 \
        --valid-mini-batch 64 \
        --overwrite --keep-best \
        --early-stopping 5 --after-epochs $EPOCHS --cost-type=ce-mean-words \
        --log model/ens$i/train.log --valid-log model/ens$i/valid.log \
        --enc-depth 6 --dec-depth 6 \
        --tied-embeddings-all \
        --transformer-dropout 0.1 --label-smoothing 0.1 \
        --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
        --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
        --devices $GPUS --sync-sgd --seed $i$i$i$i  \
        --exponential-smoothing
done

for i in $(seq 1 $N)
do
  mkdir -p model/ens-rtl$i
  # train model
    $MARIAN_TRAIN \
        --model model/ens-rtl$i/model.npz --type transformer \
        --train-sets $DATA/all.bpe.$SRC $DATA/all.bpe.$TGT \
        --max-length 100 \
        --vocabs model/vocab.$SRC$TGT.yml model/vocab.$SRC$TGT.yml \
        --mini-batch-fit -w $WORKSPACE --mini-batch 1000 --maxi-batch 1000 \
        --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-sets $DATA/valid.bpe.$SRC $DATA/valid.bpe.$TGT \
        --valid-script-path  "bash ./scripts/validate.sh" \
        --valid-translation-output $DATA/valid.bpe.$SRC.output --quiet-translation \
        --beam-size 12 --normalize=1 \
        --valid-mini-batch 64 \
        --overwrite --keep-best \
        --early-stopping 5 --after-epochs $EPOCHS --cost-type=ce-mean-words \
        --log model/ens-rtl$i/train.log --valid-log model/ens-rtl$i/valid.log \
        --enc-depth 6 --dec-depth 6 \
        --tied-embeddings-all \
        --transformer-dropout 0.1 --label-smoothing 0.1 \
        --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
        --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
        --devices $GPUS --sync-sgd --seed $i$i$i$i$i \
        --exponential-smoothing --right-left
done

# translate test sets
for split in valid test
do
    cat $DATA/$split/$split.bpe.$SRC \
        | $MARIAN_DECODER -c model/ens1/model.npz.best-translation.npz.decoder.yml \
          -m model/ens?/model.npz.best-translation.npz -d $GPUS \
          --mini-batch 16 --maxi-batch 100 --maxi-batch-sort src -w 5000 --n-best --beam-size $B \
        > $DATA/$split/$split.bpe.$SRC.output.nbest.0

    for i in $(seq 1 $N)
    do
      $MARIAN_SCORER -m model/ens-rtl$i/model.npz.best-perplexity.npz \
        -v model/vocab.$SRC$TGT.yml model/vocab.$SRC$TGT.yml -d $GPUS \
        --mini-batch 16 --maxi-batch 100 --maxi-batch-sort trg --n-best --n-best-feature R2L$(expr $i - 1) \
        -t $DATA/$split/$split.bpe.$SRC $DATA/$split/$split.bpe.$SRC.output.nbest.$(expr $i - 1) > $DATA/$split/$split.bpe.$SRC.output.nbest.$i
    done

    cat $DATA/$split/$split.bpe.$SRC.output.nbest.$N \
      | python scripts/rescore.py \
      | perl -pe 's/@@ //g' \
      | $TOOLS/moses-scripts/scripts/recaser/detruecase.perl \
      | $TOOLS/moses-scripts/scripts/tokenizer/detokenizer.perl > $DATA/$split/$split.$SRC.output
done

# calculate bleu scores on test sets
LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt16 -l $SRC.$TGT < $DATA/valid/valid.$SRC.output
LC_ALL=C.UTF-8 $TOOLS/sacreBLEU/sacrebleu.py -t wmt14 -l $SRC.$TGT < $DATA/test/test.$SRC.output
