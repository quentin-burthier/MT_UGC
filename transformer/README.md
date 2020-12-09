# Robustness experiments


## Example usages

```bash
./run_experiment.sh -f marian \
    -s en -t fr --dataset MTNT \
    --nwordssrc 8000 --nwordstgt 8000 \
    -arch transformer -m $HOME/models/TrbOPS.fr-en \
    -btm Trb.marian.MTNT.fr-en --gpus 0 1 2 3 \
    -o output -vo val_output
```

With guild.ai installed

```bash
guild run --label MTNT.en-fr \
    framework=marian \
    src=en tgt=fr dataset=MTNT \
    nwordssrc=8000 nwordstgt=8000 \
    arch=transformer model=$HOME/models/TrbOPS.fr-en \
    gpus="0 1 2 3"
```
