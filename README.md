# Robustness benchmark of Machine Translation methods

This repository contains scripts to perform Neural Machine Translation (NMT) experiments.

NMT models can be trained and evaluated on
  - Europarl (v7)
  - OpenSubtitles (v2018)
  - News-Commentary
  - MTNT
  - Foursquare

and evaluated (but not trained) on
  - PFSMB (aka Cr#pbank).

OAR scripts are also available in the `oar` branch.
## General settings and depencies

Python 3.6 with packages
  - sacrebleu 1.14.4
  - sentencepiece 0.1.91
  - fastText
  - nltk 3.5
  - pandas 1.1.5 (only required by `statitics/`)

Environment variables

```bash
export MOSES_SCRIPTS=$HOME/marian-dev/examples/tools/moses-scripts/scripts
export TOOLS=$HOME/robust_bench/tools
export DATA=/data/almanach/user/$(whoami)
```

or in a conda environement (see
[documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables))

```bash
conda activate nmt
conda env config vars set MOSES_SCRIPTS=$HOME/marian-dev/examples/tools/moses-scripts/scripts
conda env config vars set TOOLS=$HOME/robust_bench/tools
conda env config vars set DATA=/data/almanach/user/$(whoami)

conda activate nmt
```

The copora are expected to be located in `$DATA/`.
## Marian

marian-dev (commit [467b15e](https://github.com/marian-nmt/marian-dev/commit/467b15e2b94b7c7b25ceaee764f790d8faaeabf2))

```bash
export MARIAN=$HOME/marian-dev/build
```

- gcc 7.3.0
- CUDA 9.2

## Fairseq

Commit c8a0659

- Pytorch 1.5.0
- fairseq (commit [c8a0659](https://github.com/pytorch/fairseq/commit/c8a0659))
- CUDA 11.0
- CUDNN 8.0
- cmake 3.10.1

```bash
export MKL_THREADING_LAYER=GNU
```

## Usage

From `src/` directory:

```bash
./run_experiment.sh 
```

Arguments:
  - `-f` or `--framework`: marian or fairseq
  - `-arch` or `--architecture`: transformer, convtransformer
  - `-s` or `--source`: en, fr
  - `-t` or `--target`: en, fr
  - `-sseg` or `--src_segmentation`: char, bpe
  - `-tseg` or `--tgt_segmentation`: char, bpe
  - `--joint-dictionary`: Uses the same dictionary for source and target.
  - `-nws` or `--nwordssrc`: Size of the source side vocabulary (ignored if src_segmentation is char).
  - `-nwt` or `--nwordstgt`: Size of the target side vocabulary (ignored if joint-vocabulary or if tgt_segmentation is char).
  - `-d` or `--dataset`: MTNT, News-Commentary, Europarl, Europarl_small, OpenSubtitles, OpenSubtitles_small, Crapbank, Foursquare
  - `--no-shuffle`: uses unshuffled MTNT training data (MTNT only).
  - `-r` or `--ratio`: uses a subsampled training data (MTNT only)
  - `-m` or `--model`: model path
  - `-ckpt` or `--checkpoint`: checkpoint to use (default: `checkpoint_best.pt`)
  - `-btm` or `--back-translation-model`
  - `-o` or `--output-dir`: path of the output translations of the development set
  - `-vo` or `--val-output-dir`: path of the output translations at each validation step
  - `--gpus`: gpus to use (only for marian, fairseq uses everything by default)

## Logging

[guild.ai](https://guild.ai/) can be used for logging experiments.
From `src/` directory:

```bash
guild run
```

## About the implementation

  - `tools/spm/` can be replaced with `spm` if sentencepiece was installed from source.
  - Reusing the sentencepiece vocabulary when using `marian` was not tested and may require to adapt the code.
  - 
